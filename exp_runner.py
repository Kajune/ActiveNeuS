import os, sys, time, logging, argparse, json, glob, random
import numpy as np
import cv2
import open3d as o3d
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, MediumNeRF, ProjectionNetwork, ShadowField, DeformNetwork, DeformNetworkSiren
from models.renderer import NeuSRenderer
from models.LLR import local_linear_reconstruction
from camera_visualization import plot_camera_scene
from pysdf import SDF
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy.spatial.transform import Rotation as Rot
from sklearn.neighbors import NearestNeighbors
from torch.profiler import profile, record_function, ProfilerActivity


def torch_fix_seed(seed=42):
	# Numpy
	np.random.seed(seed)
	# Pytorch
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms = True
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True


def recursive_overwrite(config, overwrite_config):
	for k, v in overwrite_config.items():
		if k in config and not isinstance(config[k], dict):
			config[k] = v
		elif k in config and isinstance(config[k], dict):
			config[k] = recursive_overwrite(config[k], v)
		else:
			print("[Warning] key %s not foundin source config!" % k)
			config[k] = v
	return config


def get_event_indices(indices, image_perm, event_mode):
	if event_mode == "sequential":
		indices_prev = max(indices - 1, 0)
		indices_next = indices

	elif event_mode == "accumulated":
		indices_prev = indices * 0
		indices_next = indices

	elif event_mode == "random":
		indices_prev = image_perm[torch.randint(len(image_perm), (1,))][0]
		indices_next = indices

	return indices_prev, indices_next


def remove_floaters(mesh, epsilon=0.01):
	components = mesh.split(only_watertight=False)
	total_volume = sum(component.volume for component in components)
	filtered_components = [component for component in components if component.volume / total_volume >= epsilon]
	merged_mesh = trimesh.util.concatenate(filtered_components)
	return merged_mesh


def is_json_serializable(value):
	try:
		json.dumps(value)
		return True
	except (TypeError, OverflowError, ValueError):
		return False
	return False


class Runner:
	def __init__(self, conf_path, mode='train', exp_name='default', case='CASE_NAME', 
				load=None, load_metadata=None, load_params=None, resume=False, freeze=[],
				estimate_cam_intrinsic=False, estimate_cam_pose=False, use_mlp_for_pose=False,
				estimate_last_cam_pose=False, cam_noise={}, mask_ratio=1.0, 
				estimate_proj_intrinsic=False, estimate_proj_pose=False, proj_noise={},
				initial_illum_params=None, estimate_illumination=False, pattern_noise={}, 
				estimate_cam_refraction=False, cam_refractive_interface_params=None,
				estimate_proj_refraction=False, proj_refractive_interface_params=None,
				frame_weights=False, num_images=None, num_images_policy='interval', num_images_incremental=0, 
				num_images_incremental_start=0, min_num_images=1,
				image_ind_offset=0, dynamic=False, end_iter=None, overwrite_params=None, 
				pretrain_sdf_network=False, initial_shape_type="sphere", initial_shape_mesh=None,
				event_mode=False, simulate_event=False, scene_scale=1.0, baseline_scale=1.0, profiling=None):
		self.device = torch.device('cuda')
		self.num_images = num_images
		self.num_images_policy = num_images_policy
		self.num_images_incremental = num_images_incremental
		self.num_images_incremental_start = num_images_incremental_start
		self.min_num_images = min_num_images
		self.current_num_images = self.num_images
		self.image_ind_offset = image_ind_offset
		self.dynamic = dynamic
		self.initial_illum_params = initial_illum_params
		self.estimate_illumination = estimate_illumination
		self.estimate_cam_intrinsic = estimate_cam_intrinsic
		self.estimate_cam_pose = estimate_cam_pose
		self.use_mlp_for_pose = use_mlp_for_pose
		self.estimate_last_cam_pose = estimate_last_cam_pose
		self.estimate_proj_intrinsic = estimate_proj_intrinsic
		self.estimate_proj_pose = estimate_proj_pose
		self.estimate_cam_refraction = estimate_cam_refraction
		self.estimate_proj_refraction = estimate_proj_refraction
		self.frame_weights = frame_weights
		self.load = load
		self.load_metadata = load_metadata
		self.load_params = load_params
		self.freeze = freeze
		self.event_mode = event_mode
		self.simulate_event = simulate_event
		self.mask_ratio = mask_ratio
		self.profiling = profiling

		# Configuration
		self.conf_path = conf_path
		f = open(self.conf_path)
		conf_text = f.read()
		f.close()

		self.conf = ConfigFactory.parse_string(conf_text)
		self.conf['general']['base_exp_dir'] = os.path.join("exp", case, exp_name)
		self.conf['dataset']['data_dir'] = os.path.join("public_data", case)
		if overwrite_params is not None:
			self.conf = recursive_overwrite(self.conf, overwrite_params)

		self.base_exp_dir = self.conf['general.base_exp_dir']
		os.makedirs(self.base_exp_dir, exist_ok=True)
		self.dataset = Dataset(self.conf['dataset'], cam_noise, proj_noise, pattern_noise, self.image_ind_offset,
							cam_refractive_interface_params, proj_refractive_interface_params, 
							event_mode=self.event_mode, simulate_event=self.simulate_event, 
							scene_scale=scene_scale, baseline_scale=baseline_scale, dynamic=self.dynamic,
							use_cam_cache=(not self.estimate_cam_pose) and (not self.estimate_last_cam_pose) and (not self.estimate_cam_intrinsic),
							use_proj_cache=(not self.estimate_proj_pose) and (not self.estimate_proj_intrinsic), 
							estimate_cam_intrinsic=self.estimate_cam_intrinsic, estimate_cam_pose=self.estimate_cam_pose, 
							estimate_proj_intrinsic=self.estimate_proj_intrinsic, estimate_proj_pose=self.estimate_proj_pose,
							use_mlp_for_pose=use_mlp_for_pose)
		self.resume = resume
		if self.resume and self.load is None:
			self.load = self.base_exp_dir

		if self.num_images is None:
			self.num_images = self.dataset.n_images
			self.current_num_images = self.num_images

		self.conf['argparse'] = {
			"num_images": self.num_images,
			"num_images_policy": self.num_images_policy,
			"num_images_incremental": self.num_images_incremental,
			"num_images_incremental_start": self.num_images_incremental_start,
			"min_num_images": self.min_num_images,
			"image_ind_offset": self.image_ind_offset,
			"dynamic": self.dynamic,
			"cam_noise": cam_noise,
			"proj_noise": proj_noise,
			"initial_illum_params": self.initial_illum_params,
			"estimate_illumination": self.estimate_illumination,
			"estimate_cam_intrinsic": self.estimate_cam_intrinsic,
			"estimate_cam_pose": self.estimate_cam_pose,
			"use_mlp_for_pose": self.use_mlp_for_pose,
			"estimate_last_cam_pose": self.estimate_last_cam_pose,
			"estimate_proj_intrinsic": self.estimate_proj_intrinsic,
			"estimate_proj_pose": self.estimate_proj_pose,
			"estimate_cam_refraction": self.estimate_cam_refraction,
			"estimate_proj_refraction": self.estimate_proj_refraction,
			"frame_weights": self.frame_weights,
			"load": self.load,
			"load_metadata": self.load_metadata,
			"load_params": self.load_params,
			"event_mode": self.event_mode,
			"simulate_event": self.simulate_event,
			"scene_scale": scene_scale,
			"baseline_scale": baseline_scale,
			"mask_ratio": self.mask_ratio,
		}

		os.makedirs(os.path.join(self.conf['general.base_exp_dir'], "recording"), exist_ok=True)
		with open(os.path.join(self.conf['general.base_exp_dir'], "recording", "config.json"), "w") as f:
			json.dump(self.conf, f, indent=2)

		# Training parameters
		self.end_iter = self.conf.get_int('train.end_iter') if end_iter is None else end_iter
		self.save_freq = self.conf.get_int('train.save_freq')
		self.report_freq = self.conf.get_int('train.report_freq')
		self.val_freq = self.conf.get_int('train.val_freq')
		self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
		self.val_shadow_freq = self.conf.get_int('train.val_shadow_freq', default=None)
		self.val_mesh_resolution = self.conf.get_int('train.mesh_resolution', default=1024)
		self.mesh_extract_scale = self.conf.get_float('train.mesh_extract_scale', default=1.0)
		self.batch_size = self.conf.get_int('train.batch_size')
		self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
		self.learning_rate = self.conf.get_float('train.learning_rate')
		self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
		self.adaptive_lr = self.conf.get_float('train.adaptive_lr', default=None)
		self.atten_end_iter = self.conf.get_int('train.atten_end_iter', default=self.end_iter)
		self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
		self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
		self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
		self.shadow_field_begin = self.conf.get_int('train.shadow_field_begin', default=0)
#		self.shadow_field_end = self.conf.get_int('train.shadow_field_end', default=0)
		self.shadow_field_ratio = self.conf.get_float('train.shadow_field_ratio', default=1)
		self.loss_type = self.conf.get_string('train.loss_type', default="l1")
		self.ray_sampling_mode = self.conf.get_string('train.ray_sampling_mode', default="sequential")
		assert self.ray_sampling_mode == "sequential", "ray_sampling_mode must be sequential now."
		self.multi_ray_sample_images = self.conf.get_int('train.multi_ray_sample_images', default=2)
		self.cam_param_warm_up_strategy = self.conf.get_string('train.cam_param_warm_up_strategy', default=None)

		# Weights
		self.color_weight = self.conf.get_float('train.color_weight', default=1.0)
		self.igr_weight_begin = self.conf.get_float('train.igr_weight_begin', default=0)
		self.igr_weight_end = self.conf.get_float('train.igr_weight_end', default=0)
		self.igr_loss_scale = self.conf.get_float('train.igr_loss_scale', default=None)
		self.curv_weight_begin = self.conf.get_float('train.curv_weight_begin', default=0)
		self.curv_weight_end = self.conf.get_float('train.curv_weight_end', default=0)
		self.curv_sample_ratio = self.conf.get_float('train.curv_sample_ratio', default=0.1)
		self.curv_loss_scale = self.conf.get_float('train.curv_loss_scale', default=None)
		self.temporal_smoothness_weight = self.conf.get_float('train.temporal_smoothness_weight', default=0)
		self.mask_weight_begin = self.conf.get_float('train.mask_weight_begin', default=0)
		self.mask_weight_end = self.conf.get_float('train.mask_weight_end', default=0)
		self.depth_weight_pos_begin = self.conf.get_float('train.depth_weight_pos_begin', default=0)
		self.depth_weight_neg_begin = self.conf.get_float('train.depth_weight_neg_begin', default=0)
		self.depth_weight_pos_end = self.conf.get_float('train.depth_weight_pos_end', default=0)
		self.depth_weight_neg_end = self.conf.get_float('train.depth_weight_neg_end', default=0)
		self.aruco_weight_begin = self.conf.get_float('train.aruco_weight_begin', default=0)
		self.aruco_weight_end = self.conf.get_float('train.aruco_weight_end', default=0)
		self.sparsity_reg_weight_begin = self.conf.get_float('train.sparsity_reg_weight_begin', default=0)
		self.sparsity_reg_weight_end = self.conf.get_float('train.sparsity_reg_weight_end', default=0)
		self.unimodality_loss_weight_begin = self.conf.get_float('train.unimodality_loss_weight_begin', default=0)
		self.unimodality_loss_weight_end = self.conf.get_float('train.unimodality_loss_weight_end', default=0)
		self.smoothness_weight = self.conf.get_float('train.smoothness_weight', default=0)
		self.density_L1_weight = self.conf.get_float('train.density_L1_weight', default=0)
		self.shadow_field_weight = self.conf.get_float('train.shadow_field_weight', default=0)
		self.reverse_rendering_weight_begin = self.conf.get_float('train.reverse_rendering_weight_begin', default=0)
		self.reverse_rendering_weight_end = self.conf.get_float('train.reverse_rendering_weight_end', default=0)
		self.llr_weight = self.conf.get_float('train.llr_weight', default=0)

		self.mode = mode
		self.model_list = []
		self.writer = None

		self.pretrain_sdf_network = pretrain_sdf_network
		self.initial_shape_type = initial_shape_type
		self.initial_shape_mesh = initial_shape_mesh
		self.initialize_network()

		if self.depth_weight_pos_begin > 0 or self.depth_weight_pos_end > 0 or \
			self.depth_weight_neg_begin > 0 or self.depth_weight_neg_end > 0:
			point_cloud = trimesh.load(os.path.join(os.path.join("public_data", case, "points3d.ply")))
			point_cloud.apply_transform(np.linalg.inv(self.dataset.scale_mat.cpu().detach().numpy()))
			vertices = point_cloud.vertices
			indices = np.random.choice(np.arange(len(vertices)), 10000, replace=False)
			self.point_cloud_pos = torch.from_numpy(vertices[indices]).to(self.device).float()

		self.point_cloud_neg = torch.from_numpy(np.random.uniform(-1,1,(10000,3))).float().to(self.device)
		if self.depth_weight_neg_begin > 0 or self.depth_weight_neg_end > 0:
			knn = NearestNeighbors(n_neighbors=1)
			knn.fit(self.point_cloud_pos.cpu().numpy())
			distances, indices = knn.kneighbors(self.point_cloud_neg.cpu().numpy())
			distances = distances.flatten()
			self.point_cloud_neg_distances = torch.from_numpy(distances).to(self.device).float()

		# Backup codes and configs for debug
		if self.mode[:5] == 'train':
			self.file_backup()

		self.image_perm = self.get_image_perm()


	def initialize_network(self):
		self.iter_step = 0

		# Networks
		params_to_train = []
		if 'allow_volume_scattering' in self.conf['model.neus_renderer'] and self.conf['model.neus_renderer']['allow_volume_scattering']:
			self.nerf_outside = MediumNeRF(**self.conf['model.nerf']).to(self.device)
		else:
			self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
		self.sdf_network = SDFNetwork(self.dataset.n_images, **self.conf['model.sdf_network']).to(self.device)

		self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
		self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
		self.projection_network = None
		self.shadow_field = None
		self.deform_network = None
		if "nerf_outside" not in self.freeze:
			params_to_train += list(self.nerf_outside.parameters())
		if "sdf_network" not in self.freeze:
			params_to_train += list(self.sdf_network.parameters())
		if "deviation_network" not in self.freeze:
			params_to_train += list(self.deviation_network.parameters())
		if "color_network" not in self.freeze:
			params_to_train += list(self.color_network.parameters())

		if self.dataset.with_projection:
			params_to_train += list(self.dataset.projection_pattern.parameters())

			if self.estimate_illumination:
				params_to_train += list(self.dataset.illumination_params.parameters())
	
			if self.initial_illum_params is not None:
				nn.init.constant_(self.dataset.illumination_params.ambient, self.initial_illum_params["ambient"])
				nn.init.constant_(self.dataset.illumination_params.diffuse_denom, 1 / self.initial_illum_params["diffuse"])
				nn.init.constant_(self.dataset.illumination_params.emissive_denom, 1 / self.initial_illum_params["emissive"])

			if self.estimate_proj_pose or self.estimate_proj_intrinsic:
				self.conf['model.projection_network']['no_grad'] = False
				print("Overwriting model.projection_network.no_grad to estimate projector poses.")

			self.projection_network = ProjectionNetwork(**self.conf['model.projection_network'], n_images=self.dataset.n_images).to(self.device)
			params_to_train += list(self.projection_network.parameters())

			if 'model.shadow_field' in self.conf:
				self.shadow_field = ShadowField(**self.conf['model.shadow_field']).to(self.device)
				params_to_train += list(self.shadow_field.parameters())

		if 'model.deform_network' in self.conf:
			self.deform_network = DeformNetwork(**self.conf['model.deform_network']).to(self.device)
			params_to_train += list(self.deform_network.parameters())

		if self.estimate_cam_pose or self.estimate_last_cam_pose or self.estimate_cam_intrinsic:
			params_to_train += list(self.dataset.camera_params.parameters())
		if self.estimate_proj_pose or self.estimate_proj_intrinsic:
			params_to_train += list(self.dataset.projector_params.parameters())
		if self.dataset.with_cam_refraction and self.estimate_cam_refraction:
			params_to_train += list(self.dataset.cam_refractive_interface.parameters())
		if self.dataset.with_proj_refraction and self.estimate_proj_refraction:
			params_to_train += list(self.dataset.proj_refractive_interface.parameters())
		if self.frame_weights:
			params_to_train += list(self.dataset.frame_weights.parameters())

		self.optimizer = torch.optim.AdamW(params_to_train, lr=self.learning_rate)

		self.renderer = NeuSRenderer(self.dataset.n_images,
									 self.nerf_outside,
									 self.sdf_network,
									 self.deviation_network,
									 self.color_network,
									 self.projection_network,
									 self.shadow_field,
									 self.deform_network,
									 **self.conf['model.neus_renderer'])

		# Load checkpoint
		latest_model_name = None
		if self.load is not None:
			latest_model_name_list = sorted(glob.glob(os.path.join(self.load, 'checkpoints', '*.pth')))
			if len(latest_model_name_list) > 0:
				latest_model_name = latest_model_name_list[-1]
			else:
				print("checkpoint not found!")

		if latest_model_name is not None:
			logging.info('Find checkpoint: {}'.format(latest_model_name))
			self.load_checkpoint(latest_model_name)

		latest_model_name = None
		if self.load_metadata is not None:
			latest_model_name_list = sorted(glob.glob(os.path.join(self.load_metadata, 'checkpoints', '*.pth')))
			if len(latest_model_name_list) > 0:
				latest_model_name = latest_model_name_list[-1]
			else:
				print("checkpoint not found!")

		if latest_model_name is not None:
			logging.info('Find checkpoint: {}'.format(latest_model_name))
			self.load_checkpoint(latest_model_name, metadata_only=True)

		latest_model_name = None
		if self.load_params is not None:
			latest_model_name_list = sorted(glob.glob(os.path.join(self.load_params, 'checkpoints', '*.pth')))
			if len(latest_model_name_list) > 0:
				latest_model_name = latest_model_name_list[-1]
			else:
				print("checkpoint not found!")

		if latest_model_name is not None:
			logging.info('Find checkpoint: {}'.format(latest_model_name))
			self.load_checkpoint(latest_model_name, params_only=True)

		if self.iter_step == 0 and self.pretrain_sdf_network:
			self.pretrain()


	def event_camera_model(self, img_prev, img_next):
		is_numpy = isinstance(img_prev, np.ndarray)
		if is_numpy:
			img_prev = torch.from_numpy(img_prev)
			img_next = torch.from_numpy(img_next)

		img_prev = img_prev.float()
		img_next = img_next.float()

		out = ((img_next.mean(dim=-1, keepdim=True).clamp(0, 1)
		        - img_prev.mean(dim=-1, keepdim=True).clamp(0, 1))
		       * self.dataset.event_scale + self.dataset.event_offset).clamp(0, 1)

		if is_numpy:
			out = out.numpy()

		return out


	def render_random_rays_impl(self, indices, curv_weight, pixels=None, mode="cam"):
		if self.ray_sampling_mode == "sequential":
			data = self.dataset.gen_random_rays_at(indices, self.batch_size * (2 if self.iter_step < self.shadow_field_begin else 1), pixels=pixels, mode=mode)
		else:
			indices, data = self.dataset.gen_random_rays_multi(indices, self.batch_size * (2 if self.iter_step < self.shadow_field_begin else 1), pixels=pixels, mode=mode)

		if not (self.estimate_proj_pose or self.estimate_proj_intrinsic or self.estimate_cam_pose or self.estimate_cam_intrinsic) or \
			(self.estimate_last_cam_pose and indices != self.current_num_images - 1) or indices == self.image_ind_offset:
			data = data.detach()
		rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
		near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

		background_rgb = None
		if self.use_white_bkgd:
			background_rgb = torch.ones([1, 3])

#		if self.mask_weight > 0.0:
#			mask = (mask > 0.5).float()
#		else:
#			mask = torch.ones_like(mask)

		mask = torch.ones_like(mask) * (1 - self.mask_ratio) + mask * self.mask_ratio

		if self.dataset.with_projection:
			proj_params = self.dataset.get_proj_params(indices, mode)
			illum_params = self.dataset.illumination_params(indices)
		else:
			proj_params = []
			illum_params = None

		render_out = self.renderer.render(rays_o, rays_d, indices - self.image_ind_offset, near, far,
										  background_rgb=background_rgb,
										  cos_anneal_ratio=self.get_cos_anneal_ratio(),
										  proj_params=proj_params,
										  illum_params=illum_params,
										  compute_curvature_loss=curv_weight>0 and np.random.rand() < self.curv_sample_ratio,
										  compute_temporal_smoothness_error=self.temporal_smoothness_weight>0,
										  shadow_field_ratio=self.shadow_field_ratio,
										  disable_shadow=self.iter_step<self.shadow_field_begin,
										  mode=mode)
		return render_out, true_rgb, mask


	def render_random_rays(self, indices, curv_weight, mode='cam', return_pixels=False):
		pixels = self.dataset.gen_random_pixels(self.batch_size * (2 if self.iter_step < self.shadow_field_begin else 1), mode=mode)

		if self.event_mode is not None:
			indices_prev, indices_next = get_event_indices(indices, self.image_perm, self.event_mode)
			render_out_next, true_event_next, mask_next = self.render_random_rays_impl(indices_next, curv_weight, pixels=pixels, mode=mode)
			render_out_prev, true_event_prev, mask_prev = self.render_random_rays_impl(indices_prev, 0,           pixels=pixels, mode=mode)
			render_out_next['color_fine'] = self.event_camera_model(render_out_prev['color_fine'], render_out_next['color_fine'])
			if self.event_mode == "sequential":
				true_event = true_event_next
			else:
				true_event = self.event_camera_model(true_event_prev, true_event_next)

			"""
			valid_pixels = mask_next & mask_prev
			depth_next = render_out_next['depth'][valid_pixels]
			depth_prev = render_out_prev['depth'][valid_pixels]
			delta_depth = depth_next - depth_prev

			flow_speed = []
			for next_coord, prev_coord in zip(render_out_next['reprojection_coords'], render_out_prev['reprojection_coords']):
				flow_speed.append(torch.norm(next_coord - prev_coord, p=2))
			"""

			ret = (render_out_next, true_event.mean(dim=-1, keepdim=True), mask_next.mean(dim=-1, keepdim=True))

		else:
			ret = self.render_random_rays_impl(indices, curv_weight, mode=mode, pixels=pixels)

		if return_pixels:
			ret = (*ret, pixels)

		return ret


	def pretrain(self):
		# Make initial shape
		if self.initial_shape_type == "plane":
			cam_poses = torch.stack([self.dataset.camera_params.get_pose(i) for i in range(0,self.dataset.n_images)])
			tvecs = cam_poses[:,:3,3]
			min_z = tvecs[...,2].min()
			margin = 0.1
			plane_z = (min_z - margin).cpu().detach().numpy()

			trans = np.eye(4)
			trans[2,3] = -(1 - (plane_z + 1) / 2)
			initial_shape = trimesh.creation.box(extents=[2, 2, plane_z + 1])
			initial_shape = initial_shape.apply_transform(trans)

		elif self.initial_shape_type == "tunnel":
			cam_poses = torch.stack([self.dataset.camera_params.get_pose(i) for i in range(0,self.dataset.n_images)])
			tvecs = cam_poses[:,:3,3]

			initial_shape = trimesh.creation.box(extents=[2, 2, 2])
			tunnel = [trimesh.creation.icosphere(subdivisions=3, radius=0.1).apply_translation(tvec.cpu().detach().numpy()) for tvec in tvecs]
			initial_shape = trimesh.boolean.difference([initial_shape] + tunnel)

		elif self.initial_shape_type == "sphere":
			initial_shape = trimesh.creation.icosphere(subdivisions=3, radius=0.5)

		elif self.initial_shape_type == "box":
			initial_shape = trimesh.creation.box(extents=[0.5, 0.5, 0.5])

		elif self.initial_shape_type == "mesh":
			assert self.initial_shape_mesh is not None
			initial_shape = trimesh.load(self.initial_shape_mesh)
			initial_shape.apply_transform(np.linalg.inv(self.dataset.scale_mat.cpu().detach().numpy()))

		else:
			print("Unknown initial shape type:", self.initial_shape_type)
			exit()

		os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
		initial_shape.export(os.path.join(self.base_exp_dir, 'meshes', 'initial_shape.ply'))

		sdf_fn = SDF(initial_shape.vertices, initial_shape.faces)
		self.image_perm = self.get_image_perm()

		# Pretrain SDF network
		self.sdf_network.train()
		optimizer = torch.optim.AdamW(self.sdf_network.parameters(), lr=1e-3)
		for i in tqdm(range(5000)):
#			inputs = (torch.rand(100000, 3) * 2 - 1) * 1.5
#			inputs = inputs.to(self.device)

			if self.ray_sampling_mode == "sequential":
				image_indices = self.image_perm[self.iter_step % len(self.image_perm)]
			elif self.ray_sampling_mode == "random":
				image_indices = torch.stack([self.image_perm[i] for i in np.random.randint(0, len(self.image_perm), size=(self.multi_ray_sample_images,))])
			elif self.ray_sampling_mode == "consecutive":
				img_idx = np.random.randint(0, self.dataset.n_images - self.multi_ray_sample_images)
				image_indices = torch.from_numpy(np.int64([img_idx + i for i in range(self.multi_ray_sample_images)])).to(self.image_perm)
			elif self.ray_sampling_mode == "all":
				image_indices = self.image_perm
			else:
				raise NotImplementedError

			ret_fine, true_rgb, mask = self.render_random_rays(image_indices, 0.0)

			mask_sum = mask.sum() + 1e-5

			color_fine = ret_fine['color_fine']
			color_fine_loss = self.color_error(color_fine, true_rgb, mask, mask_sum).sum()

			gt_sdf = torch.from_numpy(-sdf_fn(ret_fine['pts'].cpu().detach().numpy())).to(self.device)
			sdf_error = F.l1_loss(ret_fine['sdf'], gt_sdf[...,None].detach())
			gradient_error = ret_fine['gradient_error']

			loss = color_fine_loss + sdf_error + gradient_error * self.igr_weight_begin

			if i % 1000 == 0:
				print("Pretrain loss: %f, color: %f, sdf: %f, grad: %f" % (loss.item(), color_fine_loss.item(), sdf_error.item(), gradient_error.item()))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		self.validate_mesh(time=self.image_ind_offset)
		self.validate_image()
		self.visualize_sdf(os.path.join(self.base_exp_dir, "meshes", "sdf"), resolution=256)


	def color_error(self, color_fine, true_rgb, mask, mask_sum):
		if color_fine.shape[-1] == 1:
			true_rgb = true_rgb.mean(dim=-1, keepdim=True)
		if self.loss_type == "l1":
			color_error = (color_fine - true_rgb) * mask
			color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='none') / mask_sum
		elif self.loss_type == "mse":
			color_fine = torch.clip(color_fine, max=1)
			color_fine_loss = ((color_fine - true_rgb) ** 2 * mask / mask_sum)
		elif self.loss_type == "scaledmse":
			color_fine = torch.clip(color_fine, max=1)
			color_fine_loss = (((color_fine - true_rgb) / (color_fine.detach() + 1e-3)) ** 2 * mask / mask_sum)
		elif self.loss_type == "ssim":
			color_fine_loss = 1 - ms_ssim(color_fine, true_rgb, data_range=1, size_average=True)
		else:
			raise NotImplementedError
		return color_fine_loss


	def train(self):
		self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
		self.update_learning_rate()
		res_step = self.end_iter - self.iter_step
		self.image_perm = self.get_image_perm()

		self.loss_history = []

		if self.estimate_cam_pose or self.estimate_last_cam_pose:
			self.validate_cam_pose()
		if self.estimate_proj_pose:
			self.validate_proj_pose()

#		self.validate_image()
#		self.validate_mesh(time=self.image_ind_offset)

		if self.aruco_weight_begin > 0 or self.aruco_weight_end > 0:
			pts = self.dataset.aruco_points_at(self.image_ind_offset, scale=False).cpu().detach().numpy()
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(pts)
			os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
			o3d.io.write_point_cloud(os.path.join(self.base_exp_dir, "meshes", "aruco_points.ply"), pcd)

		if self.profiling is not None:
			prof = profile(
				activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
				record_shapes=True,
				on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.base_exp_dir, "logs/profile")),
				with_stack=True
			)
			res_step = self.profiling + 1

		for iter_i in tqdm(range(res_step)):
			if self.profiling is not None and iter_i == 1:
				prof.start()
			alpha = min(self.iter_step / (self.anneal_end + 1e-6), 1)
			igr_weight = alpha * self.igr_weight_end + (1 - alpha) * self.igr_weight_begin
			curv_weight = alpha * self.curv_weight_end + (1 - alpha) * self.curv_weight_begin
			mask_weight = alpha * self.mask_weight_end + (1 - alpha) * self.mask_weight_begin
			sparsity_reg_weight = alpha * self.sparsity_reg_weight_end + (1 - alpha) * self.sparsity_reg_weight_begin
			unimodality_loss_weight = alpha * self.unimodality_loss_weight_end + (1 - alpha) * self.unimodality_loss_weight_begin
			reverse_rendering_weight = alpha * self.reverse_rendering_weight_end + (1 - alpha) * self.sparsity_reg_weight_begin

			if self.ray_sampling_mode == "sequential":
				image_indices = self.image_perm[self.iter_step % len(self.image_perm)]
			elif self.ray_sampling_mode == "random":
				image_indices = torch.stack([self.image_perm[i] for i in np.random.randint(0, len(self.image_perm), size=(self.multi_ray_sample_images,))])
			elif self.ray_sampling_mode == "consecutive":
				img_idx = np.random.randint(0, self.dataset.n_images - self.multi_ray_sample_images)
				image_indices = torch.from_numpy(np.int64([img_idx + i for i in range(self.multi_ray_sample_images)])).to(self.image_perm)
			elif self.ray_sampling_mode == "all":
				image_indices = self.image_perm
			else:
				raise NotImplementedError

			render_out, true_rgb, mask, pixels = self.render_random_rays(image_indices, curv_weight, return_pixels=True)
#			if self.event_mode is not None:
#				mask = torch.isclose(true_rgb - 0.5, torch.zeros_like(true_rgb)).float() * 0.9 + 0.1
			mask_sum = mask.sum() + 1e-5

			color_fine = render_out['color_fine']
			s_val = render_out['s_val']
			gradient_error = render_out['gradient_error']
			curvature_error = render_out['curvature_error']
			temporal_smoothness_loss = render_out['temporal_smoothness_error']
			unimodality_loss = render_out['unimodality_error']
			weight_max = render_out['weight_max']
			weight_sum = render_out['weight_sum']
			transmittance = render_out['transmittance']
			visibility = render_out['visibility']

			# Loss
			color_fine_loss = self.color_error(color_fine, true_rgb, mask, mask_sum)
			psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

			eikonal_loss = gradient_error
			curvature_loss = curvature_error

			if self.igr_loss_scale is not None:
				eikonal_loss = eikonal_loss / eikonal_loss.detach() * color_fine_loss.detach() * self.igr_loss_scale / igr_weight

			mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
			embedder_regularizer = self.renderer.sdf_network.update_embedder(self.iter_step / self.end_iter)
			embedder_loss = 0
			if "smoothness" in embedder_regularizer and self.smoothness_weight > 0:
				embedder_loss += embedder_regularizer["smoothness"] * self.smoothness_weight
			if "density_L1" in embedder_regularizer and self.density_L1_weight > 0:
				embedder_loss += embedder_regularizer["density_L1"] * self.density_L1_weight


			if 'shadow_camera' in render_out and self.shadow_field_weight > 0:
				shadow_field_loss = F.mse_loss(render_out['shadow_camera'], visibility[:,:shadow_camera.shape[1]].detach())
			else:
				shadow_field_loss = 0


			if self.depth_weight_pos > 0:
				depth_loss_pos = self.sdf_network.sdf(self.point_cloud_pos, time=image_indices).abs().mean()
			else:
				depth_loss_pos = 0

			if self.depth_weight_neg > 0:
				depth_loss_neg = (self.sdf_network.sdf(self.point_cloud_neg, time=image_indices).abs() - self.point_cloud_neg_distances).abs().mean()
#				depth_loss_neg = torch.exp(-1e2 * torch.abs(self.sdf_network.sdf(self.point_cloud_neg))).mean()
			else:
				depth_loss_neg = 0

			if self.aruco_weight > 0:
				aruco_loss = self.sdf_network.sdf(self.renderer.to_timed_pts(self.dataset.aruco_points_at(image_indices), image_indices), time=image_indices).abs().mean()
			else:
				aruco_loss = 0

			if sparsity_reg_weight > 0:
				sparsity_loss = torch.exp(-1e2 * torch.abs(self.sdf_network.sdf(self.renderer.to_timed_pts(self.point_cloud_neg, image_indices), time=image_indices))).mean()
			else:
				sparsity_loss = 0

			if reverse_rendering_weight > 0:
				render_out_reverse, true_rgb_reverse, _ = self.render_random_rays(image_indices, 0.0, mode='proj')
				mask_reverse = render_out_reverse["weight_sum"].detach()
				reverse_rendering_loss = self.color_error(render_out_reverse["color_fine"], true_rgb_reverse, mask_reverse, mask_reverse.sum() + 1e-5).sum()
			else:
				reverse_rendering_loss = 0

			if self.llr_weight > 0 and self.deform_network is not None:
				deformed_points = self.renderer.to_timed_pts(self.point_cloud_neg, image_indices)
				llr_loss = local_linear_reconstruction(self.point_cloud_neg, deformed_points, n_neighbors=30)
			else:
				llr_loss = 0

			# something wrong
#			frame_weight = self.dataset.get_frame_weight(image_indices, pixels)

			loss = (color_fine_loss).sum() * self.color_weight +\
				   eikonal_loss * igr_weight +\
				   curvature_loss * curv_weight / self.curv_sample_ratio +\
				   temporal_smoothness_loss * self.temporal_smoothness_weight +\
				   mask_loss * mask_weight +\
				   embedder_loss +\
				   shadow_field_loss * self.shadow_field_weight +\
				   depth_loss_pos * self.depth_weight_pos +\
				   depth_loss_neg * self.depth_weight_neg +\
				   aruco_loss * self.aruco_weight +\
				   sparsity_loss * sparsity_reg_weight +\
				   unimodality_loss * unimodality_loss_weight +\
				   reverse_rendering_loss * reverse_rendering_weight +\
				   llr_loss * self.llr_weight

			self.optimizer.zero_grad()
			loss.backward()

			if self.estimate_cam_pose or self.estimate_cam_intrinsic:
				for param in self.dataset.camera_params.parameters():
					if (self.estimate_last_cam_pose and image_indices != self.current_num_images - 1) or \
						image_indices == self.image_ind_offset:
						param.grad = None
					elif self.iter_step < self.warm_up_end and self.cam_param_warm_up_strategy is not None:
						if self.cam_param_warm_up_strategy == "step":
							param.grad = None
						elif self.cam_param_warm_up_strategy == "linear":
							param.grad *= self.iter_step / self.warm_up_end
						else:
							raise NotImplementedError

			if self.estimate_proj_pose or self.estimate_proj_intrinsic:
				for param in self.dataset.projector_params.parameters():
					if self.iter_step < self.warm_up_end:
						param.grad = None

			if self.estimate_illumination:
				if self.iter_step < self.warm_up_end:
					for param in self.dataset.illumination_params.parameters():
						param.grad = None

			if self.estimate_proj_refraction:
				if self.iter_step < self.warm_up_end:
					for param in self.dataset.proj_refractive_interface.parameters():
						param.grad = None

			self.optimizer.step()

			self.iter_step += 1

			if self.iter_step % self.report_freq == 0:
				with torch.no_grad():
					self.writer.add_scalar('Loss/loss', loss, self.iter_step)
					self.writer.add_scalar('Loss/color_loss', color_fine_loss.sum(), self.iter_step)
					self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
					self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
					self.writer.add_scalar('Loss/curvature_loss', curvature_loss, self.iter_step)
					self.writer.add_scalar('Loss/shadow_field_loss', shadow_field_loss, self.iter_step)
					self.writer.add_scalar('Loss/depth_loss_pos', depth_loss_pos, self.iter_step)
					self.writer.add_scalar('Loss/depth_loss_neg', depth_loss_neg, self.iter_step)
					self.writer.add_scalar('Loss/aruco_loss', aruco_loss, self.iter_step)
					self.writer.add_scalar('Loss/sparsity_loss', sparsity_loss, self.iter_step)
					self.writer.add_scalar('Loss/temporal_smoothness_loss', temporal_smoothness_loss, self.iter_step)
					self.writer.add_scalar('Loss/unimodality_loss', unimodality_loss, self.iter_step)
					self.writer.add_scalar('Loss/reverse_rendering_loss', reverse_rendering_loss, self.iter_step)
					self.writer.add_scalar('Loss/llr_loss', llr_loss, self.iter_step)

					self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
					self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
					self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
					self.writer.add_scalar('Statistics/current_num_images', self.current_num_images, self.iter_step)
					self.writer.add_scalar('Statistics/learning_factor', self.learning_factor, self.iter_step)
					self.writer.add_scalar('Statistics/learning_rate', self.learning_factor * self.learning_rate, self.iter_step)

					if self.projection_network is not None and self.projection_network.rot_sigma is not None:
						self.writer.add_scalar('Statistics/proj_rot_sigma', self.projection_network.rot_sigma.mean(), self.iter_step)

				self.loss_history.append(loss.item())

				print('iter:{:8>d} loss = {} lr={} psnr={},'.format(
					self.iter_step, loss, self.optimizer.param_groups[0]['lr'], psnr))

			if self.iter_step % self.save_freq == 0:
				self.save_checkpoint()

			if self.iter_step % self.val_freq == 0:
				self.renderer.eval()
				self.validate_image()
				if self.estimate_cam_pose or self.estimate_last_cam_pose:
					self.validate_cam_pose()
				if self.estimate_proj_pose:
					self.validate_proj_pose()
#				if self.shadow_field_weight > 0:
				self.renderer.train()

			if self.val_shadow_freq is not None and self.iter_step % self.val_shadow_freq == 0:
				self.renderer.eval()
				self.validate_shadow()
				self.renderer.train()

			if self.iter_step % self.val_mesh_freq == 0:
				self.renderer.eval()
				self.validate_mesh(time=self.image_ind_offset)
				self.renderer.train()

			self.update_learning_rate()

			if self.iter_step % len(self.image_perm) == 0:
				self.image_perm = self.get_image_perm()

		if self.profiling is not None:
			prof.stop()

		return True


	def get_image_perm(self):
		n_images = self.dataset.n_images

		if self.num_images is None:
			perm = torch.randperm(n_images)
			self.current_num_images = num_images
		else:
			if self.num_images_incremental > 0:
				self.current_num_images = min(
					max(
						int(np.ceil(max(self.iter_step - self.num_images_incremental_start, 0) / self.num_images_incremental)) + 1,
						self.min_num_images,
					),
					self.num_images
				)
				perm = torch.randperm(self.current_num_images) + self.image_ind_offset
			else:
				if self.num_images_policy == "interval":
					perm = torch.randperm(self.num_images) * (n_images - self.image_ind_offset) // self.num_images + self.image_ind_offset
				elif self.num_images_policy == "first":
					perm = torch.randperm(self.num_images) + self.image_ind_offset
				else:
					raise NotImplementedError

		return perm

	def get_cos_anneal_ratio(self):
		if self.anneal_end == 0.0:
			return 1.0
		else:
			return np.min([1.0, self.iter_step / self.anneal_end])

	def update_learning_rate(self):
		if self.iter_step < self.warm_up_end:
			self.learning_factor = self.iter_step / self.warm_up_end
		else:
			if self.adaptive_lr is not None and len(self.loss_history) >= 2:
				alpha = min((1 - min(self.loss_history[-1] / self.loss_history[-2], 1)) / self.adaptive_lr, 1)
				self.learning_factor = alpha + self.learning_rate_alpha
			else:
				alpha = self.learning_rate_alpha
				progress = min((self.iter_step - self.warm_up_end) / (self.atten_end_iter - self.warm_up_end + 1e-3), 1)
				self.learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

		for g in self.optimizer.param_groups:
			g['lr'] = self.learning_rate * self.learning_factor

		self.depth_weight_pos = (self.depth_weight_pos_end - self.depth_weight_pos_begin) * (self.iter_step / self.end_iter) + self.depth_weight_pos_begin
		self.depth_weight_neg = (self.depth_weight_neg_end - self.depth_weight_neg_begin) * (self.iter_step / self.end_iter) + self.depth_weight_neg_begin
		self.aruco_weight = (self.aruco_weight_end - self.aruco_weight_begin) * (self.iter_step / self.end_iter) + self.aruco_weight_begin

#		self.shadow_field_ratio = np.clip((self.iter_step - self.shadow_field_begin) / (self.shadow_field_end - self.shadow_field_begin + 1e-3), 0, 1)
#		self.shadow_field_ratio = 1.0

#		if self.iter_step < self.shadow_field_begin:
#			self.shadow_field_ratio = 0.0
#		else:
#			self.shadow_field_ratio = np.abs(np.clip((self.iter_step % self.shadow_volume_anneal_interval) / (self.shadow_volume_anneal_term + 1), 0, 1) * 2 - 1)

	def file_backup(self):
		dir_lis = self.conf['general.recording']
		os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
		for dir_name in dir_lis:
			cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
			os.makedirs(cur_dir, exist_ok=True)
			files = os.listdir(dir_name)
			for f_name in files:
				if f_name[-3:] == '.py' or f_name[-5:] == '.conf':
					copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

		copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

	def load_checkpoint(self, checkpoint_name, metadata_only=False, params_only=False):
		checkpoint = torch.load(checkpoint_name, map_location=self.device, weights_only=False)
		if not metadata_only:
			self.nerf_outside.load_state_dict(checkpoint['nerf'], strict=False)
			self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'], strict=False)
			self.deviation_network.load_state_dict(checkpoint['variance_network_fine'], strict=False)
			self.color_network.load_state_dict(checkpoint['color_network_fine'], strict=False)

			if self.deform_network is not None:
				self.deform_network.load_state_dict(checkpoint['deform_network'], strict=False)

			if self.dataset.with_projection and 'projection_network' in checkpoint:
				self.projection_network.load_state_dict(checkpoint['projection_network'], strict=False)

				if 'shadow_field' in checkpoint:
					self.shadow_field.load_state_dict(checkpoint['shadow_field'], strict=False)
				if 'illumination_params' in checkpoint:
					self.dataset.illumination_params.load_state_dict(checkpoint['illumination_params'], strict=False)

			try:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
			except ValueError:
				print("Optimizer state was not successfully loaded.")
			self.iter_step = checkpoint['iter_step']

		if not params_only:
			if 'camera_params' in checkpoint:
				self.dataset.camera_params.load_state_dict(checkpoint['camera_params'], strict=False)

			if self.dataset.with_projection:
				if 'projector_params' in checkpoint:
					self.dataset.projector_params.load_state_dict(checkpoint['projector_params'], strict=False)
				if 'projection_pattern' in checkpoint:
					self.dataset.projection_pattern.load_state_dict(checkpoint['projection_pattern'], strict=False)

			if 'cam_refractive_interface' in checkpoint:
				self.dataset.cam_refractive_interface.load_state_dict(checkpoint['cam_refractive_interface'], strict=False)
			if 'proj_refractive_interface' in checkpoint:
				self.dataset.proj_refractive_interface.load_state_dict(checkpoint['proj_refractive_interface'], strict=False)

		self.update_learning_rate()
		logging.info('End')

	def save_checkpoint(self):
		checkpoint = {
			'nerf': self.nerf_outside.state_dict(),
			'sdf_network_fine': self.sdf_network.state_dict(),
			'variance_network_fine': self.deviation_network.state_dict(),
			'color_network_fine': self.color_network.state_dict(),
			'camera_params': self.dataset.camera_params.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'iter_step': self.iter_step,
		}

		if self.dataset.with_projection:
			checkpoint = {**checkpoint,
				'projection_pattern': self.dataset.projection_pattern.state_dict(),
				'projector_params': self.dataset.projector_params.state_dict(),
				'illumination_params': self.dataset.illumination_params.state_dict(),
				'projection_network': self.projection_network.state_dict(),
			}

			if self.shadow_field is not None:
				checkpoint['shadow_field'] = self.shadow_field.state_dict()

		if self.dataset.with_cam_refraction:
			checkpoint = {**checkpoint,
				'cam_refractive_interface': self.dataset.cam_refractive_interface.state_dict(),
			}
		if self.dataset.with_proj_refraction:
			checkpoint = {**checkpoint,
				'proj_refractive_interface': self.dataset.proj_refractive_interface.state_dict(),
			}

		if self.deform_network is not None:
			checkpoint['deform_network'] = self.deform_network.state_dict()

		os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
		torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


	def render_image_core(self, idx, rays_o, rays_d, true_rgb, proj_params, illum_params, visualize_shadow=False, mode='cam'):
		H, W, _ = rays_o.shape
		rays_o = rays_o.reshape(-1, 3).split(self.batch_size // 1)
		rays_d = rays_d.reshape(-1, 3).split(self.batch_size // 1)
		true_rgb = true_rgb.reshape(-1, 3).split(self.batch_size // 1)

		out_rgb_fine = []
		out_alpha_fine = []
		out_normal_fine = []
		out_opacity_fine = []
		out_transmittance = []
		out_coord = []

		for i, (rays_o_batch, rays_d_batch, true_rgb_batch) in tqdm(enumerate(zip(rays_o, rays_d, true_rgb))):
			near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
			background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

			render_out = self.renderer.render(rays_o_batch,
											  rays_d_batch,
											  idx,
											  near,
											  far,
											  cos_anneal_ratio=self.get_cos_anneal_ratio(),
											  background_rgb=background_rgb,
											  proj_params=proj_params,
											  illum_params=illum_params,
											  shadow_field_ratio=self.shadow_field_ratio,
											  visualize_shadow=visualize_shadow,
											  disable_shadow=self.iter_step<self.shadow_field_begin and not visualize_shadow,
											  mode=mode)

			def feasible(key): return (key in render_out) and (render_out[key] is not None)

			if feasible('color_fine'):
				out_rgb_fine.append(render_out['color_fine'].cpu().detach())
			if feasible('weight_sum'):
				out_alpha_fine.append(render_out['weight_sum'].cpu().detach())
			if feasible('gradients') and feasible('weights'):
				n_samples = self.renderer.n_samples + self.renderer.n_importance
				normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
				if feasible('inside_sphere'):
					normals = normals * render_out['inside_sphere'][..., None]
				normals = normals.sum(dim=1)
				out_normal_fine.append(normals.cpu().detach())
			if feasible('opacity') and feasible('transmittance'):
				out_opacity_fine.append(render_out['opacity'].cpu().detach())
				out_transmittance.append(render_out['transmittance'].cpu().detach())
			if feasible('coord'):
				out_coord.append(render_out['coord'].cpu().detach())
			del render_out

		img_fine = None
		if len(out_rgb_fine) > 0:
			img_fine = torch.cat(out_rgb_fine, dim=0).view([H, W, -1]).clamp(0, 1)

		alpha_fine = None
		if len(out_alpha_fine) > 0:
			alpha_fine = torch.cat(out_alpha_fine, dim=0).view([H, W, 1]).clamp(0, 1)

		normal_img = None
		if len(out_normal_fine) > 0:
			normal_img = torch.cat(out_normal_fine, dim=0)
			rot = torch.linalg.inv(self.dataset.camera_params.get_pose(idx)[:3, :3])
			normal_img = (torch.matmul(rot[None, :, :].cpu().detach(), normal_img[:, :, None]).view([H, W, 3]) * 0.5 + 0.5).clamp(0, 1)

		opacity_tensor = None
		transmittance_tensor = None
		if len(out_opacity_fine) > 0 and len(out_transmittance) > 0:
			opacity_tensor = torch.cat(out_opacity_fine, dim=0).view([H, W, -1])
			transmittance_tensor = torch.cat(out_transmittance, dim=0).view([H, W, -1])
			opacity_tensor = torch.stack([opacity_tensor, transmittance_tensor], dim=0)

		return img_fine, normal_img, alpha_fine, opacity_tensor


	def render_image_impl(self, idx, resolution_level=-1, illum_params_override=None, visualize_shadow=False, mode='cam'):
		if resolution_level < 0:
			resolution_level = self.validate_resolution_level
		
		with torch.no_grad():
			rays_o, rays_d, true_rgb = self.dataset.gen_rays_at(idx, resolution_level=resolution_level, mode=mode)

		if self.dataset.with_projection:
			proj_params = self.dataset.get_proj_params(torch.tensor(idx), mode)
			if illum_params_override is None:
				illum_params = self.dataset.illumination_params(idx)
			else:
				illum_params = illum_params_override
		else:
			proj_params = []
			illum_params = None

		img_fine, normal_img, alpha_fine, opacity_tensor = self.render_image_core(idx - self.image_ind_offset, 
			rays_o, rays_d, true_rgb, proj_params, illum_params, visualize_shadow, mode)
		return img_fine, normal_img, alpha_fine, opacity_tensor


	def render_image(self, idx, resolution_level=-1, illum_params_override=None, visualize_shadow=False, mode='cam', return_gt=False):
		if self.event_mode is not None:
			idx_prev, idx_next = get_event_indices(idx, self.image_perm, self.event_mode)
			img_fine_next, normal_img, alpha_fine, _ = self.render_image_impl(idx_next, resolution_level, illum_params_override, visualize_shadow, mode)
			img_fine_prev, _, _, _                   = self.render_image_impl(idx_prev, resolution_level, illum_params_override, visualize_shadow, mode)
			img_fine = self.event_camera_model(img_fine_prev, img_fine_next)

			if return_gt:
				if self.event_mode != "sequential":
					img_prev_gt = self.dataset.image_at(idx_prev, resolution_level=resolution_level) / 255
					img_next_gt = self.dataset.image_at(idx_next, resolution_level=resolution_level) / 255
					img_gt = self.event_camera_model(img_prev_gt, img_next_gt) * 255
				else:
					img_gt = self.dataset.image_at(idx, resolution_level=resolution_level)

		else:
			img_fine, normal_img, alpha_fine, _ = self.render_image_impl(idx, resolution_level, illum_params_override, visualize_shadow, mode)
			img_fine_prev = None

			if return_gt:
				img_gt = self.dataset.image_at(idx, resolution_level=resolution_level)

		img_fine = img_fine.cpu().detach().numpy() * 255
		if normal_img is not None:
			normal_img = normal_img.cpu().detach().numpy() * 255
		if alpha_fine is not None:
			alpha_fine = alpha_fine.cpu().detach().numpy() * 255
		if img_fine_prev is not None:
			img_fine_prev = img_fine_prev.cpu().detach().numpy() * 255

		if return_gt:
			return img_fine, normal_img, alpha_fine, img_fine_prev, img_gt
		else:
			return img_fine, normal_img, alpha_fine, img_fine_prev


	def render_image_satellite(self, rot, resolution_level=-1, illum_params_override=None, visualize_shadow=False):
		rays_o, rays_d, true_rgb = self.dataset.gen_rays_satellite(rot, resolution_level=resolution_level)

		if self.dataset.with_projection:
			proj_params = self.dataset.get_proj_params(torch.from_numpy(np.int32([idx])))
			if illum_params_override is None:
				illum_params = self.dataset.illumination_params()
			else:
				illum_params = illum_params_override
		else:
			proj_params = []
			illum_params = None

		img_fine, _, _, _, _ = self.render_image_core(0, rays_o, rays_d, true_rgb, proj_params, illum_params, visualize_shadow)
		img_fine = img_fine.cpu().detach().numpy() * 255
		return img_fine


	def render_novel_image_impl(self, idx_0, idx_1, ratio, resolution_level, illum_params_override=None):
		"""
		Interpolate view between two cameras.
		"""
		rays_o, rays_d, true_rgb = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)

		if self.dataset.with_projection:
			proj_params_0 = self.dataset.get_proj_params(torch.from_numpy(np.int32([idx_0])))
			proj_params_1 = self.dataset.get_proj_params(torch.from_numpy(np.int32([idx_1])))
			proj_params = proj_params_0

			for pi in range(len(proj_params)):
				cam_pose = self.dataset.interpolate_pose(proj_params_0[pi]["cam_pose"][0], proj_params_1[pi]["cam_pose"][0], ratio)[None]
				proj_pose = self.dataset.interpolate_pose(proj_params_0[pi]["proj_pose"][0], proj_params_1[pi]["proj_pose"][0], ratio)[None]
				proj_params[pi]["cam_pose"] = cam_pose
				proj_params[pi]["proj_pose"] = proj_pose
			if illum_params_override is None:
				illum_params = self.dataset.illumination_params()
			else:
				illum_params = illum_params_override
		else:
			proj_params = []
			illum_params = None

		img_fine, normal_img, alpha_fine, _, _ = self.render_image_core(idx_0, rays_o, true_rgb, rays_d, proj_params, illum_params)
		return img_fine, alpha_fine


	def render_novel_image(self, idx_0, idx_1, ratio, resolution_level, illum_params_override=None):
		if args.event_mode is not None:
			idx_0_prev, idx_0_next = get_event_indices(idx_0, self.image_perm, args.event_mode)
			idx_1_prev, idx_1_next = get_event_indices(idx_1, self.image_perm, args.event_mode)
			img_fine_prev, alpha_fine = self.render_novel_image_impl(idx_0_prev, idx_1_prev, ratio, resolution_level, illum_params_override)
			img_fine_next, _          = self.render_novel_image_impl(idx_0_next, idx_1_next, ratio, resolution_level, illum_params_override)
			img_fine = self.event_camera_model(img_fine_prev, img_fine_next)

		else:
			img_fine, alpha_fine = self.render_novel_image_impl(idx_0, idx_1, ratio, resolution_level, illum_params_override)

		img_fine = img_fine.cpu().detach().numpy() * 255
		alpha_fine = alpha_fine.cpu().detach().numpy() * 255
#		img_fine = np.dstack([img_fine, alpha_fine])
		return img_fine


	def validate_image(self, idx=-1, resolution_level=-1, illum_params_override=None):
		if idx < 0:
			idx = int(random.choice(self.get_image_perm()).item())

		print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
		if resolution_level < 0:
			resolution_level = self.validate_resolution_level
		img_fine, normal_img, img_fine_orig, _, img_gt = self.render_image(idx, resolution_level, illum_params_override, return_gt=True)
#		img_fine_albedo, _, _, _ = self.render_image(idx, resolution_level, {"ambient": 1.0, "diffuse": 0.0, "emissive": 0.0})

		os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
		os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

		mask_gt = self.dataset.mask_at(idx, resolution_level=resolution_level)

		if img_fine.shape[-1] == 1:
			img_gt = img_gt.mean(axis=-1, keepdims=True)
			mask_gt = mask_gt.mean(axis=-1, keepdims=True)
		cv2.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{}.png'.format(self.iter_step, idx)),
					np.vstack([np.hstack([img_fine, img_gt]), np.hstack([np.abs(img_fine - img_gt), mask_gt])]))
		if normal_img is not None:
			cv2.imwrite(os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}.png'.format(self.iter_step, idx)), normal_img)

		if img_fine_orig is not None:
			os.makedirs(os.path.join(self.base_exp_dir, 'validations_orig'), exist_ok=True)
			cv2.imwrite(os.path.join(self.base_exp_dir, 'validations_orig', '{:0>8d}_{}.png'.format(self.iter_step, idx)), img_fine_orig)

		patterns_gt = self.dataset.patterns_at(idx, resolution_level=resolution_level)

		if len(patterns_gt) > 0 and (self.reverse_rendering_weight_begin > 0 or self.reverse_rendering_weight_end > 0):
			os.makedirs(os.path.join(self.base_exp_dir, 'validations_proj'), exist_ok=True)
			img_fine_proj, _, proj_mask, _ = self.render_image(idx, resolution_level, illum_params_override, mode='proj')
			pattern_diff = np.abs(patterns_gt[0] - img_fine_proj) * proj_mask / 255
			cv2.imwrite(os.path.join(self.base_exp_dir, 'validations_proj', '{:0>8d}_{}.png'.format(self.iter_step, idx)),
						np.vstack([np.hstack([img_fine_proj, patterns_gt[0]]), np.hstack([pattern_diff, np.repeat(proj_mask, 3, axis=2)])]))



	def validate_shadow(self, idx=-1, resolution_level=-1):
		if idx < 0:
			idx = int(random.choice(self.get_image_perm()).item())

		print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
		if resolution_level < 0:
			resolution_level = self.validate_resolution_level
		shadow_img, _, _, _ = self.render_image(idx, resolution_level, None, visualize_shadow=True)

		img_gt = self.dataset.image_at(idx, resolution_level=resolution_level)
		if shadow_img.shape[-1] == 1:
			img_gt = img_gt.mean(axis=-1, keepdims=True)

		os.makedirs(os.path.join(self.base_exp_dir, 'shadow'), exist_ok=True)

		cv2.imwrite(os.path.join(self.base_exp_dir, 'shadow', '{:0>8d}_{}.png'.format(self.iter_step, 0)),
					   np.concatenate([shadow_img, img_gt]))


	def validate_cam_pose(self, num_cameras=None):
		os.makedirs(os.path.join(self.base_exp_dir, 'cam_poses'), exist_ok=True)
		step = max(self.dataset.n_images // num_cameras, 1) if num_cameras is not None else 1
		with torch.no_grad():
			cam_poses = torch.stack([self.dataset.camera_params.get_pose(i) for i in range(0,self.dataset.n_images,step)])
			cam_poses_gt = torch.stack([self.dataset.camera_params.get_pose_gt(i) for i in range(0,self.dataset.n_images,step)])
			fig = plot_camera_scene(cam_poses, cam_poses_gt, "Estimated camera poses")
		fig.savefig(os.path.join(self.base_exp_dir, 'cam_poses', '{:0>8d}.png'.format(self.iter_step)))


	def validate_proj_pose(self, index=None):
		if index is None:
			index = self.image_ind_offset
		os.makedirs(os.path.join(self.base_exp_dir, 'proj_poses'), exist_ok=True)
		with torch.no_grad():
			cam_pose = self.dataset.camera_params.get_pose(index).unsqueeze(0)
			proj_poses = torch.cat([cam_pose, cam_pose @ self.dataset.projector_params.get_pose(index)], dim=0)
			proj_poses_gt = torch.cat([cam_pose, cam_pose @ self.dataset.projector_params.get_pose_gt(index)], dim=0)
			fig = plot_camera_scene(proj_poses, proj_poses_gt, "Estimated projector poses", with_icp=False)
		fig.savefig(os.path.join(self.base_exp_dir, 'proj_poses', '{:0>8d}.png'.format(self.iter_step)))


	def validate_pattern(self, num_patterns=10):
		os.makedirs(os.path.join(self.base_exp_dir, 'patterns'), exist_ok=True)
		with torch.no_grad():
			for i in np.random.randint(0, self.dataset.n_images, num_patterns):
				patterns = self.dataset.projection_pattern.get_pattern(i).cpu().numpy()[0]
				patterns = np.hstack([pattern for pattern in patterns])
#				patterns = (patterns - np.min(patterns)) / (np.max(patterns) - np.min(patterns))
				cv2.imwrite(os.path.join(self.base_exp_dir, 'patterns', '{:0>8d}_{:0>3d}.png'.format(self.iter_step, i)), (patterns * 255).astype(np.uint8))


	def validate_mesh(self, world_space=True, time=0, suffix="", swap_time_axis=None, mass_center=None):
		bound_min = torch.tensor(self.dataset.object_bbox_min * self.mesh_extract_scale, dtype=torch.float32)
		bound_max = torch.tensor(self.dataset.object_bbox_max * self.mesh_extract_scale, dtype=torch.float32)

		if mass_center is not None:
			mass_center = self.dataset.scale_mat_inv[:3,:3] @ torch.from_numpy(np.float32(mass_center)).to(self.device) + self.dataset.scale_mat_inv[:3,3][None]

		vertices, triangles =\
			self.renderer.extract_geometry(bound_min, bound_max, resolution=self.val_mesh_resolution, 
				time=time - self.image_ind_offset, swap_time_axis=swap_time_axis, mass_center=mass_center)
		os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
		if world_space:
			scale_mat = self.dataset.scale_mat.cpu().detach().numpy()
			vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]

		mesh = trimesh.Trimesh(vertices, triangles)
#		mesh = remove_floaters(mesh)
		mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}{}.ply'.format(self.iter_step, suffix)))

		self.visualize_sdf(os.path.join(self.base_exp_dir, "meshes", '{:0>8d}{}'.format(self.iter_step, suffix)), resolution=256)

		logging.info('End')


	def visualize_sdf(self, path, resolution=1024, extract_scale=1.0):
		bound_min = torch.tensor(self.dataset.object_bbox_min * self.mesh_extract_scale, dtype=torch.float32)
		bound_max = torch.tensor(self.dataset.object_bbox_max * self.mesh_extract_scale, dtype=torch.float32)

		sdf = -self.renderer.extract_fields(bound_min, bound_max, resolution=resolution)
#		np.save(path, sdf)
		xy = sdf[:,:,resolution//2]
		xz = sdf[:,resolution//2,:]
		yz = sdf[resolution//2,:,:]

		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=(24,9))
		fig, axs = plt.subplots(1, 3, figsize=(15, 5))
		cax1 = axs[0].imshow(xy, aspect='auto', cmap='viridis', vmin=np.min(sdf), vmax=np.max(sdf))
		axs[0].set_xlabel("xy")
		cax2 = axs[1].imshow(xz, aspect='auto', cmap='viridis', vmin=np.min(sdf), vmax=np.max(sdf))
		axs[1].set_xlabel("xz")
		cax3 = axs[2].imshow(yz, aspect='auto', cmap='viridis', vmin=np.min(sdf), vmax=np.max(sdf))
		axs[2].set_xlabel("yz")

		for i in range(3):
			axs[i].set_xticks(np.arange(resolution)[::128])
			axs[i].set_xticklabels(np.round(np.linspace(self.dataset.object_bbox_min[i],self.dataset.object_bbox_max[i],resolution)[::128], 2))
			axs[i].set_yticks(np.arange(resolution)[::128])
			axs[i].set_yticklabels(np.round(np.linspace(self.dataset.object_bbox_min[i],self.dataset.object_bbox_max[i],resolution)[::128], 2))

		cbar = fig.colorbar(cax1, ax=axs, orientation='vertical', fraction=0.02)
#		plt.tight_layout()
		plt.savefig(path + ".png")
		plt.close(fig)


	def interpolate_view(self, img_idx_0, img_idx_1, n_frames, illum_params_override=None, render_name=None):
		images = []
		for i in range(n_frames):
			print(i)
			images.append(self.render_novel_image(img_idx_0, img_idx_1, i / n_frames,
						  resolution_level=1,
						  illum_params_override=illum_params_override))
		return images


	def save_opacity(self, idx, save_path):
		_, _, _, opacity_tensor, _ = self.render_image_impl(idx, -1, None, False)
		np.save(save_path, opacity_tensor)
		img_gt = self.dataset.image_at(idx, resolution_level=1)
		cv2.imwrite(save_path + ".png", img_gt)


if __name__ == '__main__':
	print('Hello Wooden')

	torch.set_default_tensor_type('torch.cuda.FloatTensor')

	FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
	logging.basicConfig(level=logging.DEBUG, format=FORMAT)

	parser = argparse.ArgumentParser()
	parser.add_argument('--conf', type=str, default='./confs/base.conf')
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--exp_name', type=str, default='default')
	parser.add_argument('--load', type=str)
	parser.add_argument('--load_params', type=str)
	parser.add_argument('--load_metadata', type=str)
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--case', type=str, default='')
	parser.add_argument('--validate_mesh_after_train', type=bool, default=True)
	parser.add_argument('--render_images_after_train', type=bool, default=False)
	parser.add_argument('--end_iter', type=int)
	parser.add_argument('--render_name', type=str)

	parser.add_argument('--mask_ratio', type=float, default=1)
	parser.add_argument('--illum_params', type=float, nargs="+")
	parser.add_argument('--estimate_illumination', action='store_true')
	parser.add_argument('--cam_noise', type=json.loads)
	parser.add_argument('--estimate_cam_intrinsic', action='store_true')
	parser.add_argument('--estimate_cam_pose', action='store_true')
	parser.add_argument('--estimate_last_cam_pose', action='store_true')
	parser.add_argument('--use_mlp_for_pose', action='store_true')
	parser.add_argument('--proj_noise', type=json.loads)
	parser.add_argument('--estimate_proj_intrinsic', action='store_true')
	parser.add_argument('--estimate_proj_pose', action='store_true')
	parser.add_argument('--pattern_noise', type=json.loads)

	parser.add_argument('--estimate_cam_refraction', action='store_true')
	parser.add_argument('--cam_refractive_interface_params', type=json.loads)
	parser.add_argument('--estimate_proj_refraction', action='store_true')
	parser.add_argument('--proj_refractive_interface_params', type=json.loads)
	parser.add_argument('--frame_weights', action='store_true')

	parser.add_argument('--num_images', type=int)
	parser.add_argument('--num_images_policy', type=str, choices=['interval', 'first'], default='interval')
	parser.add_argument('--num_images_incremental', type=int, default=0)
	parser.add_argument('--num_images_incremental_start', type=int, default=0)
	parser.add_argument('--min_num_images', type=int, default=1)
	parser.add_argument('--image_ind_offset', type=int, default=0)
	parser.add_argument('--dynamic', action='store_true')
	parser.add_argument('--freeze', type=str, nargs="*", default=[])

	parser.add_argument('--overwrite_params', type=json.loads)
	parser.add_argument('--pretrain_sdf_network', action="store_true")
	parser.add_argument('--initial_shape_type', type=str, default="sphere")
	parser.add_argument('--initial_shape_mesh', type=str)
	parser.add_argument('--event_mode', type=str, choices=["sequential", "accumulated", "random"])
	parser.add_argument('--simulate_event', action="store_true")
	parser.add_argument('--scene_scale', type=float, default=1.0)
	parser.add_argument('--baseline_scale', type=float, default=1.0)

	parser.add_argument('--mass_center', type=float, nargs="+")

	parser.add_argument('--profiling', type=int)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	torch_fix_seed(args.seed)

	if args.illum_params is not None:
		illum_params = {
			"ambient": args.illum_params[0],
			"diffuse": args.illum_params[1],
			"emissive": args.illum_params[2],
		}
	else:
		illum_params = None

#	torch.cuda.set_device(args.gpu)
	runner = Runner(args.conf, args.mode, args.exp_name, args.case, args.load, args.load_metadata, args.load_params, args.resume,
		num_images=args.num_images, num_images_policy=args.num_images_policy, num_images_incremental=args.num_images_incremental, 
		num_images_incremental_start=args.num_images_incremental_start, min_num_images=args.min_num_images,
		image_ind_offset=args.image_ind_offset, dynamic=args.dynamic,
		mask_ratio=args.mask_ratio, initial_illum_params=illum_params, estimate_illumination=args.estimate_illumination, 
		estimate_cam_intrinsic=args.estimate_cam_intrinsic, estimate_cam_pose=args.estimate_cam_pose, use_mlp_for_pose=args.use_mlp_for_pose,
		estimate_last_cam_pose=args.estimate_last_cam_pose, cam_noise=args.cam_noise, 
		estimate_proj_intrinsic=args.estimate_proj_intrinsic, estimate_proj_pose=args.estimate_proj_pose, 
		proj_noise=args.proj_noise, pattern_noise=args.pattern_noise, 
		estimate_cam_refraction=args.estimate_cam_refraction, cam_refractive_interface_params=args.cam_refractive_interface_params,
		estimate_proj_refraction=args.estimate_proj_refraction, proj_refractive_interface_params=args.proj_refractive_interface_params,
		frame_weights=args.frame_weights,	end_iter=args.end_iter,	overwrite_params=args.overwrite_params, freeze=args.freeze,
		pretrain_sdf_network=args.pretrain_sdf_network, initial_shape_type=args.initial_shape_type, initial_shape_mesh=args.initial_shape_mesh,
		event_mode=args.event_mode, simulate_event=args.simulate_event, scene_scale=args.scene_scale, baseline_scale=args.baseline_scale, profiling=args.profiling)

	runner.renderer.eval()

	if args.mode == 'train':
		runner.renderer.train()
		ret = runner.train()

		if args.validate_mesh_after_train:
			runner.validate_mesh(time=args.image_ind_offset)

		if args.render_images_after_train:
			os.makedirs(os.path.join(runner.base_exp_dir, 'render', 'test'), exist_ok=True)
			for idx in range(runner.dataset.n_images):
				print("Rendering", idx)
				img, normal, _, _, _ = runner.render_image(idx, 1, illum_params)
				cv2.imwrite(os.path.join(runner.base_exp_dir, 'render', 'test', '%03d.png' % idx), img)

	elif args.mode == 'validate_mesh':
		runner.validate_mesh(time=args.image_ind_offset)

	elif args.mode.startswith('validate_mesh_interp'):
		_, _, _, axis = args.mode.split('_')
		axis = int(axis)
		runner.validate_mesh(time=args.image_ind_offset, suffix="_interp_%d" % axis, swap_time_axis=axis, mass_center=args.mass_center)

	elif args.mode.startswith('validate_mesh_multi'):
		_, _, _, begin, end, stride = args.mode.split('_')
		for i in range(int(begin), int(end), int(stride)):
			runner.validate_mesh(time=i, suffix="_%03d" % i)

	elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
		_, img_idx_0, img_idx_1, num_frames = args.mode.split('_')
		img_idx_0 = int(img_idx_0)
		img_idx_1 = int(img_idx_1)
		num_frames = int(num_frames)
		images = runner.interpolate_view(img_idx_0, img_idx_1, num_frames, illum_params)
		render_name = args.render_name if args.render_name is not None else "interpolate"
		os.makedirs(os.path.join(runner.base_exp_dir, render_name), exist_ok=True)
		for i, image in enumerate(images):
			cv2.imwrite(os.path.join(runner.base_exp_dir, render_name, "%03d.png" % i), image)

	elif args.mode == 'rendering':
		render_name = args.render_name if args.render_name is not None else "render"
		os.makedirs(os.path.join(runner.base_exp_dir, render_name), exist_ok=True)

		for idx in range(runner.num_images):
			print("Rendering", idx)
			img, normal, alpha, img_orig, img_gt = runner.render_image(idx, 1, illum_params, return_gt=True)
#			if img.shape[-1] == 1:
#				img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#			img = np.dstack([img, alpha])
			cv2.imwrite(os.path.join(runner.base_exp_dir, render_name, '%03d.png' % idx), img)
			cv2.imwrite(os.path.join(runner.base_exp_dir, render_name, '%03d_gt.png' % idx), img_gt)

			if img_orig is not None:
				os.makedirs(os.path.join(runner.base_exp_dir, 'render_orig'), exist_ok=True)
				cv2.imwrite(os.path.join(runner.base_exp_dir, 'render_orig', '%03d.png' % idx), img_orig)


	elif args.mode.startswith('render_satellite'):
		_, _, yaw, pitch, roll = args.mode.split('_')
		os.makedirs(os.path.join(runner.base_exp_dir, 'render', 'satellite'), exist_ok=True)

		rot = Rot.from_euler("zxy", [float(yaw), float(pitch), float(roll)], degrees=True).as_matrix()
		img = runner.render_image_satellite(rot, 4, illum_params)
		cv2.imwrite(os.path.join(runner.base_exp_dir, 'render', 'satellite', 'img.png'), img)


	elif args.mode.startswith('rendering_interp'):
		_, _, begin, end, num_frames = args.mode.split('_')
		begin = int(begin)
		end = int(end)
		num_frames = int(num_frames)
		if begin == end:
			end += runner.dataset.n_images
		os.makedirs(os.path.join(runner.base_exp_dir, 'render_interp'), exist_ok=True)

		for i in range(begin, end):
			idx = i % runner.dataset.n_images
			idx_next = (i + 1) % runner.dataset.n_images
			print("Rendering", idx)
			images = runner.interpolate_view(idx, idx_next, num_frames, illum_params)
			for i, image in enumerate(images):
				cv2.imwrite(os.path.join(runner.base_exp_dir, 'render_interp', '%03d_%03d.png' % (idx, i)), image.squeeze())

	elif args.mode.startswith('superres'):
		_, idx, scale = args.mode.split('_')
		os.makedirs(os.path.join(runner.base_exp_dir, 'render', 'superres'), exist_ok=True)
		idx = int(idx)
		scale = int(scale)

		img, normal, _, _, _ = runner.render_image(idx, 1 / scale, illum_params)
		cv2.imwrite(os.path.join(runner.base_exp_dir, 'render', 'superres', '%03d_%d.png' % (idx, scale)), img)

	elif args.mode == 'visualize_sdf':
		os.makedirs(os.path.join(runner.base_exp_dir, 'render', 'sdf'), exist_ok=True)
		vis_list = runner.visualize_sdf(os.path.join(runner.base_exp_dir, 'render', 'sdf', 'vis'), resolution=256)

	elif args.mode.startswith('save_opacity'):
		_, _, idx = args.mode.split('_')
		idx = int(idx)
		os.makedirs(os.path.join(runner.base_exp_dir, 'opacity'), exist_ok=True)
		runner.save_opacity(idx, os.path.join(runner.base_exp_dir, 'opacity', '%03d' % idx))
