import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import cv2
import open3d as o3d
import numpy as np
import os, json
import itertools
from tqdm import tqdm
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

#from pytorch3d.transforms import so3_log_map, so3_exp_map
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
from models import camera
from models.embedder import get_embedder
from .utils import GradientScaler, create_gaussian_kernel, find_peak_interval, SingleConv


def recursive_to_ndarray(obj):
	return {k: np.array(v) for k, v in obj.items()}


def to_homo(m):
	last_dim = torch.zeros_like(m[...,:1,:])
	last_dim[...,-1] = 1
	return torch.cat([m, last_dim], dim=-2)


def gather_unique(indices, fn):
	unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
	results = fn(unique_indices)
	return results[inverse_indices]


def refract(ray_dir, interface_normal, mu):
	dot = torch.sum(ray_dir * (-interface_normal), dim=-1)[...,np.newaxis]
	refracted_ray_dir = (ray_dir / mu - (dot + torch.sqrt(mu ** 2 + dot ** 2 - 1.0)) / mu * (-interface_normal))
	return refracted_ray_dir


def compute_distance_to_plane(ray_origin, ray_dir, plane_normal, plane_origin):
	distance = -(torch.sum(plane_normal * (ray_origin - plane_origin), dim=-1) / torch.sum(plane_normal * ray_dir, dim=-1))
	return distance


def solve2nd(coefs):
	a = coefs[...,0]
	b = coefs[...,1]
	c = coefs[...,2]

	D = b ** 2 - 4 * a * c
	D = D.type(torch.complex128)

	x0 = (-b + torch.sqrt(D)) / (2 * a)
	x1 = (-b - torch.sqrt(D)) / (2 * a)
	return torch.stack([x0, x1], dim=-1)


def solve3rd(coefs):
	a = coefs[...,0]
	b = coefs[...,1]
	c = coefs[...,2]
	d = coefs[...,3]

	A0 = d / a
	A1 = c / a
	A2 = b / a
	p = A1 - A2 * A2 / 3
	q = A0 - A1 * A2 / 3 + A2 * A2 * A2 * 2 / 27
	D = (q / 2) ** 2 + (p / 3) ** 3 + 1e-12

	r = torch.sqrt(D.type(torch.complex128))
	o = complex(-0.5, np.sqrt(3) / 2)

	L = torch.pow(-q / 2 + r, 1/3)
	R = torch.pow(-q / 2 - r, 1/3)
	x0 = L + R - A2 / 3
	x1 = o * o * L + o * R - A2 / 3
	x2 = o * L + o * o * R - A2 / 3

	# L_ = torch.float_power(-q / 2 + r, 1/3)
	# R_ = torch.float_power(-q / 2 - r, 1/3)
	# x0_ = L_ + R_ - A2 / 3
	# x1_ = o * o * L_ + o * R_ - A2 / 3
	# x2_ = o * L_ + o * o * R_ - A2 / 3

	# # 還元不能の場合
	# x0[D < 0] = x0_[D < 0]
	# x1[D < 0] = x1_[D < 0]
	# x2[D < 0] = x2_[D < 0]

	return torch.stack([x0, x1, x2], dim=-1)


def solve4th(coefs):
	a = coefs[...,0]
	b = coefs[...,1]
	c = coefs[...,2]
	d = coefs[...,3]
	e = coefs[...,4]

	A0 = e / a
	A1 = d / a
	A2 = c / a
	A3 = b / a
	p = A2 - 6 * A3 * A3 / 16
	q = A1 - A2 * A3 / 2 + 8 * (A3 / 4) ** 3
	r = A0 - A1 * A3 / 4 + A2 * A3 * A3 / 16 - 3 * (A3 / 4) ** 4

	coefs2 = torch.stack([
		torch.ones_like(p),
		2 * p,
		p * p - 4 * r,
		-q * q
	]).T

	u = solve3rd(coefs2)
	cost = u.imag.abs()
	cost[u.real < 0] = 1e6
	t = u[np.arange(u.shape[0]),torch.argmin(cost,dim=1)].real.clip(min=1e-6)

	coefs3_1 = torch.stack([
		torch.ones_like(t),
		torch.sqrt(t),
		(p + t) / 2 - torch.sqrt(t) * q / (2 * t),
	]).T

	coefs3_2 = torch.stack([
		torch.ones_like(t),
		-torch.sqrt(t),
		(p + t) / 2 + torch.sqrt(t) * q / (2 * t),
	]).T

	y0 = solve2nd(coefs3_1)
	y1 = solve2nd(coefs3_2)

	x0 = y0[:,0] - A3 / 4
	x1 = y0[:,1] - A3 / 4
	x2 = y1[:,0] - A3 / 4
	x3 = y1[:,1] - A3 / 4

	return torch.stack([x0, x1, x2, x3], dim=-1)


def solventh(coefs):
	comp = torch.zeros((*coefs.shape, coefs.shape[-1]), dtype=torch.cfloat).to(coefs)
	for i in range(coefs.shape[-1] - 1):
		comp[...,0,i] = -coefs[...,i+1] / coefs[...,0]
		if i > 0:
			comp[...,i,i-1] = 1

	eigenvalues = torch.linalg.eig(comp)[0]
	return eigenvalues


def refractive_forward_projection(pos3d, n, d, mu):
	normal = -F.normalize(torch.cat([n, torch.ones((1,))], dim=0).view(1,3), dim=-1)
	por = F.normalize(torch.cross(normal, pos3d, dim=-1), dim=-1)

	z1 = -normal
	z2 = torch.cross(por, z1, dim=-1)

	v = torch.sum(pos3d * z1, dim=-1)
	u = torch.sum(pos3d * z2, dim=-1)

	coef1 = (mu - 1) * (mu + 1)
	coef2 = -2 * u * (mu - 1) * (mu + 1)
	coef3 = d * d * mu * mu - d * d + 2 * d * v + mu * mu * u * u - u * u - v * v
	coef4 = -2 * d * d * mu * mu * u
	coef5 = d * d * mu * mu * u * u
	coef1 = torch.ones_like(coef2) * coef1
	coefs = torch.stack([coef1, coef2, coef3, coef4, coef5], dim=-1)

	x0 = solve4th(coefs.double()).cfloat().T
	xx = x0.real.float()

	a = 1 / mu
	n2 = torch.tensor([0, -1])
	vi = torch.stack([xx, torch.ones_like(xx) * d], dim=-1)
	bb = torch.sum(vi * n2, dim=-1)
	bbb = torch.sum(vi * vi, dim=-1)

	b = -bb - torch.sqrt(bb * bb - (1 - mu * mu) * bbb)
	b /= mu

	vr = a * vi + b[...,np.newaxis] * n2
	vrd = torch.stack([u, v], dim=-1)[np.newaxis]
	vrd = vrd - vi

	error = torch.abs(vrd[...,0] * vr[...,1] - vrd[...,1] * vr[...,0])
	error[torch.abs(x0.imag) > 1e-3] = np.inf
	best = xx[torch.argmin(error, dim=0), np.arange(xx.shape[1])]
	return best[...,np.newaxis] * z2 + d * z1


def compute_essential_and_fundamental_matrices(cam_mat, proj_mat, cam_proj_pose):
	R = cam_proj_pose[:3, :3]
	t = cam_proj_pose[:3, 3]

	t_skew = np.array([
		[0, -t[2], t[1]],
		[t[2], 0, -t[0]],
		[-t[1], t[0], 0]
	])

	E = t_skew @ R
	F = np.linalg.inv(cam_mat).T @ E @ np.linalg.inv(proj_mat)

	return E, F


def detect_correspondences(img, pattern, cam_mat, proj_mat, cam_proj_pose, aruco_detector, epipolar_thresh):
	E, F = compute_essential_and_fundamental_matrices(cam_mat[:3,:3], proj_mat[:3,:3], cam_proj_pose)

	corners_img, ids_img, _ = aruco_detector.detectMarkers(img)
	corners_pattern, ids_pattern, _ = aruco_detector.detectMarkers(pattern)
	if ids_img is None or ids_pattern is None:
		return [], []

#	img_vis = img.copy()
#	cv2.aruco.drawDetectedMarkers(img_vis, corners_img, ids_img, (255, 0, 0))
#	cv2.imshow("", img_vis)
#	cv2.waitKey()

	common_ids = np.unique(np.intersect1d(ids_img, ids_pattern))
	corners_img = np.array([corners_img[np.argmax(ids_img == common_id)][0] for common_id in common_ids])
	corners_pattern = np.array([corners_pattern[np.argmax(ids_pattern == common_id)][0] for common_id in common_ids])
	corners_img = np.concatenate([corners_img, np.ones_like(corners_img[...,:1])], axis=-1)
	corners_pattern = np.concatenate([corners_pattern, np.ones_like(corners_pattern[...,:1])], axis=-1)

	corners_img_selected = []
	corners_pattern_selected = []
	ids_selected = []
	for id, corner_img, corner_pattern in zip(common_ids, corners_img, corners_pattern):
		error = np.mean(np.abs(np.diag(corner_img @ F @ corner_pattern.T)))
		if error < epipolar_thresh:
			corners_img_selected.append([corner_img[:,:2]])
			corners_pattern_selected.append([corner_pattern[:,:2]])
			ids_selected.append([id])
	corners_img_selected = np.array(corners_img_selected)
	corners_pattern_selected = np.array(corners_pattern_selected)
	ids_selected = np.array(ids_selected)

#	img_vis = img.copy()
#	cv2.aruco.drawDetectedMarkers(img_vis, corners_img_selected, ids_selected, (255, 0, 0))
#	cv2.imshow("", img_vis)
#	cv2.waitKey()

	return corners_img_selected, corners_pattern_selected


def triangulate(cam_mat, proj_mat, cam_proj_pose, pts1, pts2):
	P1 = cam_mat[:3]
	P2 = (proj_mat @ cam_proj_pose)[:3]

	points_hom = cv2.triangulatePoints(P1, P2, pts1.reshape(-1, 2).T, pts2.reshape(-1, 2).T)
	points3d = points_hom[:3] / points_hom[3]
	points3d = points3d.T.reshape(*pts1.shape[:-1], 3)
	return points3d


def augment_point_cloud(points3d, num_points=10000):
	vertices = points3d.reshape(-1, 3)
	triangles = []

	for i in range(0, len(vertices), 4):
		v0 = i + 0
		v1 = i + 1
		v2 = i + 2
		v3 = i + 3
		triangles.append([v0, v2, v1])
		triangles.append([v0, v3, v2])

	mesh = o3d.geometry.TriangleMesh()
	mesh.vertices = o3d.utility.Vector3dVector(vertices)
	mesh.triangles = o3d.utility.Vector3iVector(triangles)
#	mesh.compute_vertex_normals()

	pcd = mesh.sample_points_uniformly(number_of_points=num_points)
	return pcd



class IlluminationParams(nn.Module):
	def __init__(self, ambient, diffuse, emissive):
		super().__init__()
#		self.ambient = nn.Parameter(torch.tensor(np.clip(ambient, 0, 1) + 1e-3, dtype=torch.float))
		self.diffuse_denom = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
		self.emissive_denom = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
#		self.shadow = nn.Parameter(torch.tensor(1e-3, dtype=torch.float))

		self.register_buffer('ambient', torch.tensor(max(ambient, 0.1), dtype=torch.float))
#		self.register_buffer('ambient', torch.tensor(1.0, dtype=torch.float))
		self.register_buffer('diffuse_base', torch.tensor(diffuse, dtype=torch.float))
		self.register_buffer('emissive_base', torch.tensor(emissive, dtype=torch.float))


	def forward(self, image_idx):
		return {
			"ambient": self.ambient,
			"diffuse": self.diffuse_base / (F.relu(self.diffuse_denom) + 1e-8),
			"emissive": self.emissive_base / (F.relu(self.emissive_denom) + 1e-8),
#			"shadow": F.relu(self.shadow),
		}



class IlluminationParamsMulti(nn.Module):
	def __init__(self, illumination_params_list):
		super().__init__()

		self.illumination_params_list = nn.ParameterList(
			[IlluminationParams(**illum_params.item()) for illum_params in illumination_params_list]
		)


	def forward(self, image_idx):
		return self.illumination_params_list[image_idx](image_idx)



class FrameWeights(nn.Module):
	def __init__(self, num_images, initial_weights=None):
		super().__init__()
		if initial_weights is not None:
			initial_weights = torch.from_numpy(initial_weights)
		else:
			initial_weights = torch.ones(num_images, dtype=torch.float)
		self.weights = nn.Parameter(initial_weights)


	def forward(self, image_idx, coords=None):	# coords is just a dummy
		weights = F.softmax(self.weights, dim=0)
		return weights[image_idx][None,None]



class PixelwiseFrameWeights(nn.Module):
	def __init__(self, num_images, width, height, initial_weights=None, num_levels=3):
		super().__init__()
		self.num_images = num_images
		self.levels = nn.ParameterList()
		self.level_shapes = []

		if initial_weights is not None:
			assert len(initial_weights) == num_images
			initial_weights = torch.from_numpy(initial_weights).float()
		else:
			initial_weights = torch.ones(num_images)

		for i in range(num_levels):
			scale = 2 ** i
			w, h = width // scale, height // scale
			self.level_shapes.append((h, w))

			level_weight = initial_weights.view(num_images, 1, 1).expand(num_images, h, w).clone()
			param = nn.Parameter(level_weight)
			self.levels.append(param)


	def forward(self, image_idx, coords):
		upscaled_weights = []
		for i, param in enumerate(self.levels):
			h, w = self.level_shapes[i]
			soft = F.softmax(param, dim=0)  # across frames
			upsampled = F.interpolate(soft.unsqueeze(1), size=self.level_shapes[0], mode='bilinear', align_corners=False)
			upscaled_weights.append(upsampled.squeeze(1))  # (N, H, W)

		merged = torch.stack(upscaled_weights, dim=0).mean(dim=0)  # (N, H, W)

		return merged[image_idx, coords[1], coords[0]][:, None]



class RefractiveInterface(nn.Module):
	def __init__(self, normal=[0.0,0.0], depth=0.0, mu=1.0, scale=1.0):
		super().__init__()
		self.n = nn.Parameter(torch.from_numpy(np.float32(normal)))
		self.d = nn.Parameter(torch.tensor(float(depth)))
		self.register_buffer('mu', torch.tensor(float(mu)))
		self.register_buffer('scale', torch.tensor(float(scale)))


	def backward_projection(self, rays_o, rays_v):
		normal = torch.cat([self.n, torch.ones((1,)).to(rays_o)], dim=0)
		distance_to_interface = compute_distance_to_plane(rays_o, rays_v, normal, normal * F.relu(self.d) / self.scale)
		rays_o = rays_o + rays_v * distance_to_interface[...,np.newaxis]
		rays_v = refract(rays_v, normal, self.mu)
		return rays_o, rays_v


	def forward_projection(self, pos3d):
		return refractive_forward_projection(pos3d, self.n, F.relu(self.d) / self.scale + 1e-3, self.mu)



class PoseMLP(nn.Module):
	def __init__(self, rots, tvecs, pretrain_steps=1000, lr=5e-4, device='cuda', fourier_dim=0, use_isolated_buffer=False):
		super().__init__()

		self.device = device
		self.pose_num = rots.shape[0]
		self.rot_dim = rots.shape[-1]
		self.tvec_dim = tvecs.shape[-1]
		self.fourier_dim = fourier_dim

		if fourier_dim > 0:
			input_dim = fourier_dim * 2 + 1
		else:
			input_dim = 128
			self.start_feat = nn.Parameter(torch.randn(1, input_dim))
			self.end_feat = nn.Parameter(torch.randn(1, input_dim))

		self.pose_mlp = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.ReLU(),
			nn.Linear(128, self.rot_dim + self.tvec_dim)
		).to(device)

		with torch.no_grad():
			for m in self.pose_mlp:
				if isinstance(m, nn.Linear):
					nn.init.normal_(m.weight, mean=0, std=1e-6)
					nn.init.zeros_(m.bias)

		self.use_isolated_buffer = use_isolated_buffer
		if self.use_isolated_buffer:
			self.register_buffer("pose_embedding", torch.cat([rots, tvecs], dim=-1).clone())
		else:
			self._pretrain_pose_mlp(rots, tvecs, pretrain_steps, lr)


	def encode(self, indices: torch.Tensor):
		"""
		indices: shape (N, 1)
		return: shape (N, fourier_dim * 2)
		"""
		x = indices / (self.pose_num - 1)

		if self.fourier_dim > 0:
			freq_bands = 2 ** torch.arange(self.fourier_dim, device=self.device).float() * np.pi
			# (N, 1) x (freq_dim,) → (N, freq_dim)
			angles = x * freq_bands
			return torch.cat([x, torch.sin(angles), torch.cos(angles)], dim=-1)
		else:
			return (1 - x) * self.start_feat + x * self.end_feat


	def _pretrain_pose_mlp(self, rots, tvecs, steps, lr):
		print("Pretraining Pose MLP")
		device = self.device
		target = torch.cat([rots, tvecs], dim=-1).to(device)

		optimizer = optim.Adam(self.pose_mlp.parameters(), lr=lr)
		loss_fn = nn.MSELoss()

		indices = torch.arange(self.pose_num, device=device).float().unsqueeze(-1)

		for i in tqdm(range(steps), desc="Pretraining", ncols=80):
			inputs = self.encode(indices)
			pred = self.pose_mlp(inputs)
			loss = loss_fn(pred, target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i) % max(1, steps // 10) == 0:
				tqdm.write(f"Step [{i+1}/{steps}] Loss: {loss.item():.6f}")

		self.pose_mlp.eval()


	def forward(self, indices: torch.Tensor):
		inputs = self.encode(indices.float().unsqueeze(-1))
		out = self.pose_mlp(inputs)
		if self.use_isolated_buffer:
			out = out + self.pose_embedding[indices]
		return out


	def get_rot(self, indices):
		out = self.forward(indices)
		return out[:, :self.rot_dim]


	def get_tvec(self, indices):
		out = self.forward(indices)
		return out[:, self.rot_dim:]



class OpticalDevice(nn.Module):
	def __init__(self, intrinsic, extrinsic, reference_index, resolution, consistent_images, dynamic=False, noise=None, use_cache=True, 
		rot_type="quaternion", rots_sigma=0.0, tvecs_sigma=0.0, scale=1.0, estimate_intrinsic=False, estimate_pose=False, use_mlp_for_pose=False):
		super().__init__()
		self.reference_index = reference_index
		self.resolution = resolution
		self.register_buffer('intrinsic_gt', intrinsic)
		self.register_buffer('intrinsic_inv_gt', torch.inverse(intrinsic))

		# extrinsic is c2w
		rots_gt = extrinsic[...,:3,:3]
		tvecs_gt = extrinsic[...,:3,3] * scale
		self.register_buffer('rots_gt', camera.Quaternion().R_to_q(rots_gt))
		self.register_buffer('tvecs_gt', tvecs_gt)
		self.max_trans = torch.linalg.norm(self.tvecs_gt[1:] - self.tvecs_gt[:-1], dim=-1).mean().item()

		rots = rots_gt.clone().cpu().detach().numpy()
		tvecs = tvecs_gt.clone().cpu().detach().numpy()

		self.dist = None

		if noise is not None:
			for key in noise:
				assert key in ["focus", "center", "dist", "translation", "rotation", "consective"]

			def apply_rotation_noise(rots, noise_deg):
				orig_shape = rots.shape  # (N, M, 3, 3) or (N, 3, 3)
				rots_reshaped = rots.reshape(-1, 3, 3)  # (N*M, 3, 3)
				rot_noise = (np.random.rand(len(rots_reshaped), 3) * 2 - 1) * noise_deg
				rot_noise_matrix = Rot.from_euler('zxy', rot_noise, degrees=True).as_matrix()  # (N*M, 3, 3)
				rots_noised = rot_noise_matrix @ rots_reshaped
				return rots_noised.reshape(orig_shape)

			def add_uniform_fov_noise_to_intrinsics(intrinsic_matrix, image_width, image_height, noise_x_deg, noise_y_deg):
				fx = intrinsic_matrix[..., 0, 0].cpu().numpy()
				fy = intrinsic_matrix[..., 1, 1].cpu().numpy()

				fov_x = 2 * np.arctan(0.5 * image_width / fx)
				fov_y = 2 * np.arctan(0.5 * image_height / fy)

#				noise_x_deg = np.random.uniform(-fov_noise_range_deg, fov_noise_range_deg, size=fx.shape)
#				noise_y_deg = np.random.uniform(-fov_noise_range_deg, fov_noise_range_deg, size=fy.shape)

				fov_x_noisy = np.deg2rad(np.rad2deg(fov_x) + noise_x_deg)
				fov_y_noisy = np.deg2rad(np.rad2deg(fov_y) + noise_y_deg)

				fx_noisy = 0.5 * image_width / np.tan(0.5 * fov_x_noisy)
				fy_noisy = 0.5 * image_height / np.tan(0.5 * fov_y_noisy)

				intrinsic_noisy = intrinsic_matrix.clone()
				intrinsic_noisy[..., 0, 0] = torch.from_numpy(fx_noisy).to(intrinsic_matrix.device, intrinsic_matrix.dtype)
				intrinsic_noisy[..., 1, 1] = torch.from_numpy(fy_noisy).to(intrinsic_matrix.device, intrinsic_matrix.dtype)

				return intrinsic_noisy

			if "focus" in noise:
				intrinsic = add_uniform_fov_noise_to_intrinsics(intrinsic, *self.resolution, 
					noise_x_deg=noise["focus"]["x"] if "x" in noise["focus"] else 0, 
					noise_y_deg=noise["focus"]["y"] if "y" in noise["focus"] else 0)

			if "center" in noise:
				if "x" in noise["center"]:
					intrinsic[..., 0, 2] += noise["center"]["x"]
				if "y" in noise["center"]:
					intrinsic[..., 1, 2] += noise["center"]["y"]

			if "dist" in noise:
				self.dist = noise["dist"]

			if "rotation" in noise:
				rots = apply_rotation_noise(rots, noise["rotation"])

			if "translation" in noise:
				trans_noise = (np.random.rand(*tvecs.shape) * 2 - 1) * noise["translation"] * self.max_trans / 100
				tvecs = tvecs + trans_noise

			if "consective" in noise:
				orig_shape = rots.shape
				N = orig_shape[0]
				M = orig_shape[1] if len(orig_shape) == 4 else 1

				rots_reshaped = rots.reshape(N * M, 3, 3)
				tvecs_reshaped = tvecs.reshape(N * M, 3)

				rot_noise = (np.random.rand(N * M, 3) * 2 - 1) * noise["consective"]["rotation"]
				rot_noise_matrix = Rot.from_euler('zxy', rot_noise, degrees=True).as_matrix()

				trans_noise = (np.random.rand(*tvecs_reshaped.shape) * 2 - 1) * noise["consective"]["translation"] * self.max_trans / 100

				rot_list = []
				tvec_list = []
				current_rot = rots_reshaped[0]
				current_tvec = tvecs_reshaped[0]
				rot_list.append(torch.tensor(current_rot))
				tvec_list.append(torch.tensor(current_tvec))

				for i in range(N * M - 1):
					current_rot = rot_noise_matrix[i] @ rots_reshaped[i + 1] @ np.linalg.inv(rots_reshaped[i]) @ current_rot
					current_tvec += (current_rot @ np.linalg.inv(rots_reshaped[i])) @ (trans_noise[i] + tvecs_reshaped[i + 1] - tvecs_reshaped[i])
					rot_list.append(torch.tensor(current_rot.copy()))
					tvec_list.append(torch.tensor(current_tvec.copy()))

				rots = torch.stack(rot_list, dim=0).reshape(orig_shape)
				tvecs = torch.stack(tvec_list, dim=0).reshape(tvecs.shape)

		if dynamic:
#			assert not use_cache, "use_cache is not compatible with dynamic."

			for i in range(len(rots)):
				rots[i] = rots[0]
				tvecs[i] = tvecs[0]

		intrinsic_params = torch.stack([intrinsic[...,0,0], intrinsic[...,0,2], intrinsic[...,1,1], intrinsic[...,1,2]], dim=-1)
		if estimate_intrinsic:
			self.intrinsic = nn.Parameter(intrinsic_params)
		else:
			self.register_buffer('intrinsic', intrinsic_params)

		self.rot_type = rot_type

		rots = torch.from_numpy(rots).float()
		if self.rot_type == "quaternion":
			rots = camera.Quaternion().R_to_q(rots)
		elif self.rot_type == "rotation_vector":
			rots = camera.Lie().SO3_to_so3(rots)
		elif self.rot_type == "euler_angle":
			rots = matrix_to_euler_angles(rots, convention="ZXY")
		else:
			raise NotImplementedError
		tvecs = torch.from_numpy(tvecs).float()

		if estimate_pose:
			self.use_mlp_for_pose = use_mlp_for_pose
			if self.use_mlp_for_pose:
				self.pose_mlp = PoseMLP(rots, tvecs).to(rots)
			else:
				self.rots_ = nn.Parameter(rots)
				self.tvecs_ = nn.Parameter(tvecs)
		else:
			self.use_mlp_for_pose = False
			self.register_buffer('rots_', rots)
			self.register_buffer('tvecs_', tvecs)

#		self.rots_sigma = nn.Parameter(torch.ones_like(rots) * rots_sigma)
#		self.tvecs_sigma = nn.Parameter(torch.ones_like(tvecs) * tvecs_sigma)

		self.consistent_images = consistent_images
		self.use_cache = use_cache
		self.pose_cahce = {}
		self.pose_inv_cahce = {}
		self.rot_cahce = {}
		self.rot_inv_cahce = {}
		self.tvec_cahce = {}
		self.tvec_inv_cahce = {}


	def _normalize_indices(self, indices):
		# int -> [int]
		if isinstance(indices, int):
			indices = torch.tensor([indices], dtype=torch.long)
		elif isinstance(indices, torch.Tensor):
			if indices.ndim == 0:  # 0-dim tensor -> 1-dim tensor
				indices = indices.unsqueeze(0)
			indices = indices.long()
		else:
			indices = torch.tensor(indices, dtype=torch.long)
		return indices


	def rots(self, indices):
#		rot = torch.normal(mean=self.rots_[indices].unsqueeze(0), std=self.rots_sigma[indices].unsqueeze(0))
		if self.use_mlp_for_pose:
			indices = self._normalize_indices(indices)
			rot = self.pose_mlp.get_rot(indices)
		else:
			rot = self.rots_[indices].unsqueeze(0)

#		if indices == self.reference_index:
#			rot = rot.detach()

		if self.rot_type == "quaternion":
			return camera.Quaternion().q_to_R(rot)[0]
		elif self.rot_type == "rotation_vector":
			return camera.Lie().so3_to_SO3(rot)[0]
		elif self.rot_type == "euler_angle":
			return euler_angles_to_matrix(rot, convention="ZXY")[0]
		else:
			raise NotImplementedError


	def tvecs(self, indices):
#		tvecs = torch.normal(mean=self.tvecs_, std=self.tvecs_sigma)
		if self.use_mlp_for_pose:
			indices = self._normalize_indices(indices)
			tvecs = self.pose_mlp.get_tvec(indices).squeeze()
		else:
			tvecs = self.tvecs_[indices]

#		if indices == self.reference_index:
#			tvecs = tvecs.detach()

		return tvecs


	def get_intrinsic(self, indices):
		if self.consistent_images:
			indices_ = indices * 0
		else:
			indices_ = indices

		params = self.intrinsic[indices_]
		intrinsic_matrix = torch.zeros(*params.shape[:-1], 4, 4).to(params)
		intrinsic_matrix[...,0,0] = params[...,0]
		intrinsic_matrix[...,0,2] = params[...,1]
		intrinsic_matrix[...,1,1] = params[...,2]
		intrinsic_matrix[...,1,2] = params[...,3]
		intrinsic_matrix[...,2,2] = 1
		intrinsic_matrix[...,3,3] = 1
		return intrinsic_matrix


	def get_intrinsic_inv(self, indices):
		if self.consistent_images:
			indices_ = indices * 0
		else:
			indices_ = indices

		params = self.intrinsic[indices_]
		intrinsic_matrix = torch.zeros(*params.shape[:-1], 4, 4).to(params)
		intrinsic_matrix[...,0,0] = 1 / params[...,0]
		intrinsic_matrix[...,0,2] = -params[...,1] / params[...,0]
		intrinsic_matrix[...,1,1] = 1 / params[...,2]
		intrinsic_matrix[...,1,2] = -params[...,3] / params[...,2]
		intrinsic_matrix[...,2,2] = 1
		intrinsic_matrix[...,3,3] = 1
		return intrinsic_matrix


	def backward_projection(self, pos2d, indices, proj_index=None):
		mat = self.get_intrinsic_inv(indices)[...,:3,:3]
		if proj_index is not None:
			mat = mat[proj_index]
		ray_dir = torch.stack([
			pos2d[...,0] * mat[...,0,0] + mat[...,0,2], 
			pos2d[...,1] * mat[...,1,1] + mat[...,1,2], 
			torch.ones_like(pos2d[...,0])], dim=-1).float()
#		ray_dir = torch.matmul(pos2d_homo, mat.T)

		# undistort
		if self.dist is not None:
			k1, k2, p1, p2 = self.dist
			x = ray_dir[...,0]
			y = ray_dir[...,1]
			r2 = x**2 + y**2
			x_ = (1 + k1 * r2 + k2 * r2 ** 2) * x + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
			y_ = (1 + k1 * r2 + k2 * r2 ** 2) * y + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
			ray_dir[...,0] = x_
			ray_dir[...,1] = y_

		ray_dir = F.normalize(ray_dir, p=2, dim=-1)
		ray_origin = torch.zeros_like(ray_dir)

		return ray_origin, ray_dir


	def transform(self, pts, indices, proj_index=None):
		rot = self.get_rot(indices)
		tvec = self.get_tvec(indices)
		if proj_index is not None:
			rot = rot[proj_index]
			tvec = tvec[proj_index]
		pts = torch.matmul(rot, pts[..., None])[...,0]
		pts = pts + tvec.expand(pts.shape)
		return pts


	def transform_ray(self, ray_origin, ray_dir, indices, proj_index=None):
		rot = self.get_rot(indices)
		tvec = self.get_tvec(indices)
		if proj_index is not None:
			rot = rot[proj_index]
			tvec = tvec[proj_index]
		ray_dir = torch.matmul(rot, ray_dir[..., None])[...,0]
		ray_origin = torch.matmul(rot, ray_origin[..., None])[...,0]
		ray_origin = ray_origin + tvec.expand(ray_origin.shape)
		return ray_origin, ray_dir


	def get_pose(self, indices):
		if self.use_cache and indices in self.pose_cahce:
			return self.pose_cahce[indices]
		pose = torch.cat([self.get_rot(indices), self.get_tvec(indices).unsqueeze(-1)], dim=-1)
		pose = to_homo(pose)
		if self.use_cache:
			self.pose_cahce[indices] = pose
		return pose


	def get_pose_gt(self, indices):
		pose = torch.cat([self.get_rot_gt(indices), self.get_tvec_gt(indices).unsqueeze(-1)], dim=-1)
		pose = to_homo(pose)
		return pose


	def get_pose_inv(self, indices):
		if self.use_cache and indices in self.pose_inv_cahce:
			return self.pose_inv_cahce[indices]
		pose = torch.cat([self.get_rot_inv(indices), self.get_tvec_inv(indices).unsqueeze(-1)], dim=-1)
		pose = to_homo(pose)
		if self.use_cache:
			self.pose_inv_cahce[indices] = pose
		return pose


	def get_pose_inv_gt(self, indices):
		pose = torch.cat([self.get_rot_inv_gt(indices), self.get_tvec_inv_gt(indices).unsqueeze(-1)], dim=-1)
		pose = to_homo(pose)
		return pose


	def get_rot(self, indices):
		if self.use_cache and indices in self.rot_cahce:
			return self.rot_cahce[indices]
		if isinstance(indices, int) or indices.dim() == 0:
			rot = self.rots(indices)
		else:
			rot = gather_unique(indices, lambda indices: self.rots(indices))
		if self.use_cache:
			self.rot_cahce[indices] = rot
		return rot


	def get_rot_inv(self, indices):
		if self.use_cache and indices in self.rot_inv_cahce:
			return self.rot_inv_cahce[indices]
		rot = self.get_rot(indices).transpose(-1, -2)
		if self.use_cache:
			self.rot_inv_cahce[indices] = rot
		return rot


	def get_rot_gt(self, indices):
		if isinstance(indices, int) or indices.dim() == 0:
			rot = camera.Quaternion().q_to_R(self.rots_gt[indices].unsqueeze(0))[0]
		else:
			rot = gather_unique(indices, lambda indices: camera.Quaternion().q_to_R(self.rots_gt[indices]))
		return rot


	def get_rot_inv_gt(self, indices):
		return self.get_rot_inv_gt(indices).transpose(-1, -2)


	def get_tvec(self, indices):
		if self.use_cache and indices in self.tvec_cahce:
			return self.tvec_cahce[indices]
		tvec = self.tvecs(indices)
		if self.use_cache:
			self.tvec_cahce[indices] = tvec
		return tvec


	def get_tvec_inv(self, indices):
		if self.use_cache and indices in self.tvec_inv_cahce:
			return self.tvec_inv_cahce[indices]
		tvec = -(self.get_rot_inv(indices) @ self.tvecs(indices).unsqueeze(-1))[...,0]
		if self.use_cache:
			self.tvec_inv_cahce[indices] = tvec
		return tvec


	def get_tvec_gt(self, indices):
		tvec = self.tvecs_gt[indices]
		return tvec


	def get_tvec_inv_gt(self, indices):
		tvec = -(self.get_rot_inv_gt(indices) @ self.tvecs_gt[indices].unsqueeze(-1))[...,0]
		return tvec




class CameraParams(OpticalDevice):
	def __init__(self, n_images, camera_dict, reference_index, resolution, noise, consistent_images, dynamic=False, use_cache=True, 
		rot_type="quaternion", rots_sigma=0.0, tvecs_sigma=0.0, estimate_intrinsic=False, estimate_pose=False, use_mlp_for_pose=False):
		self.n_images = n_images
		self.dynamic = dynamic

		cam_mats_np = np.array([camera_dict['cam_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)])
		cam_poses_np = np.array([camera_dict['cam_pose_%d' % idx].astype(np.float32) for idx in range(self.n_images)])
		cam_poses_np_inv = np.linalg.inv(cam_poses_np)

		super().__init__(torch.from_numpy(cam_mats_np), torch.from_numpy(cam_poses_np_inv), reference_index, resolution,
			consistent_images, self.dynamic, noise, use_cache=use_cache, rot_type=rot_type, rots_sigma=rots_sigma, tvecs_sigma=tvecs_sigma,
			estimate_intrinsic=estimate_intrinsic, estimate_pose=estimate_pose, use_mlp_for_pose=use_mlp_for_pose)




class ProjectorParams(OpticalDevice):
	def __init__(self, n_images, camera_dict, reference_index, resolution, noise, consistent_images, use_cache=True, 
		rot_type="quaternion", rots_sigma=0.0, tvecs_sigma=0.0, baseline_scale=1.0, estimate_intrinsic=False, estimate_pose=False, use_mlp_for_pose=False):
		self.n_images = n_images

		proj_mats_np = np.array([camera_dict['proj_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)])
		proj_poses_np = np.array([camera_dict['cam_proj_pose_%d' % idx].astype(np.float32) for idx in range(self.n_images)])
		proj_poses_np_inv = np.linalg.inv(proj_poses_np)

		super().__init__(torch.from_numpy(proj_mats_np), torch.from_numpy(proj_poses_np_inv), reference_index, resolution,
			consistent_images, noise=noise, use_cache=use_cache, rot_type=rot_type, rots_sigma=rots_sigma, tvecs_sigma=tvecs_sigma,
			scale=baseline_scale, estimate_intrinsic=estimate_intrinsic, estimate_pose=estimate_pose, use_mlp_for_pose=use_mlp_for_pose)



class ProjectionPattern(nn.Module):
	def __init__(self, patterns, pattern_noise, consistent_patterns, consistent_projectors, 
		pattern_conv=None, pattern_conv_size=5, pattern_initial_blur=0, pattern_offset_size=None):
		super().__init__()
		self.n_images = patterns.shape[0]
		self.n_projectors = patterns.shape[1]
		height = patterns.shape[2]
		width = patterns.shape[3]
		self.consistent_patterns = consistent_patterns
		self.consistent_projectors = consistent_projectors

		if consistent_patterns:
			if consistent_projectors:
				patterns = patterns[0,0]
				color_correction = np.zeros((3,))
				affine_correction = np.zeros((1, 2, 3))
			else:
				patterns = patterns[0]
				color_correction = np.zeros((patterns.shape[0], 3,))
				affine_correction = np.zeros((patterns.shape[0], 2, 3))
		else:
			if consistent_projectors:
				patterns = patterns[:,0]
				color_correction = np.zeros((patterns.shape[0], 3,))
				affine_correction = np.zeros((patterns.shape[0], 2, 3))
			else:
				patterns = patterns
				color_correction = np.zeros((patterns.shape[0], patterns.shape[1], 3,))
				affine_correction = np.zeros((patterns.shape[0], patterns.shape[1], 2, 3))

		affine_correction[...,0,0] = 1
		affine_correction[...,1,1] = 1

#		self.color_correction = nn.Parameter(torch.from_numpy(color_correction))
		self.register_buffer('affine_correction', torch.from_numpy(affine_correction))

		if pattern_noise is not None:
			if "gauss_noise" in pattern_noise:
				patterns *= np.random.rand(*patterns.shape) * pattern_noise["gauss_noise"] + (1 - pattern_noise)

			if "affine" in pattern_noise:
				def random_affine():
#					pts1 = np.float32([[0,0],[0,height],[width,0],[width,height]])
#					pts2 = np.float32(pts1 + np.random.uniform(low=-pattern_noise["affine"], high=pattern_noise["affine"], size=pts1.shape))
#					M = cv2.getPerspectiveTransform(pts1, pts2)
					sx = np.random.uniform(2 ** (-pattern_noise["affine"]), 2 ** (pattern_noise["affine"]))
					sy = np.random.uniform(2 ** (-pattern_noise["affine"]), 2 ** (pattern_noise["affine"]))
					tx = np.random.uniform(-width, width) * pattern_noise["affine"]
					ty = np.random.uniform(-height, height) * pattern_noise["affine"]
					shx = np.random.uniform(-1, 1) * pattern_noise["affine"]
					shy = np.random.uniform(-1, 1) * pattern_noise["affine"]

					M = np.float32([
						[sx, shx, tx],
						[shy, sy, ty],
					])
					return M

				if patterns.ndim == 5:
					for i in range(patterns.shape[0]):
						for j in range(patterns.shape[0]):
							patterns[i,j] = cv2.warpAffine(patterns[i,j], random_affine(), (patterns[i,j].shape[0], patterns[i,j].shape[1]))
				elif patterns.ndim == 4:
					for i in range(patterns.shape[0]):
						patterns[i] = cv2.warpAffine(patterns[i], random_affine(), (patterns[i].shape[0], patterns[i].shape[1]))
				else:
					patterns = cv2.warpAffine(patterns, random_affine(), (patterns.shape[0], patterns.shape[1]))


			if "color_space" in pattern_noise:
				def random_color_space_change(pattern):
					"""
					# HSV
					pattern = cv2.cvtColor((pattern * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
					pattern[...,0] = (pattern[...,0].astype(np.int32) + int(np.random.uniform(low=-pattern_noise["color_space"], high=pattern_noise["color_space"]))) % 360
					pattern = cv2.cvtColor(pattern, cv2.COLOR_HSV2BGR).astype(np.float32) / 255
					"""

#					pattern_gt = pattern.copy()
					mask = np.any(pattern > 0, axis=-1)
					rgb = np.random.uniform(low=-pattern_noise["color_space"], high=pattern_noise["color_space"], size=(3,))
					pattern = pattern + rgb * mask[...,np.newaxis]
					pattern = np.clip(pattern, 0, 1)

#					x = np.mean((pattern - pattern_gt)[mask], axis=0)
#					pattern += -x * mask[...,np.newaxis]
#					cv2.imshow("", pattern)
#					cv2.waitKey()

					return pattern

				if patterns.ndim == 5:
					for i in range(patterns.shape[0]):
						for j in range(patterns.shape[0]):
							patterns[i,j] = random_color_space_change(patterns[i,j])
				elif patterns.ndim == 4:
					for i in range(patterns.shape[0]):
						patterns[i] = random_color_space_change(patterns[i])
				else:
					patterns = random_color_space_change(patterns)


			if "blur" in pattern_noise:
				kernel = pattern_noise["blur"]
				if patterns.ndim == 5:
					for i in range(patterns.shape[0]):
						for j in range(patterns.shape[0]):
							patterns[i,j] = cv2.GaussianBlur(patterns[i,j], ksize=(kernel, kernel), sigmaX=0)
				elif patterns.ndim == 4:
					for i in range(patterns.shape[0]):
						patterns[i] = cv2.GaussianBlur(patterns[i], ksize=(kernel, kernel), sigmaX=0)
				else:
					patterns = cv2.GaussianBlur(patterns, ksize=(kernel, kernel), sigmaX=0)

		self.register_buffer('patterns', torch.from_numpy(patterns))
#		self.patterns = nn.Parameter(torch.from_numpy(patterns))

		k = pattern_conv_size
		self.pattern_conv_type = pattern_conv
		if pattern_conv == "DSC":
			self.pattern_conv = torch.nn.Conv2d(3, 3, (k, k), padding='same', groups=3)
		elif pattern_conv == "DC":
			assert pattern_offset_size is not None, "pattern_offset_size must be specified when pattern_conv == DC."
			self.pattern_conv = torchvision.ops.DeformConv2d(3, 3, (k, k), padding=(k//2,k//2), groups=3)
			self.deform_offset = nn.Parameter(torch.zeros((self.n_images, 2 * k * k, pattern_offset_size[1], pattern_offset_size[0])))
		elif pattern_conv == "DCNet":
			assert pattern_offset_size is not None, "pattern_offset_size must be specified when pattern_conv == DC."
			self.pattern_conv = torchvision.ops.DeformConv2d(3, 3, (k, k), padding=(k//2,k//2), groups=3)
			self.embedpattern_fn, input_dims = get_embedder("default", 6, input_dims=3)
			self.deform_offset_fn = nn.Linear(input_dims, 2 * k * k)
		elif pattern_conv == "Conv":
			self.pattern_conv = SingleConv((k, k), padding='same', bias=False)
		else:
			self.pattern_conv = None

		if self.pattern_conv is not None:
			torch.nn.init.normal_(self.pattern_conv.weight, mean=0.0, std=1e-4)
			if self.pattern_conv.bias is not None:
				torch.nn.init.constant_(self.pattern_conv.bias, 1e-4)
			with torch.no_grad():
				if pattern_initial_blur > 0:
					gaussian_kernel = create_gaussian_kernel(k, pattern_initial_blur)
					for out_ch in range(self.pattern_conv.weight.shape[0]):
						for in_ch in range(self.pattern_conv.weight.shape[1]):
							self.pattern_conv.weight[:, :] = gaussian_kernel.clone()
				else:
					self.pattern_conv.weight[:, :, k//2, k//2] = 1.0


	def correct_pattern(self, indices, apply_pattern_conv=True):
		pattern = self.patterns

		if not self.consistent_patterns:
			pattern = pattern[indices]

		if pattern.ndim == 3:
			pattern = pattern.unsqueeze(0)

		pattern = pattern.permute(0,3,1,2).float()

		if self.pattern_conv is not None and apply_pattern_conv:
			if self.pattern_conv_type == "DC":
				resized_offset = F.interpolate(self.deform_offset[indices][None], size=(pattern.shape[-2], pattern.shape[-1]), mode='nearest')
				pattern = self.pattern_conv(pattern, resized_offset)
			elif self.pattern_conv_type == "DCNet":
				B, C, H, W = pattern.shape
				device = pattern.device
				y_grid, x_grid = torch.meshgrid(
					torch.arange(H, device=device),
					torch.arange(W, device=device),
					indexing='ij'
				)  # (H, W), (H, W)
				grid = torch.stack([y_grid, x_grid], dim=-1)  # (H, W, 2)

				indices_expanded = indices.view(B, 1, 1, 1).expand(-1, H, W, -1)  # (B, H, W, 1)
				grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
				concat_input = torch.cat([grid, indices_expanded], dim=-1)  # (B, H, W, 3)

				embedded_indices = self.embedpattern_fn(concat_input)	# (B, H, W, X)
				offset = self.deform_offset_fn(embedded_indices).permute(0, 3, 1, 2)	# (B, N, H, W)
				pattern = self.pattern_conv(pattern, offset)
			else:
				pattern = self.pattern_conv(pattern)

		return pattern
#		return torch.tanh(self.patterns)
#		mask = torch.any(self.patterns > 0, dim=-1)
#		pattern = self.patterns + (self.color_correction[...,np.newaxis,np.newaxis,:] * mask[...,np.newaxis])
#		affine_grid = F.affine_grid(self.affine_correction, pattern.view(-1,*pattern.shape[-3:]).shape, align_corners=None)
#		pattern = F.grid_sample(pattern.view(-1,*pattern.shape[-3:]), affine_grid).view(*pattern.shape)
#		return pattern


	def get_pattern(self, indices, apply_pattern_conv=True):
		if self.consistent_projectors:
			return torch.stack([self.correct_pattern(indices, apply_pattern_conv)] * self.n_projectors, dim=1)
		else:
			return self.correct_pattern(indices, apply_pattern_conv).unsqueeze(0)



class Dataset:
	def __init__(self, conf, cam_noise={}, proj_noise={}, pattern_noise={}, reference_index=0,
				cam_refractive_interface_params=None, proj_refractive_interface_params=None, 
				event_mode=None, simulate_event=False, scene_scale=1.0, baseline_scale=1.0, dynamic=False, use_cam_cache=True, use_proj_cache=True,
				estimate_cam_intrinsic=False, estimate_cam_pose=False, estimate_proj_intrinsic=False, estimate_proj_pose=False, use_mlp_for_pose=False):
		super(Dataset, self).__init__()
		print('Load data: Begin')
		self.device = torch.device('cuda')
		self.conf = conf
		self.event_mode = event_mode
		self.simulate_event = simulate_event
		self.dynamic = dynamic

		self.data_dir = conf.get_string('data_dir')
		self.render_cameras_name = conf.get_string('render_cameras_name')
		self.object_cameras_name = conf.get_string('object_cameras_name')

		self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
		self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

		self.with_projection = conf.get_bool('with_projection', default=False)
		self.with_scatter = conf.get_bool('with_scatter', default=False)
		self.estimate_illum_per_image = conf.get_bool('estimate_illum_per_image', default=False)

		self.pixel_wise_frame_weights = conf.get_string('frame_weights_type', default="frame_wise") == "pixel_wise"
		self.median_blur = conf.get_int('median_blur', default=None)
		self.gaussian_blur = conf.get_int('gaussian_blur', default=None)
		self.quantinization = conf.get_bool('quantinization', default=False)
		self.binary = conf.get_bool('binary', default=False)

		json_path = os.path.join(self.data_dir, self.render_cameras_name.replace('npz', 'json'))
		if os.path.exists(json_path):
			with open(json_path) as f:
				camera_dict = recursive_to_ndarray(json.load(f))
		else:
			camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name), allow_pickle=True)

		self.camera_dict = camera_dict
		self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
		self.n_images = len(self.images_lis)

		if self.event_mode is not None:
			# Quantinaztion error handling
			self.event_scale = float(camera_dict.get("event_scale", 1.0))
			self.event_offset = float(camera_dict.get("event_offset", 128))
			self.images_np = np.stack([self.imread(im_name) for im_name in self.images_lis])
			self.images_np = (self.images_np - self.event_offset) / 255.0 / self.event_scale

			if self.simulate_event:
				self.images_np[1:] -= self.images_np[:-1]
				self.images_np[0] *= 0

			if self.event_mode != "sequential":
				img_tmp = np.zeros_like(self.images_np[0])
				accum_img_list = []
				for i in range(len(self.images_np)):
					img_tmp = self.images_np[i] + img_tmp
					accum_img_list.append(img_tmp.copy())
				accum_img_list = np.array(accum_img_list)
				self.images_np = accum_img_list

			self.event_offset /= 255
			self.images_np = (self.images_np * self.event_scale + self.event_offset).clip(0, 1)

		else:
			self.images_np = np.stack([self.imread(im_name) for im_name in self.images_lis]) / 255.0

		self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
		self.masks_np = np.stack([cv2.imread(im_name, cv2.IMREAD_COLOR) for im_name in self.masks_lis]) / 255.0

		if self.binary:
			self.images_np[self.masks_np > 0] += 0.5

		self.with_cam_refraction = conf.get_bool('with_cam_refraction', default="camera_refraction" in camera_dict)
		self.with_proj_refraction = conf.get_bool('with_proj_refraction', default="projector_refraction" in camera_dict)
		self._images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
		self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device)   # [n_images, H, W, 3]
		self.H, self.W = self._images.shape[1], self._images.shape[2]

		if self.with_projection:
			self.patterns_lis = []
			self.patterns_np = []
			for image_path in self.images_lis:
				frame = os.path.splitext(os.path.basename(image_path))[0]
				patterns_lis_tmp = sorted(glob(os.path.join(self.data_dir, "pattern/%s*.png" % frame)))
				patterns_np_tmp = np.stack([self.imread(im_name) for im_name in patterns_lis_tmp]) / 255.0
				self.patterns_lis.append(patterns_lis_tmp)
				self.patterns_np.append(patterns_np_tmp)
			self.patterns_np = np.stack(self.patterns_np)

		scale_mat_np = camera_dict['scale_mat_%d' % 0].astype(np.float32)
		scale_mat_np[:3,:3] *= scene_scale
		scale_mat = torch.from_numpy(scale_mat_np).to(self.device)
		self.scale_mat = scale_mat
		self.scale_mat_inv = torch.inverse(scale_mat)

		self.camera_params = CameraParams(self.n_images, camera_dict, reference_index, (self.W, self.W), cam_noise,
										conf.get_bool('consistent_images', default=True), self.dynamic, 
										use_cache=use_cam_cache, 
										rot_type=conf.get_string('cam_rot_type', default="rotation_vector"),
										rots_sigma=conf.get_float('cam_rots_sigma', default=0.0),
										tvecs_sigma=conf.get_float('cam_tvecs_sigma', default=0.0),
										estimate_intrinsic=estimate_cam_intrinsic,
										estimate_pose=estimate_cam_pose,
										use_mlp_for_pose=use_mlp_for_pose).to(self.device)
		if self.with_projection:
			self.projector_params = ProjectorParams(self.n_images, camera_dict, reference_index, (self.W, self.W), proj_noise, 
													conf.get_bool('consistent_images', default=True),
													use_cache=use_proj_cache, 
													rot_type=conf.get_string('proj_rot_type', default="rotation_vector"),
													rots_sigma=conf.get_float('proj_rots_sigma', default=0.0),
													tvecs_sigma=conf.get_float('proj_tvecs_sigma', default=0.0),
													estimate_intrinsic=estimate_proj_intrinsic,
													estimate_pose=estimate_proj_pose,
													baseline_scale=baseline_scale,
													use_mlp_for_pose=use_mlp_for_pose).to(self.device)

		if self.with_projection:
			self.projection_pattern = ProjectionPattern(
				self.patterns_np, pattern_noise,
				conf.get_bool('consistent_patterns', default=True), 
				conf.get_bool('consistent_projectors', default=True),
				conf.get_string('pattern_conv', default=None),
				conf.get_int('pattern_conv_size', default=5),
				conf.get_float('pattern_initial_blur', default=0),
				conf.get_list('pattern_offset_size', default=None)).to(self.device)

			if "illumination_params" in camera_dict:
				if self.estimate_illum_per_image:
					self.illumination_params = IlluminationParamsMulti([camera_dict["illumination_params"] for i in range(self.n_images)]).to(self.device)
				else:
					self.illumination_params = IlluminationParams(**camera_dict["illumination_params"].item()).to(self.device)
			else:
				self.illumination_params = IlluminationParamsMulti([camera_dict["illumination_params_%d" % i] for i in range(self.n_images)]).to(self.device)

			self.pH, self.pW = self.patterns_np.shape[2], self.patterns_np.shape[3]

		if self.with_cam_refraction:
			params = cam_refractive_interface_params if cam_refractive_interface_params is not None else camera_dict["camera_refraction"].item()
			self.cam_refractive_interface = RefractiveInterface(**params, scale=self.scale_mat[0,0]).to(self.device)

		if self.with_proj_refraction:
			params_list = proj_refractive_interface_params if proj_refractive_interface_params is not None else camera_dict["projector_refraction"]
			self.proj_refractive_interface = nn.ModuleList([RefractiveInterface(**params, scale=self.scale_mat[0,0]) for params in params_list]).to(self.device)

		self.image_pixels = self.H * self.W

		self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
		self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])

		self.use_aruco = conf.get_bool('use_aruco', default=False)
		if self.use_aruco:
			self.aruco_points = self.prepare_aruco_points()

		initial_weights = np.ones((self.n_images,), dtype=np.float32)
		if self.quantinization:
			for i, img in enumerate(self.images_np):
				interval = find_peak_interval(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY))
				initial_weights[i] = np.log(interval / 255)

		if self.pixel_wise_frame_weights:
			self.frame_weights = PixelwiseFrameWeights(self.n_images, self.W, self.H, initial_weights).to(self.device)
		else:
			self.frame_weights = FrameWeights(self.n_images, initial_weights).to(self.device)

		print('Load data: End')


	def imread(self, path):
		img = cv2.imread(path, cv2.IMREAD_COLOR)

		if self.median_blur is not None:
			img = cv2.medianBlur(img, ksize=self.median_blur)

		if self.gaussian_blur is not None:
			img = cv2.GaussianBlur(img, ksize=(self.gaussian_blur, self.gaussian_blur), sigmaX=0)

		if self.binary:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			img = img // 2
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

		return img


	def get_proj_params(self, img_indices, mode='cam'):
		if mode == 'cam':
			patterns = self.projection_pattern.get_pattern(img_indices).transpose(0,1)
			scale_mats = self.scale_mat.unsqueeze(0).expand(patterns.shape[0],*self.scale_mat.shape)
			cam_poses = self.camera_params.get_pose_inv(img_indices).view(1,-1,4,4).expand(patterns.shape[0],-1,4,4)
			proj_mats = self.projector_params.get_intrinsic(img_indices)
			proj_poses = self.projector_params.get_pose_inv(img_indices)
			proj_params = []
			for pattern, proj_mat, cam_pose, proj_pose, scale_mat in zip(patterns, proj_mats, cam_poses, proj_poses, scale_mats):
				proj_params.append({"pattern": pattern, "proj_mat": proj_mat, "cam_pose": cam_pose, "proj_pose": proj_pose, "scale_mat": scale_mat})

			if self.with_proj_refraction:
				for i in range(len(proj_params)):
					proj_params[i]["refractive_interface"] = self.proj_refractive_interface[i]

		else:
			patterns = self.images[img_indices].permute(2,0,1)[None,None]
			scale_mats = self.scale_mat[None,None]
			cam_poses = self.camera_params.get_pose_inv(img_indices)[None,None]
			proj_mats = self.camera_params.get_intrinsic(img_indices)[None,None]
			proj_poses = torch.eye(4).to(scale_mats).view(1,1,4,4)
			proj_params = []
			for pattern, proj_mat, cam_pose, proj_pose, scale_mat in zip(patterns, proj_mats, cam_poses, proj_poses, scale_mats):
				proj_params.append({"pattern": pattern, "proj_mat": proj_mat, "cam_pose": cam_pose, "proj_pose": proj_pose, "scale_mat": scale_mat})

			if self.with_cam_refraction:
				proj_params[0]["refractive_interface"] = self.cam_refractive_interface

		return proj_params


	def interpolate_pose(self, pose_0, pose_1, ratio):
		pose_0 = pose_0.cpu().detach().numpy()
		pose_1 = pose_1.cpu().detach().numpy()
		rots = Rot.from_matrix(np.stack([pose_0[:3, :3], pose_1[:3, :3]]))
		pose = np.eye(4).astype(np.float32)
		pose[:3, :3] = Slerp([0, 1], rots)(ratio).as_matrix()
		pose[:3, 3] = (1.0 - ratio) * pose_0[:3, 3] + ratio * pose_1[:3, 3]
		return torch.from_numpy(pose)


	def gen_random_pixels(self, batch_size, mode):
		pixels_x = torch.randint(low=0, high=self.W if mode == 'cam' else self.pW, size=[batch_size], device=self.device)
		pixels_y = torch.randint(low=0, high=self.H if mode == 'cam' else self.pH, size=[batch_size], device=self.device)
		return pixels_x, pixels_y


	@property
	def images(self):
		return self._images


	def get_frame_weight(self, image_idx, coords):
		return self.frame_weights(image_idx, coords)


	def compute_rays(self, img_idx, pixels, mode, proj_index=None):
		if mode == 'cam':
			rays_o, rays_v = self.camera_params.backward_projection(pixels, img_idx)
			if self.with_cam_refraction:
				rays_o, rays_v = self.cam_refractive_interface.backward_projection(rays_o, rays_v)
			rays_o, rays_v = self.camera_params.transform_ray(rays_o, rays_v, img_idx)
		elif mode == 'proj':
			rays_o, rays_v = self.projector_params.backward_projection(pixels, img_idx, proj_index=proj_index)
			if self.with_proj_refraction:
				rays_o, rays_v = self.proj_refractive_interface.backward_projection(rays_o, rays_v)
			rays_o, rays_v = self.projector_params.transform_ray(rays_o, rays_v, img_idx, proj_index=proj_index)
			rays_o, rays_v = self.camera_params.transform_ray(rays_o, rays_v, img_idx)
		else:
			raise NotImplementedError

		rays_o = rays_o @ self.scale_mat_inv[:3,:3].transpose(-1,-2) + self.scale_mat_inv[:3,3]

		return rays_o, rays_v


	def gen_rays_at(self, img_idx, resolution_level=1, mode='cam'):
		"""
		Generate rays at world space from one camera.
		"""
		l = resolution_level
		w = self.W if mode == 'cam' else self.pW
		h = self.H if mode == 'cam' else self.pH
		tx = torch.linspace(0, w - 1, int(w // l)).to(self.device)
		ty = torch.linspace(0, h - 1, int(h // l)).to(self.device)
		pixels_x, pixels_y = torch.meshgrid(tx, ty)
		pixels = torch.stack([pixels_x, pixels_y], dim=-1)

		if mode == 'cam':
			color = self.images[img_idx]
			proj_index = None
		else:
			color = self.projection_pattern.get_pattern(img_idx, apply_pattern_conv=False)[0]
			proj_index = 0

		rays_o, rays_v = self.compute_rays(img_idx, pixels, mode, proj_index=proj_index)

		return rays_o.transpose(0, 1), rays_v.transpose(0, 1), color.transpose(0, 1)


	def gen_random_rays_at(self, img_idx, batch_size, pixels=None, mode='cam'):
		"""
		Generate random rays at world space from one camera.
		"""
		if pixels is not None:
			pixels_x, pixels_y = pixels
		else:
			pixels_x, pixels_y = self.gen_random_pixels(batch_size, mode=mode)

		if mode == 'cam':
			color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
			mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
			proj_index = None
		else:
			proj_index = 0
			color = self.projection_pattern.get_pattern(img_idx, apply_pattern_conv=False)[0,proj_index].permute(1,2,0)[(pixels_y, pixels_x)]
			mask = torch.ones_like(color)

		pixels = torch.stack([pixels_x, pixels_y], dim=-1)

		rays_o, rays_v = self.compute_rays(img_idx, pixels, mode, proj_index=proj_index)

		return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1)    # batch_size, 10


	def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
		assert isinstance(idx_0, int) and isinstance(idx_1, int), "idx_0 and idx_1 must be int"
		"""
		Interpolate pose between two cameras.
		"""
		l = resolution_level
		w = self.W if mode == 'cam' else self.pW
		h = self.H if mode == 'cam' else self.pH
		tx = torch.linspace(0, w - 1, w // l)
		ty = torch.linspace(0, h - 1, h // l)
		pixels_x, pixels_y = torch.meshgrid(tx, ty)
		color = self.images[idx_0][(pixels_y, pixels_x)] * (1 - ratio) + self.images[idx_1][(pixels_y, pixels_x)] * ratio

		rays_o, rays_v = self.camera_params.backward_projection(torch.stack([pixels_x, pixels_y], dim=-1), idx_0)

		if self.with_cam_refraction:
			rays_o, rays_v = self.cam_refractive_interface.backward_projection(rays_o, rays_v)

		pose_0 = torch.linalg.inv(self.camera_params.get_pose(idx_0))
		pose_1 = torch.linalg.inv(self.camera_params.get_pose(idx_1))
		pose = torch.linalg.inv(self.interpolate_pose(pose_0, pose_1, ratio))
		rot = pose[:3, :3]
		trans = pose[:3, 3]

		rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None])[...,0]  # W, H, 3
		rays_o = torch.matmul(rot[None, None, :3, :3], rays_o[:, :, :, None])[...,0]  # W, H, 3
		rays_o += trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
		return rays_o.transpose(0, 1), rays_v.transpose(0, 1), color.transpose(0, 1)


	def gen_rays_satellite(self, rot, resolution_level=1):
		"""
		Generate rays at world space from one camera.
		"""
		l = resolution_level
		x = torch.linspace(-1, 1, int(self.W * l))
		y = torch.linspace(-1, 1, int(self.W * l))
		xx, yy = torch.meshgrid(x, y)
		rays_o = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1) # W, H, 3
		rays_v = torch.stack([torch.zeros_like(xx), torch.zeros_like(yy), -torch.ones_like(xx)], dim=-1) # W, H, 3

		rot = torch.from_numpy(rot).to(rays_o)
		rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None])[...,0]  # W, H, 3
		rays_o = torch.matmul(rot[None, None, :3, :3], rays_o[:, :, :, None])[...,0]  # W, H, 3

		return rays_o.transpose(0, 1), rays_v.transpose(0, 1), None


	def near_far_from_sphere(self, rays_o, rays_d):
		a = torch.sum(rays_d**2, dim=-1, keepdim=True)
		b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
		mid = 0.5 * (-b) / a
		near = mid - 1.0
		far = mid + 1.0
		return near, far


	def image_at(self, idx, resolution_level):
		img = self.images_np[idx] * 255
		return (cv2.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


	def mask_at(self, idx, resolution_level):
		mask = self.masks_np[idx] * 255
		return (cv2.resize(mask, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


	def patterns_at(self, idx, resolution_level):
		if not self.with_projection:
			return []
		patterns = self.patterns_np[idx] * 255
		return [(cv2.resize(pattern, (self.pW // resolution_level, self.pH // resolution_level))).clip(0, 255) for pattern in patterns]


	def aruco_points_at(self, idx, scale=True):
		if not self.use_aruco:
			return None

		pts = self.aruco_points[idx]
		if pts is not None:
			pts = self.camera_params.transform(pts, idx)
			if scale:
				pts = pts @ self.scale_mat_inv[:3,:3].transpose(-1,-2) + self.scale_mat_inv[:3,3]
		return pts


	def prepare_aruco_points(self, epipolar_thresh=1e-1):
		print("Initializing ARUCO points")
		aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
		detector_parameters = cv2.aruco.DetectorParameters()
#		detector_parameters.adaptiveThreshWinSizeStep = 1
		detector_parameters.adaptiveThreshConstant = 3
		detector_parameters.polygonalApproxAccuracyRate = 0.1
		detector_parameters.minCornerDistanceRate = 0.01
		detector_parameters.minMarkerPerimeterRate = 0.01
		aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_parameters)

		points_list = []

		for idx in range(self.n_images):
			img = self.image_at(idx, 1).astype(np.uint8)
			patterns = self.patterns_at(idx, 1)
			assert len(patterns) == 1
			pattern = patterns[0].astype(np.uint8)

			cam_mat = self.camera_params.get_intrinsic(idx).cpu().detach().numpy()
			proj_mat = self.projector_params.get_intrinsic(idx)[0].cpu().detach().numpy()
			cam_proj_pose = self.projector_params.get_pose_inv(idx)[0].cpu().detach().numpy()
			corners_img, corners_pattern = detect_correspondences(img, pattern, cam_mat, proj_mat, 
				np.linalg.inv(cam_proj_pose), aruco_detector, epipolar_thresh)

			print("Found %d points for image %d" % (len(corners_img), idx))
			if len(corners_img) > 0:
				points3d = triangulate(cam_mat, proj_mat, cam_proj_pose, 
					corners_img[:, 0], corners_pattern[:, 0]
				)
				pcd = augment_point_cloud(points3d)
				points_list.append(torch.from_numpy(np.float32(pcd.points)).to(self.device))
			else:
				points_list.append(None)

		return points_list


