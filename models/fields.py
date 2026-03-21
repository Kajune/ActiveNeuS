import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
from models.embedder import get_embedder
#from models.embedder_hash import get_embedder
from .utils import *
from . import camera


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
	def __init__(self,
				 max_time,
				 d_in,
				 d_out,
				 d_hidden,
				 n_layers,
				 skip_in=(4,),
				 multires=0,
				 encoding="default",
				 encoding_dropout=0.0,
				 bias=0.5,
				 scale=1,
				 geometric_init=True,
				 weight_norm=True,
				 inside_outside=False,
				 time_dim=0,
				 time_embedding_type=None,
				 time_encoding="default",
				 n_time_emb_mlps=[5,20],
				 concat_input_to_feature=False,
				 grad_mode="analytical",
				 normal_eps=1e-3,
		):
		super(SDFNetwork, self).__init__()

		d_in += time_dim

		dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

		self.embed_fn_fine = None
		self.time_embedding_type = time_embedding_type
		self.concat_input_to_feature = concat_input_to_feature
		self.grad_mode = grad_mode
		self.normal_eps = normal_eps

		if multires > 0:
			embed_fn, input_ch = get_embedder(encoding, multires, input_dims=d_in, dropout_ratio=encoding_dropout)
			self.embed_fn_fine = embed_fn
			dims[0] = input_ch

		if self.time_embedding_type is not None:
			assert time_dim > 0

			embed_fn_static, static_ch = get_embedder(time_encoding, multires, input_dims=d_in - time_dim, dropout_ratio=encoding_dropout)
			self.embed_fn_static = embed_fn_static

			if self.time_embedding_type == "interp":
				self.time_embedding = TimeEmbedding(static_ch, dims[0], max_time, num_mlps=n_time_emb_mlps)
			elif self.time_embedding_type == "grid":
				self.time_embedding = TimeEmbeddingGrid(static_ch, dims[0], max_time, num_mlps=n_time_emb_mlps)
			elif self.time_embedding_type == "cascade":
				self.time_embedding = TimeEmbeddingCascade(static_ch, dims[0], max_time, num_mlps=n_time_emb_mlps)
			else:
				raise NotImplementedError

			dims[0] = self.time_embedding.total_dim

		self.num_layers = len(dims)
		self.skip_in = skip_in
		self.scale = scale
		self.delta_size = 1e-3
#		self.register_buffer("numerical_delta", torch.from_numpy(np.float32([[-1,0,0],[0,-1,0],[0,0,-1],[1,0,0],[0,1,0],[0,0,1]]) * self.delta_size))

		for l in range(0, self.num_layers - 1):
#			if l + 1 in self.skip_in:
#				out_dim = dims[l + 1] - dims[0]
#			else:
#				out_dim = dims[l + 1]
#			in_dim = dims[l]

			if l in self.skip_in:
				in_dim = dims[l] + dims[0]
			else:
				in_dim = dims[l]
			out_dim = dims[l+1]

			lin = nn.Linear(in_dim, out_dim)

			if geometric_init:
				if l == self.num_layers - 2:
					if not inside_outside:
						torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
						torch.nn.init.constant_(lin.bias, -bias)
					else:
						torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
						torch.nn.init.constant_(lin.bias, bias)
				elif multires > 0 and l == 0:
					torch.nn.init.constant_(lin.bias, 0.0)
					torch.nn.init.constant_(lin.weight[:, d_in:], 0.0)
					torch.nn.init.normal_(lin.weight[:, :d_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
				elif multires > 0 and l in self.skip_in:
					torch.nn.init.constant_(lin.bias, 0.0)
					torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
					torch.nn.init.constant_(lin.weight[:, dims[l]:], 0.0)
				else:
					torch.nn.init.constant_(lin.bias, 0.0)
					torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.activation = nn.Softplus(beta=100)


	def forward(self, inputs, time, is_reference_frame=False):
		inputs = inputs * self.scale

		if self.time_embedding_type is not None:
			if self.embed_fn_fine is not None:
				inputs_static = self.embed_fn_static(inputs[...,:3]).float()
				inputs_dynamic = self.embed_fn_fine(inputs).float()
			else:
				inputs_static = inputs[...,:3]
				inputs_dynamic = inputs

			inputs = self.time_embedding(inputs_static, inputs_dynamic, time, is_reference_frame)
		else:
			if self.embed_fn_fine is not None:
				inputs = self.embed_fn_fine(inputs).float()

		x = inputs
		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			if l in self.skip_in:
				x = torch.cat([x, inputs], 1) / np.sqrt(2)

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.activation(x)

		outputs = [x[:, :1] / self.scale, x[:, 1:]]
		if self.concat_input_to_feature:
			outputs.append(inputs)
		return torch.cat(outputs, dim=-1)

	def sdf(self, x, time):
		return self.forward(x, time)[:, :1]

	def sdf_hidden_appearance(self, x, time):
		return self.forward(x, time)

	def gradient(self, x, time, compute_hessian=False):
		x.requires_grad_(True)
		sdf = self.sdf(x, time)

		if self.grad_mode == "analytical":
			gradients = torch.autograd.grad(sdf.sum(), x, create_graph=True)[0]
			if compute_hessian:
				hessians = torch.autograd.grad(gradients.sum(), x, create_graph=True)[0]
				return gradients.unsqueeze(1), hessians.unsqueeze(1)

			return gradients.unsqueeze(1)

		else:
			eps = self.normal_eps / np.sqrt(3)
			k_vectors = torch.tensor(
				[[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]],
				dtype=x.dtype, device=x.device
			)  # [4, 3]

			x_expanded = x.unsqueeze(0) + k_vectors.unsqueeze(1) * eps  # [4, N, 3]
			sdf_values = self.sdf(x_expanded.view(-1, x.shape[-1]), time).view(4, *x.shape[:-1], 1)  # [4, ..., 1]

			gradients = (k_vectors.unsqueeze(-2) * sdf_values).sum(dim=0) / (4.0 * eps)

			if compute_hessian:
				hessian_xx = (sdf_values.sum(dim=0) / 2.0 - 2 * sdf) / eps ** 2  # [..., 1]
				hessians = hessian_xx.expand(-1, 3) / 3.0  # [..., 3]
				return gradients.unsqueeze(1), hessians.unsqueeze(1)

			return gradients.unsqueeze(1)


	def update_embedder(self, progress):
		if not hasattr(self.embed_fn_fine, "update"):
			return {}
		else:
			return self.embed_fn_fine.update(progress)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
	def __init__(self,
				 d_feature,
				 mode,
				 d_in,
				 d_out,
				 d_hidden,
				 n_layers,
				 weight_norm=True,
				 multires=0,
				 encoding="default",
				 multires_view=0,
				 encoding_view="default",
				 squeeze_out=True,
				 scale_viewdir=False,
				 render_mode="view_dependent",
				 time_dim=0):
		super().__init__()

		self.mode = mode
		self.squeeze_out = squeeze_out
		self.scale_viewdir = scale_viewdir
		self.render_mode = render_mode
		pos_dim = 3
		pos_dim += time_dim
		d_in += time_dim

		dims_view_independent = [pos_dim + 3 + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
		dims_view_dependent = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

		self.embed_fn = None
		if encoding is not None:
			self.embed_fn, input_ch = get_embedder(encoding, multires, input_dims=pos_dim)
			dims_view_independent[0] += (input_ch - pos_dim)
			dims_view_dependent[0] += (input_ch - pos_dim)

		self.embedview_fn = None
		if encoding_view is not None:
			embedview_fn, input_ch = get_embedder(encoding_view, multires_view)
			self.embedview_fn = embedview_fn
			dims_view_dependent[0] += (input_ch - 3)

		self.num_layers = len(dims_view_dependent)

		layers = self.make_layers(dims_view_independent, weight_norm)
		self.view_independent_network = nn.Sequential(*layers)

		layers = self.make_layers(dims_view_dependent, weight_norm)
		self.view_dependent_network = nn.Sequential(*layers)


	def make_layers(self, dims, weight_norm):
		layers = []
		for l in range(0, len(dims) - 1):
			out_dim = dims[l + 1]
			in_dim = dims[l]
			lin = nn.Linear(in_dim, out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			layers.append(lin)

			if l < len(dims) - 2:
				layers.append(nn.ReLU())

		return layers


	def forward_view_independent(self, points, normals, feature_vectors):
		if self.mode == 'no_normal':
			rendering_input = torch.cat([points, feature_vectors], dim=-1)
		else:
			rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)

		x = rendering_input
		if self.embed_fn is not None:
			x = self.embed_fn(x)
		x = self.view_independent_network(x)
		if self.squeeze_out:
			x = torch.sigmoid(x)
		return x


	def forward_view_dependent(self, points, normals, view_dirs, feature_vectors):
		if not self.scale_viewdir:
			view_dirs = view_dirs / torch.linalg.norm(view_dirs, dim=-1)[...,np.newaxis]

		if self.embed_fn is not None:
			points = self.embed_fn(points)

		if self.embedview_fn is not None:
			view_dirs = self.embedview_fn(view_dirs).float()

		rendering_input = None

		if self.mode == 'idr':
			rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
		elif self.mode == 'no_view_dir':
			rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
		elif self.mode == 'no_normal':
			rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
		elif self.mode == 'no_pos':
			rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)

		x = rendering_input
		x = self.view_dependent_network(x)
		if self.squeeze_out:
			x = torch.sigmoid(x)
		return x


	def render_double(self, points, normals, view_dirs, feature_vectors):
		x_view_independent = self.forward_view_independent(points, normals, feature_vectors)
		x_view_dependent = self.forward_view_dependent(points, normals, view_dirs, feature_vectors)
		x = x_view_independent + x_view_dependent
		return x


	def forward(self, points, normals, view_dirs, feature_vectors):
		if self.render_mode == "view_independent":
			ret = self.forward_view_independent(points, normals, feature_vectors)

		elif self.render_mode == "view_dependent":
			ret = self.forward_view_dependent(points, normals, view_dirs, feature_vectors)

		elif self.render_mode == "double":
			ret = self.render_double(points, normals, view_dirs, feature_vectors)

		elif self.render_mode.startswith("random_"):
			p = float(self.render_mode[len("random_"):])
			if np.random.uniform() <= p:
				ret = self.forward_view_independent(points, normals, feature_vectors)
			else:
				ret = self.render_double(points, normals, view_dirs, feature_vectors)

		return ret



class ProjectionNetwork(nn.Module):
	def __init__(self, 
				 mode,
				 d_feature,
				 d_hidden,
				 n_layers,
				 weight_norm=True,
				 multires=0,
				 encoding="default",
				 multires_view=0,
				 encoding_view="default",
				 pattern_supervision="supervised",
				 blend_dropout_ratio=0.0,
				 grayscale=False,
				 pattern_conv=None,
				 no_grad=True,
				 time_dim=0,
				 pattern_offset_size=None,
				 n_images=None):
		super().__init__()
		self.mode = mode
		self.pattern_supervision = pattern_supervision
		self.grayscale = grayscale
		self.blend_dropout_ratio = blend_dropout_ratio
		self.no_grad = no_grad
		self.pos_dim = 3 + time_dim

		if pattern_conv == "DSC":
			k = 5
			self.pattern_conv = torch.nn.Conv2d(3, 3, (k, k), padding='same', groups=3)
		elif pattern_conv == "DC":
			assert pattern_offset_size is not None, "pattern_offset_size must be specified when pattern_conv == DC."
			assert n_images is not None, "n_images must be specified when pattern_conv == DC."
			k = 5
			self.pattern_conv = torchvision.ops.DeformConv2d(3, 3, (k, k), padding=(k//2,k//2), groups=3)
			self.deform_offset = nn.Parameter(torch.zeros((n_images, 2 * k * k, pattern_offset_size[1], pattern_offset_size[0])))
		elif pattern_conv == "Conv":
			k = 1
			self.pattern_conv = SingleConv((k, k), padding='same', bias=False)
		else:
			self.pattern_conv = None

		if self.pattern_conv is not None:
			torch.nn.init.normal_(self.pattern_conv.weight, mean=0.0, std=1e-4)
			if self.pattern_conv.bias is not None:
				torch.nn.init.constant_(self.pattern_conv.bias, 1e-4)
			with torch.no_grad():
				self.pattern_conv.weight[:, :, k//2, k//2] = 1.0

		self.nerf_illum_param = nn.Parameter(torch.from_numpy(np.float32([1,1,1e-6])))

#		assert (self.pattern_supervision in ["semisupervised", "unsupervised"] and self.mode in ["implicit", "implicit_w_blend"]) or self.pattern_supervision == "supervised", \
#			"un/semisupervised projection is only available with implicit!"
		assert (self.blend_dropout_ratio == 0.0 or self.mode in ["implicit_w_blend", "original", "explicit", "explicit_w_ambient"]), \
			"blend_dropout_ratio is only effective with blended modes!"

		if self.mode in ["explicit", "explicit_w_ambient"]:
			dims = [self.pos_dim + d_feature]
			self.embed_fn = None
			if encoding is not None:
				self.embed_fn, input_ch = get_embedder(encoding, multires, input_dims=self.pos_dim)
				dims[0] += (input_ch - self.pos_dim)

			self.explicit_projection_network = nn.Linear(dims[0], 2)

			if self.mode == "explicit_w_ambient":
				self.ambient_bias = nn.Parameter(torch.from_numpy(np.float32([0,0,0])))

			self.embedpattern_fn = None
			if self.pattern_supervision in ["semisupervised", "unsupervised", "weaksupervised"]:
				embedpattern_fn, input_ch = get_embedder("hybrid", 10, input_dims=2)
				self.embedpattern_fn = embedpattern_fn
				if self.pattern_supervision in ["semisupervised", "weaksupervised"]:
					input_ch += 3	# for color

			self.pattern_decoder = nn.Linear(input_ch, 3)


		elif self.mode in ["implicit", "implicit_w_blend"]:
			dims = [self.pos_dim + 3 + 3 + 3 + 3 + d_feature + 1] + [d_hidden for _ in range(n_layers)] + [3]

			self.embedview_fn = None
			if encoding_view is not None:
				embedview_fn, input_ch = get_embedder(encoding_view, multires_view, input_dims=3)
				self.embedview_fn = embedview_fn
				dims[0] += (input_ch - 3)	# for view_dirs

			self.embedpattern_fn = None
			if self.pattern_supervision in ["semisupervised", "unsupervised", "weaksupervised"]:
				embedpattern_fn, input_ch = get_embedder("hybrid", 10, input_dims=2)
				self.embedpattern_fn = embedpattern_fn
				if self.pattern_supervision in ["semisupervised", "weaksupervised"]:
					dims[0] += input_ch	# for pattern
				elif self.pattern_supervision in ["unsupervised"]:
					dims[0] += input_ch	- 3 # for pattern

#				embedpattern_fn_shadow, input_ch = get_embedder("hybrid", 10, input_dims=2)
#				self.embedpattern_fn_shadow = embedpattern_fn_shadow

			self.num_layers = len(dims)

			layers = self.make_layers(dims, weight_norm)
			self.implicit_projection_network = nn.Sequential(*layers)


	def make_layers(self, dims, weight_norm):
		layers = []
		for l in range(0, len(dims) - 1):
			out_dim = dims[l + 1]
			in_dim = dims[l]
			lin = nn.Linear(in_dim, out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			layers.append(lin)

			if l < len(dims) - 2:
				layers.append(nn.ReLU())

		return layers


	def prepare_proj_params(self, pts_homo, indices, proj_param, shadow_map, scatter_density=None, scatter_color=None):
#		if self.no_grad:
#			no_grad = torch.no_grad()
#			no_grad.__enter__()

		proj_mat = proj_param["proj_mat"].to(pts_homo)
		proj_pose = proj_param["proj_pose"].to(pts_homo)
		cam_pose = proj_param["cam_pose"].to(pts_homo)
		scale_mat = proj_param["scale_mat"].to(pts_homo)
		proj_world_mat = torch.matmul(proj_pose, torch.matmul(cam_pose, scale_mat))
#		proj_world_mat = proj_param["proj_world_mat"].to(pts_homo)
		pattern = proj_param["pattern"].to(pts_homo).permute(0,3,1,2)

		input_dirs = F.normalize(cam_pose[:,:3,3][:,None] - pts_homo[...,:3], dim=-1)

		pts_homo_trans = torch.matmul(proj_world_mat, pts_homo.transpose(1,2)).transpose(1,2).reshape(*pts_homo.shape)
		pts_trans = pts_homo_trans[...,:3]
		distance_to_proj = pts_trans[...,2]
		is_forward = distance_to_proj >= 0

		if "refractive_interface" in proj_param:
			pts_trans = proj_param["refractive_interface"].forward_projection(pts_trans.view(-1,3)).view(*pts_trans.shape)

		pts_trans = pts_trans / (distance_to_proj[...,None] + 1e-8)
		pts_trans = torch.matmul(proj_mat[...,:3,:3], pts_trans.transpose(1,2)).transpose(1,2).view(*pts_trans.shape)

		# Interpolation
		pts_trans[...,0] = pts_trans[...,0] / pattern.shape[-1] * 2 - 1
		pts_trans[...,1] = pts_trans[...,1] / pattern.shape[-2] * 2 - 1

#		if self.no_grad:
#			no_grad.__exit__(None, None, None)

		if self.pattern_conv is not None:
			if isinstance(self.pattern_conv, torchvision.ops.DeformConv2d):
#				positional_embed = create_positional_encoding(pattern.shape[-1], pattern.shape[-2]).unsqueeze(0).to(pattern)
				resized_offset = F.interpolate(self.deform_offset[indices][None], size=(pattern.shape[-2], pattern.shape[-1]), mode='nearest')
				pattern = self.pattern_conv(pattern, resized_offset)
			else:
				pattern = self.pattern_conv(pattern)

		projected_color = F.grid_sample(pattern.expand(len(pts_trans), *pattern.shape[1:]), pts_trans[...,:2].unsqueeze(1))
		projected_color = projected_color[:,:,0].transpose(-1,-2) * is_forward[...,None]

		# Attenuation
		if scatter_density is not None and scatter_color is not None:
			projected_color = projected_color * torch.exp(-scatter_color * scatter_density[...,None] * distance_to_proj[...,None])

		projected_color = projected_color.reshape(-1,3)

		if self.pattern_supervision == "semisupervised":
			projected_color = torch.cat([self.embedpattern_fn(pts_trans[...,:2].reshape(-1,2)), projected_color], dim=-1)
			if self.mode in ["original", "explicit", "explicit_w_ambient"]:
				projected_color = self.pattern_decoder(projected_color)
		elif self.pattern_supervision == "weaksupervised":
			if self.mode in ["original", "explicit", "explicit_w_ambient"]:
				projected_color = projected_color + self.pattern_decoder(self.embedpattern_fn(pts_trans[...,:2].reshape(-1,2)))
			else:
				projected_color = torch.cat([self.embedpattern_fn(pts_trans[...,:2].reshape(-1,2)), projected_color], dim=-1)
		elif self.pattern_supervision == "unsupervised":
			projected_color = self.embedpattern_fn(pts_trans[...,:2].reshape(-1,2))
			if self.mode in ["original", "explicit", "explicit_w_ambient"]:
				projected_color = self.pattern_decoder(projected_color)

		if shadow_map is not None:
			projected_color = projected_color * shadow_map[...,None]

		return input_dirs, projected_color, pts_trans


	def forward(self, points, points_timed, indices, normals, view_dirs, feature_vectors, proj_params, color_network, illum_params, 
		shadow_maps=None, wo_pattern=False, scatter_density=None, scatter_color=None):
		projected_colors = []
		pts_homo = torch.cat([points, torch.ones_like(points[...,:1]).to(points)], dim=-1)

		if normals is not None:
			normals = F.normalize(normals, dim=-1)

		for pi, proj_param in enumerate(proj_params):
			input_dirs, projected_color, pts_trans = self.prepare_proj_params(pts_homo, indices, proj_param,
				shadow_maps[pi] if shadow_maps is not None else None, scatter_density, scatter_color)

			if self.mode == "original":
				projected_colors.append(projected_color)

			elif self.mode in ["explicit", "explicit_w_ambient"]:
				points_timed = points_timed.reshape(-1,self.pos_dim)
				input_dirs_ = input_dirs.view(-1,3)
				if self.embed_fn is not None:
					points_timed = self.embed_fn(points_timed)

				x = torch.cat([points_timed, feature_vectors], dim=-1)
				x = self.explicit_projection_network(x)
				reflectance_ratio, roughness = x[...,0], x[...,1]
				reflectance_ratio = torch.sigmoid(reflectance_ratio)[...,None]
#				roughness = F.relu(roughness)

				dot = (input_dirs_ * normals).sum(dim=-1)[...,None]
#				reflection_vector = 2 * normals * dot - input_dirs_
				color_lambert = (1 - reflectance_ratio) * projected_color * dot
				color_emissive = reflectance_ratio * projected_color
#				color_specular = reflectance_ratio * projected_color * (F.relu((view_dirs * reflection_vector).sum(dim=-1)) ** roughness)[...,None]
#				projected_colors.append(color_lambert + color_specular)
				projected_colors.append(color_lambert + color_emissive)

			elif self.mode in ["implicit", "implicit_w_blend"]:
				# Input
				#  points
				#  normals
				#  view_dirs
				#  feature_vectors
				#  projected_color
				#  input_dirs
				points_ = points_timed.view(-1,self.pos_dim)
				input_dirs_ = input_dirs.view(-1,3)
				if self.embedview_fn is not None:
					view_dirs_ = self.embedview_fn(view_dirs).float()
				else:
					view_dirs_ = view_dirs

				if shadow_maps is None:
					shadow_map = torch.ones(points_.shape[0], 1).to(points_)
				else:
					shadow_map = shadow_maps[pi][...,None]

				x = torch.cat([points_, view_dirs_, normals, feature_vectors, projected_color * (0 if wo_pattern else 1), input_dirs_, shadow_map], dim=-1)	
				x = self.implicit_projection_network(x)
				x = torch.sigmoid(x)

				projected_colors.append(x)

			else:
				raise NotImplementedError()

		projected_colors = torch.stack(projected_colors).sum(dim=0)
		if self.mode == "explicit_w_ambient":
			projected_colors += self.ambient_bias

		if self.grayscale:
			projected_colors = projected_colors.mean(dim=-1, keepdim=True)

		if self.mode in ["original", "explicit", "explicit_w_ambient", "implicit_w_blend"]:
			return self.blend(points_timed, normals, view_dirs, feature_vectors, illum_params, color_network, projected_colors, wo_pattern)
		elif self.mode in ["implicit", "implicit_wo_feature", "implicit_only_feature"]:
			return projected_colors
		else:
			raise NotImplementedError()


	def blend(self, points, normals, view_dirs, feature_vectors, illum_params, color_network, projected_colors, wo_pattern=False):
		sampled_color = color_network(points.reshape(-1,self.pos_dim), normals, view_dirs, feature_vectors)
		if sampled_color.shape[-1] == 1:
			projected_colors = projected_colors.mean(dim=-1, keepdims=True)
		if self.blend_dropout_ratio > 0 and np.random.rand() < self.blend_dropout_ratio:
			# Technique to avoid no grad problem
			return sampled_color * illum_params["ambient"] + (sampled_color * illum_params["diffuse"] * 0 + illum_params["emissive"] * 0) * projected_colors
		if not wo_pattern:
			sampled_color = sampled_color * illum_params["ambient"] + \
			(sampled_color * illum_params["diffuse"] + illum_params["emissive"]) * projected_colors
		return sampled_color


	def blend_nerf(self, points, indices, proj_params, illum_params, sampled_color, scatter_density=None, scatter_color=None):
		projected_colors = []
		pts_homo = torch.cat([points, torch.ones_like(points[...,:1]).to(points)], dim=-1)

		for pi, proj_param in enumerate(proj_params):
			_, projected_color, _ = self.prepare_proj_params(pts_homo, proj_param, indices, None, scatter_density, scatter_color)
			projected_colors.append(projected_color)

		projected_colors = torch.stack(projected_colors).sum(dim=0)

		if sampled_color.shape[-1] == 1:
			projected_colors = projected_colors.mean(dim=-1, keepdims=True)

		sampled_color = sampled_color * illum_params["ambient"] * self.nerf_illum_param[0] + \
			(sampled_color * illum_params["diffuse"] * self.nerf_illum_param[1] + illum_params["emissive"] + self.nerf_illum_param[2]) * projected_colors
		return sampled_color


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
	def __init__(self,
				 D=8,
				 W=256,
				 d_in=3,
				 d_in_view=3,
				 multires=0,
				 multires_view=0,
				 encoding="default",
				 encoding_view="default",
				 output_ch=4,
				 skips=[4],
				 use_viewdirs=False,
				 time_dim=0):
		super(NeRF, self).__init__()
		self.D = D
		self.W = W
		self.d_in = d_in
		self.d_in_view = d_in_view
		self.input_ch = 3
		self.input_ch_view = 3
		self.embed_fn = None
		self.embed_fn_view = None
		self.d_in += time_dim

		if multires > 0:
			embed_fn, input_ch = get_embedder(encoding, multires, input_dims=d_in)
			self.embed_fn = embed_fn
			self.input_ch = input_ch

		if multires_view > 0:
			embed_fn_view, input_ch_view = get_embedder(encoding_view, multires_view, input_dims=d_in_view)
			self.embed_fn_view = embed_fn_view
			self.input_ch_view = input_ch_view

		self.skips = skips
		self.use_viewdirs = use_viewdirs

		self.pts_linears = nn.ModuleList(
			[nn.Linear(self.input_ch, W)] +
			[nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

		### Implementation according to the official code release
		### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
		self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

		### Implementation according to the paper
		# self.views_linears = nn.ModuleList(
		#     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

		if use_viewdirs:
			self.feature_linear = nn.Linear(W, W)
			self.alpha_linear = nn.Linear(W, 1)
			self.rgb_linear = nn.Linear(W // 2, output_ch - 1)
		else:
			self.output_linear = nn.Linear(W, output_ch)

	def forward(self, input_pts, input_views):
		if self.embed_fn is not None:
			input_pts = self.embed_fn(input_pts).float()
		if self.embed_fn_view is not None:
			input_views = self.embed_fn_view(input_views).float()

		h = input_pts
		for i, l in enumerate(self.pts_linears):
			h = self.pts_linears[i](h)
			h = F.relu(h)
			if i in self.skips:
				h = torch.cat([input_pts, h], -1)

		if self.use_viewdirs:
			alpha = self.alpha_linear(h)
			feature = self.feature_linear(h)
			h = torch.cat([feature, input_views], -1)

			for i, l in enumerate(self.views_linears):
				h = self.views_linears[i](h)
				h = F.relu(h)

			rgb = self.rgb_linear(h)
			return alpha, rgb
		else:
			assert False


class SingleVarianceNetwork(nn.Module):
	def __init__(self, init_val):
		super(SingleVarianceNetwork, self).__init__()
		self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

	def forward(self, x):
		return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)



class TimeEmbedding(nn.Module):
	def __init__(self, pos_dim, time_dim, max_time, 
		static_dim=128, dynamic_dim=64, num_mlps=[5, 20]):
		super().__init__()
		self.pos_dim = pos_dim
		self.time_dim = time_dim
		self.max_time = max_time
		self.static_dim = static_dim
		self.dynamic_dim = dynamic_dim
		self.total_dim = self.static_dim + self.dynamic_dim * len(num_mlps)
		self.num_mlps = num_mlps

		self.static_mlp = self._make_mlp(self.pos_dim, self.static_dim, 3)

		dynamic_mlps = {}
		for l in range(len(self.num_mlps)):
			for i in range(self.num_mlps[l]):
				dynamic_mlps["%d_%d" % (l, i)] = self._make_mlp(self.time_dim, self.dynamic_dim, 3)
		self.dynamic_mlps = nn.ModuleDict(dynamic_mlps)


	def _make_mlp(self, in_dim, out_dim, n_layers):
		dims = [in_dim] + [out_dim] * (n_layers - 1)
		layers = []
		for i in range(len(dims) - 1):
			lin = nn.Linear(dims[i], dims[i+1])
			torch.nn.init.constant_(lin.bias, 0.0)
			torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
			lin = nn.utils.weight_norm(lin)
			layers.append(lin)
			layers.append(nn.Softplus(beta=100))

		return nn.Sequential(*layers[:-1])


	def forward(self, x_static, x_dynamic, t : int, is_reference_frame=False):
		t_ = t / self.max_time

		feature_list = [self.static_mlp(x_static)]

		input_to_dynamic = x_dynamic

		for l in range(len(self.num_mlps)):
			mlp_index = t_ * (self.num_mlps[l] - 1)
			alpha = mlp_index - int(mlp_index)
			mlp_index = int(mlp_index)

			if self.num_mlps[l] >= 2:
				v_d1 = self.dynamic_mlps["%d_%d" % (l, mlp_index)](input_to_dynamic)
				v_d2 = self.dynamic_mlps["%d_%d" % (l, mlp_index + 1)](input_to_dynamic)
				v_d = v_d1 * (1 - alpha) + v_d2 * alpha
			else:
				v_d = self.dynamic_mlps["%d_%d" % (l, mlp_index)](input_to_dynamic)

			if is_reference_frame:
				v_d = v_d.detach()

			feature_list.append(v_d)

		return torch.cat(feature_list, dim=-1)



class TimeEmbeddingGrid(nn.Module):
	def __init__(self, pos_dim, time_dim, max_time, num_mlps=None):
		super().__init__()
		self.pos_dim = pos_dim
		self.time_dim = time_dim
		self.max_time = max_time
		self.total_dim = self.pos_dim + self.time_dim


	def forward(self, x_static, x_dynamic, t : int, is_reference_frame=False):
		return torch.cat([x_static, x_dynamic], dim=-1)



class TimeEmbeddingCascade(nn.Module):
	def __init__(self, pos_dim, time_dim, max_time, latent_dim=128, num_mlps=[2, 8]):
		super().__init__()
		self.pos_dim = pos_dim
		self.time_dim = time_dim
		self.max_time = max_time
		self.latent_dim = latent_dim
		self.total_dim = self.latent_dim * len(num_mlps)
		self.num_mlps = num_mlps

		self.static_mlp = self._make_mlp(self.pos_dim, self.total_dim, 3)
		dynamic_mlps = {}
		for l in range(len(self.num_mlps)):
			for i in range(self.num_mlps[l]):
				dynamic_mlps["%d_%d" % (l, i)] = self._make_mlp(self.time_dim + self.total_dim, self.latent_dim, 3)
		self.dynamic_mlps = nn.ModuleDict(dynamic_mlps)


	def _make_mlp(self, in_dim, out_dim, n_layers):
		dims = [in_dim] + [out_dim] * (n_layers - 1)
		layers = []
		for i in range(len(dims) - 1):
			lin = nn.Linear(dims[i], dims[i+1])
			torch.nn.init.constant_(lin.bias, 0.0)
			torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
			lin = nn.utils.weight_norm(lin)
			layers.append(lin)
			layers.append(nn.Softplus(beta=100))

		return nn.Sequential(*layers[:-1])


	def forward(self, x_static, x_dynamic, t : int, is_reference_frame=False):
		t_ = t / self.max_time

		x_static = self.static_mlp(x_static)

		if is_reference_frame:
			return x_static

		input_to_dynamic = torch.cat([x_static, x_dynamic], dim=-1)

		feature_list = []
		for l in range(len(self.num_mlps)):
			mlp_index = t_ * (self.num_mlps[l] - 1)
			alpha = mlp_index - int(mlp_index)
			mlp_index = int(mlp_index)

			if self.num_mlps[l] >= 2:
				v_d1 = self.dynamic_mlps["%d_%d" % (l, mlp_index)](input_to_dynamic)
				v_d2 = self.dynamic_mlps["%d_%d" % (l, mlp_index + 1)](input_to_dynamic)
				v_d = v_d1 * (1 - alpha) + v_d2 * alpha
			else:
				v_d = self.dynamic_mlps["%d_%d" % (l, mlp_index)](input_to_dynamic)

			feature_list.append(v_d)

		return torch.cat(feature_list, dim=-1)



class MediumNeRF(nn.Module):
	def __init__(self,
				 D=8,
				 W=256,
				 d_in=3,
				 d_in_view=3,
				 multires=0,
				 multires_view=0,
				 encoding="default",
				 encoding_view="default",
				 output_ch=4,
				 skips=[4],
				 use_viewdirs=False):
		super(MediumNeRF, self).__init__()
		self.D = D
		self.W = W
		self.d_in = d_in
		self.d_in_view = d_in_view
		self.input_ch = 3
		self.input_ch_view = 3
		self.embed_fn = None
		self.embed_fn_view = None

		if multires > 0:
			embed_fn, input_ch = get_embedder(encoding, multires, input_dims=d_in)
			self.embed_fn = embed_fn
			self.input_ch = input_ch

		if multires_view > 0:
			embed_fn_view, input_ch_view = get_embedder(encoding_view, multires_view, input_dims=d_in_view)
			self.embed_fn_view = embed_fn_view
			self.input_ch_view = input_ch_view

		self.skips = skips
		self.use_viewdirs = use_viewdirs

		self.pts_linears = nn.ModuleList(
			[nn.Linear(self.input_ch, W)] +
			[nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

		### Implementation according to the official code release
		### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
		self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])
#		self.views_linears = nn.ModuleList([nn.Linear(W, W // 2)])

		### Implementation according to the paper
		# self.views_linears = nn.ModuleList(
		#     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

		if use_viewdirs:
			self.feature_linear = nn.Linear(W, W)
			self.alpha_linear = nn.Parameter(torch.tensor(1e-4))
#			self.rgb_linear = nn.Linear(W // 2, 3)
			self.rgb_linear = nn.Parameter(torch.from_numpy(np.float32([0.1, 0.1, 0.1])))
		else:
			self.output_linear = nn.Linear(W, output_ch)

	def forward(self, input_pts, input_views):
#		if self.embed_fn is not None:
#			input_pts = self.embed_fn(input_pts).float()
#		if self.embed_fn_view is not None:
#			input_views = self.embed_fn_view(input_views).float()

#		h = input_pts
#		for i, l in enumerate(self.pts_linears):
#			h = self.pts_linears[i](h)
#			h = F.relu(h)
#			if i in self.skips:
#				h = torch.cat([input_pts, h], -1)

		if self.use_viewdirs:
			alpha = self.alpha_linear.view(1,1).expand(input_pts.shape[0],1).clamp(0, 1e-2)
#			feature = self.feature_linear(h)
#			h = torch.cat([feature, input_views], -1)
#			h = feature

#			for i, l in enumerate(self.views_linears):
#				h = self.views_linears[i](h)
#				h = F.relu(h)

#			rgb = self.rgb_linear(h)
			rgb = self.rgb_linear.view(1,3).expand(input_pts.shape[0],3)
			return alpha, rgb
		else:
			assert False



class ShadowField(nn.Module):
	def __init__(self,
				 D=8,
				 W=256,
				 d_in=3,
				 d_in_view=3,
				 multires=0,
				 multires_view=0,
				 encoding="default",
				 encoding_view="default",
				 skips=[4]):
		super(ShadowField, self).__init__()
		self.D = D
		self.W = W
		self.d_in = d_in
		self.d_in_view = d_in_view
		self.input_ch = 3
		self.input_ch_view = 3
		self.embed_fn = None
		self.embed_fn_view = None

		if multires > 0:
			embed_fn, input_ch = get_embedder(encoding, multires, input_dims=d_in)
			self.embed_fn = embed_fn
			self.input_ch = input_ch

		if multires_view > 0:
			embed_fn_view, input_ch_view = get_embedder(encoding_view, multires_view, input_dims=d_in_view)
			self.embed_fn_view = embed_fn_view
			self.input_ch_view = input_ch_view

		self.skips = skips

		layers = [nn.Linear(self.input_ch + self.input_ch_view + 1, W)]
		for i in range(D - 1):
			if i not in self.skips:
				layers.append(nn.Linear(W, W))
			else:
				layers.append(nn.Linear(W+self.input_ch+self.input_ch_view, W))
		layers += [nn.Linear(W, 1)]
		self.layers = nn.ModuleList(layers)


	def forward(self, input_pts, input_views, sdf):
		if self.embed_fn is not None:
			input_pts = self.embed_fn(input_pts).float()
		if self.embed_fn_view is not None:
			input_views = self.embed_fn_view(input_views).float()

		h = torch.cat([input_pts, input_views, sdf], dim=-1)
		for i, l in enumerate(self.layers):
			h = self.layers[i](h)
			if i < len(self.layers) - 1:
				h = F.relu(h)
			if i in self.skips:
				h = torch.cat([input_pts, input_views, h], -1)

		return torch.sigmoid(h)



class DeformNetwork(nn.Module):
	def __init__(self,
				 d_hidden,
				 n_layers,
				 skip_in=(4,),
				 multires=0,
				 encoding="default",
				 bias=0.0,
		):
		super().__init__()

		transform_dim = 3
		dims = [4] + [d_hidden for _ in range(n_layers)] + [transform_dim]

		self.embed_fn_fine = None

		if multires > 0:
			embed_fn, input_ch = get_embedder(encoding, multires, input_dims=4)
			self.embed_fn_fine = embed_fn
			dims[0] = input_ch

		self.num_layers = len(dims)
		self.skip_in = skip_in

		for l in range(0, self.num_layers - 1):
			if l + 1 in self.skip_in:
				out_dim = dims[l + 1] - dims[0]
			else:
				out_dim = dims[l + 1]

			lin = nn.Linear(dims[l], out_dim)
			torch.nn.init.constant_(lin.bias, bias)
			torch.nn.init.normal_(lin.weight, 0.0, std=1e-3)

			setattr(self, "lin" + str(l), lin)

		self.activation = nn.Softplus(beta=100)



	def forward(self, inputs):
		if self.embed_fn_fine is not None:
			inputs_ = self.embed_fn_fine(inputs).float()
		else:
			inputs_ = inputs

		x = inputs_
		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			if l in self.skip_in:
				x = torch.cat([x, inputs_], -1) / np.sqrt(2)

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.activation(x)

		return inputs[...,:3] + x * (inputs[...,3:] > 0).float()



class DeformNetworkInterp(nn.Module):
	def __init__(self,
				 d_hidden,
				 n_layers,
				 skip_in=(4,),
				 multires=0,
				 encoding="default",
				 bias=0.0,
		):
		super().__init__()

		self.transform_dim = 3
		self.degree = 4
		dims = [4] + [d_hidden for _ in range(n_layers)] + [self.transform_dim * self.degree]

		self.embed_fn_fine = None

		if multires > 0:
			embed_fn, input_ch = get_embedder(encoding, multires, input_dims=4)
			self.embed_fn_fine = embed_fn
			dims[0] = input_ch

		self.num_layers = len(dims)
		self.skip_in = skip_in

		for l in range(0, self.num_layers - 1):
			if l + 1 in self.skip_in:
				out_dim = dims[l + 1] - dims[0]
			else:
				out_dim = dims[l + 1]

			lin = nn.Linear(dims[l], out_dim)
			torch.nn.init.constant_(lin.bias, bias)
			torch.nn.init.normal_(lin.weight, 0.0, std=1e-3)

			setattr(self, "lin" + str(l), lin)

		self.activation = nn.Softplus(beta=100)



	def forward(self, inputs):
		if self.embed_fn_fine is not None:
			inputs_ = self.embed_fn_fine(inputs).float()
		else:
			inputs_ = inputs

		x = inputs_
		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			if l in self.skip_in:
				x = torch.cat([x, inputs_], -1) / np.sqrt(2)

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.activation(x)

		coefs = x.view(-1, self.degree, self.transform_dim)
		t = inputs[...,3:]
		delta = (torch.stack([t ** 3, t ** 2, t, torch.ones_like(t)], dim=-2) * coefs).sum(dim=-2)

		return inputs[...,:3] + delta * (inputs[...,3:] > 0).float()



class DeformNetworkRigid(nn.Module):
	def __init__(self,
				 d_hidden,
				 n_layers,
				 skip_in=(4,),
				 multires=0,
				 encoding="default",
				 bias=0.0,
		):
		super().__init__()

		transform_dim = 7	# Quaternion + translation
		dims = [3] + [d_hidden for _ in range(n_layers)] + [transform_dim]

		self.embed_fn_fine = None

		if multires > 0:
			embed_fn, input_ch = get_embedder(encoding, multires, input_dims=3)
			self.embed_fn_fine = embed_fn
			dims[0] = input_ch

		self.num_layers = len(dims)
		self.skip_in = skip_in

		for l in range(0, self.num_layers - 1):
			if l + 1 in self.skip_in:
				out_dim = dims[l + 1] - dims[0]
			else:
				out_dim = dims[l + 1]

			lin = nn.Linear(dims[l], out_dim)
			setattr(self, "lin" + str(l), lin)

		self.activation = nn.Softplus(beta=100)


	def forward(self, inputs):
		if self.embed_fn_fine is not None:
			inputs_ = self.embed_fn_fine(inputs[...,:3]).float()
		else:
			inputs_ = inputs[...,:3]

		x = inputs_
		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			if l in self.skip_in:
				x = torch.cat([x, inputs_], -1) / np.sqrt(2)

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.activation(x)

		time = inputs[...,3:]
		transforms = x
		quat = transforms[...,:4]
		quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
		identity_quat = torch.zeros_like(quat)
		identity_quat[...,0] = 1
		R = camera.Quaternion().q_to_R(camera.Quaternion().interpolate(identity_quat, quat, time))
		T = transforms[...,4:] * time

		transformed = torch.matmul(R, inputs[:,:3,None])[...,0] + T
		return transformed

