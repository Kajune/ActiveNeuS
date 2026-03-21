import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic


def extract_fields(bound_min, bound_max, resolution, query_func, flat_axis):
	N = 1024

	xyz = []
	for d in range(3):
		v = torch.linspace(bound_min[d], bound_max[d], resolution)
		if d != flat_axis:
			v = v.split(N)
		xyz.append(v)

	X, Y, Z = xyz

	u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
	with torch.no_grad():
		for xi, xs in enumerate(X):
			for yi, ys in enumerate(Y):
				for zi, zs in enumerate(Z):
					if flat_axis == 2:
						xx, yy = torch.meshgrid(xs, ys)
						pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), torch.ones_like(xx.reshape(-1, 1)) * zs], dim=-1)
						val = query_func(pts).reshape(len(xs), len(ys)).detach().cpu().numpy()
						u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi] = val
					elif flat_axis == 1:
						xx, zz = torch.meshgrid(xs, zs)
						pts = torch.cat([xx.reshape(-1, 1), torch.ones_like(xx.reshape(-1, 1)) * ys, zz.reshape(-1, 1)], dim=-1)
						val = query_func(pts).reshape(len(xs), len(zs)).detach().cpu().numpy()
						u[xi * N: xi * N + len(xs), yi, zi * N: zi * N + len(zs)] = val
					elif flat_axis == 0:
						yy, zz = torch.meshgrid(ys, zs)
						pts = torch.cat([torch.ones_like(yy.reshape(-1, 1)) * xs, yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
						val = query_func(pts).reshape(len(ys), len(zs)).detach().cpu().numpy()
						u[xi, yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

	return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, flat_axis):
	print('threshold: {}'.format(threshold))
	u = extract_fields(bound_min, bound_max, resolution, query_func, flat_axis)
	vertices, triangles = mcubes.marching_cubes(u, threshold)
	b_max_np = bound_max.detach().cpu().numpy()
	b_min_np = bound_min.detach().cpu().numpy()

	vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
	return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
	# This implementation is from NeRF
	# Get pdf
	weights = weights + 1e-5  # prevent nans
	pdf = weights / torch.sum(weights, -1, keepdim=True)
	cdf = torch.cumsum(pdf, -1)
	cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
	# Take uniform samples
	if det:
		u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
		u = u.expand(list(cdf.shape[:-1]) + [n_samples])
	else:
		u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

	# Invert CDF
	u = u.contiguous()
	inds = torch.searchsorted(cdf, u, right=True)
	below = torch.max(torch.zeros_like(inds - 1), inds - 1)
	above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
	inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

	matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
	cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
	bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

	denom = (cdf_g[..., 1] - cdf_g[..., 0])
	denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
	t = (u - cdf_g[..., 0]) / denom
	samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

	return samples


def unimodality_loss(alpha, transmittance, eps=1e-6):
	N, D = alpha.shape
	tau = transmittance

	return ((1 - (tau / tau.max()).detach() - 0.5).clip(0, 1) * alpha).mean(dim=-1)

#	domain = torch.linspace(0, 1, D)[None,...].expand_as(alpha)
#	alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + eps)
#	mu = (domain * alpha).sum(dim=-1, keepdim=True)
#	return (-(alpha[:,:-1] - alpha[:,1:]) * torch.sign(domain[:,:-1] - mu)).clip(min=0.0).mean(dim=-1)

#	sigma = torch.sqrt((alpha * (domain - mu) ** 2).sum(dim=-1, keepdim=True))
#	pdf = (1 / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((domain - mu) / sigma) ** 2)
#	pdf = pdf / (pdf.sum(dim=-1, keepdim=True) + eps)
#	return ((alpha - pdf) ** 2).sum()

#	alpha_norm = alpha / alpha_sum
#	pdf_norm = pdf / (pdf.sum(dim=-1, keepdim=True) + eps)

#	kl_div1 = (alpha_norm * (torch.log(alpha_norm + eps) - torch.log(pdf_norm + eps)))
#	kl_div2 = (pdf_norm * (torch.log(pdf_norm + eps) - torch.log(alpha_norm + eps)))
#	kl_div = (kl_div1 + kl_div2) / 2
#	return kl_div.sum()


def compute_weights(alpha):
	transmittance = torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
	weights = alpha * transmittance
	return weights, transmittance


@torch.no_grad()
def intersect_sphere(ray_o, ray_d, r, keepdim=False):
	"""
	ray_o, ray_d: [..., 3]
	compute the depth of the intersection point between this ray and unit sphere
	"""
	# note: d1 becomes negative if this mid point is behind camera
	d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
	p = ray_o + d1.unsqueeze(-1) * ray_d

	tmp = r * r - torch.sum(p * p, dim=-1)
	mask_intersect = tmp > 0.0
	d2 = torch.sqrt(torch.clamp(tmp, min=0.0)) / torch.norm(ray_d, dim=-1)

	near = d1 - d2
	far = d1 + d2
	if keepdim:
		near = near.unsqueeze(-1)
		far = far.unsqueeze(-1)
	return mask_intersect, near, far



class NeuSRenderer:
	def __init__(self,
				 n_images,
				 nerf,
				 sdf_network,
				 deviation_network,
				 color_network,
				 projection_network,
				 shadow_field,
				 deform_network,
				 n_samples,
				 n_importance,
				 n_outside,
				 up_sample_steps,
				 perturb,
				 enable_shadow=False,
				 shadow_alpha_thresh=0.0,
				 shadow_sdf_thresh=0.0,
				 allow_volume_scattering=False,
				 allow_subsurface_scattering=False,
				 constant_bg_density=None,
				 constant_bg_color=None,
				 visualize_shadow=False,
				 wo_pattern=False,
				 shadow_volume_mode="weights",
				 density_mode="sdf",
				 sdf_type="neus",
				 outside_dropout_ratio=0.0,
				 input_time=False,
				 time_gradient_type=None,
				 curvature_cap=None):
		self.n_images = n_images
		self.nerf = nerf
		self.sdf_network = sdf_network
		self.deviation_network = deviation_network
		self.color_network = color_network
		self.projection_network = projection_network
		self.shadow_field = shadow_field
		self.deform_network = deform_network
		self.n_samples = n_samples
		self.n_importance = n_importance
		self.n_outside = n_outside
		self.up_sample_steps = up_sample_steps
		self.perturb = perturb
		self.enable_shadow = enable_shadow
		self.shadow_alpha_thresh = shadow_alpha_thresh
		self.shadow_sdf_thresh = shadow_sdf_thresh
		self.allow_volume_scattering = allow_volume_scattering
		self.allow_subsurface_scattering = allow_subsurface_scattering
		self.constant_bg_density = constant_bg_density
		self.constant_bg_color = constant_bg_color
		self.visualize_shadow = visualize_shadow
		self.wo_pattern = wo_pattern
		self.shadow_volume_mode = shadow_volume_mode
		self.density_mode = density_mode
		self.hybrid_rate = 0.1
		self.nerf_coef = 1.0
		self.sdf_type = sdf_type
		self.outside_dropout_ratio = outside_dropout_ratio
		self.input_time = input_time
		self.pos_dim = 3
		if self.input_time:
			self.pos_dim += 1
		self.time_gradient_type = time_gradient_type
		self.curvature_cap = curvature_cap


	def to_timed_pts(self, pts, indices):
		if self.input_time or self.deform_network is not None:
			if isinstance(indices, int):
				indices = torch.tensor(indices).to(pts).long()
			if isinstance(indices, float):
				indices = torch.tensor(indices).to(pts).float()
			if indices.dim() == 0:
				indices = indices.view(1,1)
			if pts.dim() == 2:
				indices_ = indices.expand(pts.shape[0], 1)
			elif pts.dim() == 3:
				indices_ = indices[:,None,:].expand(pts.shape[0], pts.shape[1], 1)

			time_encoding = indices_ / self.n_images
			pts_ = torch.cat([pts, time_encoding], dim=-1)

			if self.deform_network is not None:
				return self.deform_network(pts_.view(-1,4)).view(*pts.shape)
			else:
				return pts_
		else:
			return pts


	def render_core_outside(self, rays_o, rays_d, indices, z_vals, sample_dist, nerf, background_rgb=None,
							proj_params=None, illum_params=None):
		"""
		Render background
		"""
		batch_size, n_samples = z_vals.shape

		# Section length
		dists = z_vals[..., 1:] - z_vals[..., :-1]
		dists = torch.cat([dists, (z_vals[...,1] - z_vals[...,0])[...,None]], -1)
		mid_z_vals = z_vals + dists * 0.5

		# Section midpoints
		pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

		dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
		pts_ = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

		dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)
		dirs = dirs * z_vals[...,np.newaxis]

		pts_ = self.to_timed_pts(pts_, indices)
		pts_ = pts_.reshape(-1, self.pos_dim + 1)
		dirs = dirs.reshape(-1, 3)
		dirs = F.normalize(dirs, dim=-1)

		density, sampled_color = nerf(pts_, dirs)
		sampled_color = torch.sigmoid(sampled_color)

		if self.constant_bg_density is not None:
			density = torch.ones_like(density) * self.constant_bg_density

		if self.constant_bg_color is not None:
			sampled_color = torch.ones_like(sampled_color) * torch.from_numpy(np.float32(self.constant_bg_color)).to(sampled_color)

		alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
		alpha = alpha.reshape(batch_size, n_samples)
		weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

		density = density.reshape(batch_size, n_samples)
		if proj_params is not None and len(proj_params) > 0:
			sampled_color = self.projection_network.blend_nerf(pts, indices, proj_params, illum_params, sampled_color, 
#				density, sampled_color.reshape(batch_size, n_samples, -1)
			).view(*sampled_color.shape)

		sampled_color = sampled_color.reshape(batch_size, n_samples, -1)

		color = (weights[:, :, None] * sampled_color).sum(dim=1)
		if background_rgb is not None:
			color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

		return {
			'color': color,
			'sampled_color': sampled_color,
			'alpha': alpha,
			'weights': weights,
			'density': density,
		}

	def up_sample(self, rays_o, rays_d, indices, z_vals, sdf, n_importance, inv_s):
		"""
		Up sampling give a fixed inv_s
		"""
		batch_size, n_samples = z_vals.shape
		pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
		radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
		inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
		sdf = sdf.reshape(batch_size, n_samples)
		prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
		dist = (next_z_vals - prev_z_vals)

		if self.density_mode in ["nerf", "hybrid"]:
			alpha = 1.0 - torch.exp(-F.softplus(-sdf[:, :-1] * self.nerf_coef) * dist)
	
		if self.density_mode in ["sdf", "hybrid"]:
			if self.sdf_type == "neus":
				prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
				mid_sdf = (prev_sdf + next_sdf) * 0.5
				cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

				# ----------------------------------------------------------------------------------------------------------
				# Use min value of [ cos, prev_cos ]
				# Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
				# robust when meeting situations like below:
				#
				# SDF
				# ^
				# |\          -----x----...
				# | \        /
				# |  x      x
				# |---\----/-------------> 0 level
				# |    \  /
				# |     \/
				# |
				# ----------------------------------------------------------------------------------------------------------
				prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
				cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
				cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
				cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

				prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
				next_esti_sdf = mid_sdf + cos_val * dist * 0.5
				prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
				next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
				alpha_ = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

			elif self.sdf_type == "stylesdf":
				sigma = torch.sigmoid(-sdf[:, :-1] / (inv_s * 1e-2)) / (inv_s * 1e-2)
				alpha_ = 1 - torch.exp(-sigma * dist)

			if self.density_mode == "hybrid":
				alpha = alpha * (1 - self.hybrid_rate) + alpha_ * self.hybrid_rate
			else:
				alpha = alpha_

		weights = alpha * torch.cumprod(
			torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

		z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
		return z_samples


	def cat_z_vals(self, rays_o, rays_d, indices, z_vals, new_z_vals, sdf, last=False):
		batch_size, n_samples = z_vals.shape
		_, n_importance = new_z_vals.shape
		pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
		z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
		z_vals, index = torch.sort(z_vals, dim=-1)

		if not last:
			pts = self.to_timed_pts(pts, indices)
			new_sdf = self.sdf_network.sdf(pts.reshape(-1, self.pos_dim), indices).reshape(batch_size, n_importance)
			sdf = torch.cat([sdf, new_sdf], dim=-1)
			xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
			index = index.reshape(-1)
			sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

		return z_vals, sdf


	def importance_sampling(self, rays_o, rays_d, indices, z_vals, n_importance, up_sample_steps):
		batch_size, n_samples = z_vals.shape

		with torch.no_grad():
			pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
			pts = self.to_timed_pts(pts, indices)
			sdf = self.sdf_network.sdf(pts.reshape(-1, self.pos_dim), indices).reshape(pts.shape[:-1])

			for i in range(up_sample_steps):
				new_z_vals = self.up_sample(rays_o,	rays_d, indices, z_vals, sdf, n_importance // up_sample_steps, 64 * 2**i)
				z_vals, sdf = self.cat_z_vals(rays_o, rays_d, indices, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

		n_samples = self.n_samples + n_importance

		return z_vals, new_z_vals, sdf, n_samples


	def ray_marching(self,
					rays_o,
					rays_d,
					indices,
					z_vals,
					sdf_network,
					deviation_network,
					cos_anneal_ratio=0.0,
					sharpness_coeff=10,
					compute_curvature_loss=False,
					is_reference_frame=False):
			batch_size, n_samples = z_vals.shape

			# Section length
			dists = z_vals[..., 1:] - z_vals[..., :-1]
			dists = torch.cat([dists, (z_vals[...,1] - z_vals[...,0])[...,None]], -1)
			mid_z_vals = z_vals + dists * 0.5

			# Section midpoints
			pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
			dirs = rays_d[:, None, :].expand(pts.shape)
			dirs = dirs * z_vals[...,np.newaxis]

			pts = self.to_timed_pts(pts, indices)
			pts = pts.reshape(-1, self.pos_dim)
			dirs = dirs.reshape(-1, 3)
			dirs = F.normalize(dirs, dim=-1)

			sdf_nn_output = sdf_network(pts, indices, is_reference_frame=is_reference_frame)
			sdf = sdf_nn_output[:, :1]
			feature_vector = sdf_nn_output[:, 1:]

			inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
			inv_s = inv_s.expand(batch_size * n_samples, 1)

			if self.density_mode in ["nerf", "hybrid"]:
				alpha = 1.0 - torch.exp(-F.softplus(-sdf * self.nerf_coef) * dists.reshape(-1, 1)).reshape(batch_size, n_samples)
				gradients_all = sdf_network.gradient(pts, indices, compute_hessian=False)
				gradients = gradients_all.squeeze()[...,:3]
				hessians_all = None
				hessians = None

			if self.density_mode in ["sdf", "hybrid"]:
				if compute_curvature_loss:
					gradients_all, hessians_all = sdf_network.gradient(pts, indices, compute_hessian=True)
					gradients = gradients_all.squeeze()[...,:3]
					hessians = hessians_all.squeeze()[...,:3]
				else:
					gradients_all = sdf_network.gradient(pts, indices, compute_hessian=False)
					gradients = gradients_all.squeeze()[...,:3]
					hessians_all = None
					hessians = None

				if self.sdf_type == "neus":
					# From https://github.com/mabaorui/TowardsBetterGradient
			#		gradient_norm = F.normalize(gradients, dim=-1)
			#		pts_moved = pts + gradient_norm * sdf
			#		sdf_moved = sdf_network(pts_moved)[:, :1]
			#		gradient_moved = sdf_network.gradient(pts_moved).squeeze()
			#		gradient_moved_norm = F.normalize(gradient_moved, dim=-1)
			#		consis_constraint = 1 - F.cosine_similarity(gradient_moved_norm, gradient_norm, dim=-1)
			#		weight_moved = torch.exp(-sharpness_coeff * torch.abs(sdf)).reshape(-1,consis_constraint.shape[-1]) 
			#		consistency_error = (consis_constraint * weight_moved).sum()

					true_cos = (dirs * gradients).sum(-1, keepdim=True)

					# "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
					# the cos value "not dead" at the beginning training iterations, for better convergence.
					iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
								 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

					# Estimate signed distances at section points
					estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
					estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

					prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
					next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

					p = prev_cdf - next_cdf
					c = prev_cdf

					alpha_ = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

				elif self.sdf_type == "stylesdf":
					sigma = torch.sigmoid(-sdf / (inv_s * 1e-2)) / (inv_s * 1e-2)
					alpha_ = 1 - torch.exp(-sigma * dists.reshape(-1, 1)).reshape(batch_size, n_samples)

				if self.density_mode == "hybrid":
#					self.hybrid_rate = (alpha_ / (alpha + 1e-6)).clamp(0, 1).mean()
					alpha = alpha * (1 - self.hybrid_rate) + alpha_ * self.hybrid_rate
				else:
					alpha = alpha_

			if gradients_all.shape[-1] == 4:
				gradients_time = gradients_all[...,3:]
			else:
				gradients_time = None

			if hessians_all is not None and hessians_all.shape[-1] == 4:
				hessians_time = hessians_all[...,3:]
			else:
				hessians_time = None

			return pts[...,:3], dirs, dists, sdf, feature_vector, alpha, gradients, gradients_time, hessians, hessians_time, inv_s


	def render_core(self,
					rays_o,
					rays_d,
					indices,
					z_vals,
					sample_dist,
					sdf_network,
					deviation_network,
					color_network,
					background_alpha=None,
					background_sampled_color=None,
					background_density=None,
					background_rgb=None,
					cos_anneal_ratio=0.0,
					sharpness_coeff=10,
					proj_params=[],
					illum_params=None,
					compute_curvature_loss=False,
					compute_temporal_smoothness_error=False,
					shadow_field_ratio=False,
					disable_shadow=False,
					visualize_shadow=False,
					is_reference_frame=False):

		batch_size, n_samples = z_vals.shape
		pts, dirs, dists, sdf, feature_vector, foreground_alpha, gradients, gradients_time, hessians, hessians_time, inv_s = \
			self.ray_marching(rays_o, rays_d, indices, z_vals, sdf_network, 
								deviation_network, cos_anneal_ratio, sharpness_coeff, compute_curvature_loss, is_reference_frame)
		pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
		inside_sphere = (pts_norm < 1.0).float().detach()
		relax_inside_sphere = (pts_norm < 1.2).float().detach()

		#
		# Shadow mapping
		#
		if self.enable_shadow and not disable_shadow:
			shadow_maps = []

#			with torch.no_grad():
			if self.shadow_field is not None and shadow_field_ratio > 0:
				use_shadow_field = np.random.rand() <= shadow_field_ratio
			else:
				use_shadow_field = False

			if use_shadow_field:
				valid_points = torch.ones_like(foreground_alpha.view(-1).cpu().detach()) > 0
			else:
				valid_points = (foreground_alpha.view(-1).cpu().detach().numpy() > self.shadow_alpha_thresh)

			pts_homo = torch.cat([pts[valid_points], torch.ones_like(pts[valid_points][...,:1]).to(pts)], dim=-1)

			for proj_param in proj_params:
				shadow_mask = torch.ones(pts.shape[:-1]).to(pts)

				if valid_points.any():
					proj_pose = proj_param["proj_pose"].to(pts_homo)
					cam_pose = proj_param["cam_pose"].to(pts_homo)
					scale_mat = proj_param["scale_mat"].to(pts_homo)
					proj_world_mat = torch.matmul(proj_pose, torch.matmul(cam_pose, scale_mat))
					p_rays_o = torch.linalg.inv(proj_world_mat)[...,:3,3].expand(len(pts_homo), -1)
					p_rays_d = F.normalize(pts_homo[...,:3] - p_rays_o, dim=-1)
					distance_to_pts_from_proj = torch.linalg.norm(pts_homo[...,:3] - p_rays_o, dim=-1)

					if not use_shadow_field:
						_, near_z, _ = intersect_sphere(p_rays_o, p_rays_d, 1.0, keepdim=True)
						p_z_vals = torch.linspace(0.0, 1.0, self.n_samples + 1)[None,:].expand(pts_homo.shape[0],-1)
						p_z_vals = p_z_vals * (distance_to_pts_from_proj[...,None] - near_z) + near_z - ((1 / inv_s.mean().detach()) * 2).clip(0.01, 0.1)
#						p_rays_o = pts_homo[...,:3]
#						p_rays_d = -p_rays_d
						p_z_vals, _, p_sdf, _ = self.importance_sampling(p_rays_o, p_rays_d, indices, p_z_vals, self.n_importance, self.up_sample_steps)

						valid_shadow = (p_sdf < self.shadow_sdf_thresh).any(dim=-1)
						if valid_shadow.any():
							mask_tmp = torch.ones_like(shadow_mask[valid_points])
							_, _, _, p_sdf, _, p_alpha, _, _, _, _, _ = self.ray_marching(p_rays_o[valid_shadow], p_rays_d[valid_shadow], indices,
								p_z_vals[valid_shadow], sdf_network, deviation_network, 1.0, sharpness_coeff, compute_curvature_loss=False)
							p_weights, p_transmittance = compute_weights(p_alpha)

							if self.shadow_volume_mode == "weights":
								mask_tmp[valid_shadow] = 1.0 - p_weights[...,:-1].sum(dim=-1).clip(0, 1)
							elif self.shadow_volume_mode == "transmittance":
								mask_tmp[valid_shadow] = p_transmittance[...,-1].clip(0, 1)
							elif self.shadow_volume_mode == "sdf":
								mask_tmp[valid_shadow] = 1.0 - (p_sdf.view(*p_z_vals[valid_shadow].shape) < self.shadow_sdf_thresh).any(dim=-1).float()
							else:
								raise NotImplementedError

							shadow_mask[valid_points] = mask_tmp#.detach()

#						shadow_mask[valid_points] = p_sdf.min(dim=-1)[0]

					else:
						with torch.no_grad():
							shadow_mask[valid_points] = self.shadow_field(pts_homo[...,:3], p_rays_d, sdf[valid_points])[...,0]

				shadow_maps.append(shadow_mask)

		else:
			shadow_maps = None
			shadow_camera = None

		if self.shadow_field is not None:
			shadow_camera = self.shadow_field(pts, dirs, sdf).view(batch_size, n_samples)
		else:
			shadow_camera = None

		if len(proj_params) > 0:
			assert illum_params is not None, "illum_params must be provided to render with projection."

			if background_density is not None and background_sampled_color is not None and self.allow_volume_scattering:
				background_density_ = background_density[:,:n_samples]
				background_sampled_color_ = background_sampled_color[:,:n_samples]
			else:
				background_density_ = None
				background_sampled_color_ = None

			sampled_color = self.projection_network(pts.view(batch_size, n_samples, 3), self.to_timed_pts(pts, indices), indices,
														gradients, dirs, feature_vector, proj_params,
														color_network, illum_params, shadow_maps if not disable_shadow else None, 
														self.wo_pattern, background_density_, background_sampled_color_)
			sampled_color = sampled_color.reshape(batch_size, n_samples, -1)
		else:
			pts = self.to_timed_pts(pts, indices)
			sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, -1)


		# Render with background
		if background_alpha is not None:
			if self.allow_volume_scattering or self.allow_subsurface_scattering:
				outside_mask = torch.ones_like(inside_sphere)
			else:
				outside_mask = 1.0 - inside_sphere

			background_margin = 1 - foreground_alpha * inside_sphere
#			background_margin = foreground_alpha.mean().detach()
			outside_alpha = background_alpha[:, :n_samples].clip(None, background_margin)

			if self.allow_volume_scattering and not self.allow_subsurface_scattering:
				outside_alpha = outside_alpha * (sdf.view(batch_size, n_samples) > 0).float()
			elif not self.allow_volume_scattering and self.allow_subsurface_scattering:
				outside_alpha = outside_alpha * (sdf.view(batch_size, n_samples) < 0).float()

			alpha = foreground_alpha * inside_sphere + outside_alpha * outside_mask
			alpha_solid = foreground_alpha * inside_sphere + outside_alpha * (1.0 - inside_sphere)
			alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
			alpha_solid = torch.cat([alpha_solid, background_alpha[:, n_samples:]], dim=-1)

			sampled_color = sampled_color * inside_sphere[:, :, None] +\
							background_sampled_color[:, :n_samples] * outside_mask[:, :, None]
			sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

		else:
			alpha = foreground_alpha
			alpha_solid = foreground_alpha

		mask_weights, transmittance = compute_weights(alpha)
		color = (sampled_color * mask_weights[:, :, None]).sum(dim=1)

		if self.allow_volume_scattering or self.allow_subsurface_scattering:
			mask_weights_solid, transmittance_solid = compute_weights(alpha_solid)
		else:
			transmittance_solid = transmittance
			mask_weights_solid = mask_weights

		weights_sum = mask_weights.sum(dim=-1, keepdim=True)
		if background_rgb is not None:    # Fixed background, usually black
			color = color + background_rgb * (1.0 - weights_sum)

		visibility = torch.cumprod((sdf.view(batch_size, n_samples) >= 0).float(), dim=-1)

		# Eikonal loss
		if gradients is not None:
			gradients = gradients.reshape(batch_size, n_samples, 3)
			gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2
			gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
		else:
			gradient_error = 0

		# Curvature loss
		if compute_curvature_loss:
			hessians = hessians.reshape(batch_size, n_samples, 3)
			curvature_error = hessians.sum(dim=-1).abs()
			if self.curvature_cap is not None:
#				curvature_error = curvature_error.clip(0, self.curvature_cap)
				y = curvature_error / self.curvature_cap
				y = torch.clamp(y, min=1e-8)
				curvature_error = (y < 1).float() * (-y ** 2 + 2 * y) + \
									(y >= 1).float() * (-(((y - 1) / (2 * y)) ** 2) / 2).exp()
			curvature_error = (relax_inside_sphere * curvature_error).sum() / (relax_inside_sphere.sum() + 1e-5)
		else:
			curvature_error = 0

		# Temporal smoothness error
		if compute_temporal_smoothness_error:
#			sdf_prev = sdf_network.sdf(self.to_timed_pts(pts, indices - 0.5), indices - 0.5)
#			sdf_next = sdf_network.sdf(self.to_timed_pts(pts, indices + 0.5), indices + 0.5)
#			temporal_smoothness_error = (((sdf_prev + sdf_next) / 2 - sdf) ** 2).mean()

			random_pts = torch.empty(10000, 3).uniform_(-1, 1).to(pts)
			random_indices = torch.empty(10000, 1).uniform_(0, 1).to(pts)
			gradients_all = sdf_network.gradient(self.to_timed_pts(random_pts, random_indices), random_indices, compute_hessian=False)
			temporal_smoothness_error = gradients_all[...,3:].abs().mean()
		else:
			temporal_smoothness_error = 0

		# Unimodality error
		unimodality_error = unimodality_loss(alpha, transmittance).mean()

		# Override color here for visualization
		if (self.visualize_shadow or visualize_shadow) and self.enable_shadow:
			shadow_maps = (torch.stack(shadow_maps).mean(dim=0).view(batch_size, n_samples) * mask_weights[:,:n_samples]).sum(dim=1)
			color = torch.ones_like(color) * shadow_maps[:,None] + (1.0 - weights_sum) * 0.5

		return {
			'color': color,
			'sdf': sdf,
			'pts': pts,
			'dists': dists,
			'gradients': gradients,
			'gradients_time': gradients_time,
			's_val': 1.0 / inv_s,
			'weights': mask_weights_solid,
			'opacity': alpha,
			'transmittance': transmittance_solid,
			'visibility': visibility,
			'gradient_error': gradient_error,
			'curvature_error': curvature_error,
			'unimodality_error': unimodality_error,
			'temporal_smoothness_error': temporal_smoothness_error,
			'inside_sphere': inside_sphere,
			'shadow_camera': shadow_camera,
		}

	def render(self, rays_o, rays_d, indices, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, 
		proj_params=[], illum_params=None, compute_curvature_loss=False, compute_temporal_smoothness_error=False, 
		shadow_field_ratio=False, disable_shadow=False, visualize_shadow=False):
		batch_size = len(rays_o)
		sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
		z_vals = torch.linspace(0.0, 1.0, self.n_samples)
		z_vals = near + (far - near) * z_vals[None, :]

		compute_outside = (self.n_outside > 0 or self.allow_subsurface_scattering or self.allow_volume_scattering) and (np.random.rand() >= self.outside_dropout_ratio)
		z_vals_outside = None
		if compute_outside > 0:
			z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

		n_samples = self.n_samples
		perturb = self.perturb

		if perturb_overwrite >= 0:
			perturb = perturb_overwrite
		if perturb > 0:
			t_rand = (torch.rand([batch_size, 1]) - 0.5)
			z_vals = z_vals + t_rand * 2.0 / self.n_samples

			if compute_outside:
				mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
				upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
				lower = torch.cat([z_vals_outside[..., :1], mids], -1)
				t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
				z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand
		else:
			z_vals = z_vals.expand(batch_size, z_vals.shape[1])

		if compute_outside:
			z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

		background_alpha = None
		background_sampled_color = None
		background_density = None

		# Up sample
		if self.n_importance > 0:
			z_vals, _, _, n_samples = self.importance_sampling(rays_o, rays_d, indices, z_vals, self.n_importance, self.up_sample_steps)

		# Background model
		if compute_outside:
			z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
			z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
			ret_outside = self.render_core_outside(rays_o, rays_d, indices, z_vals_feed, sample_dist, self.nerf,
									proj_params=proj_params, illum_params=illum_params)

			background_sampled_color = ret_outside['sampled_color']
			background_alpha = ret_outside['alpha']
			background_density = ret_outside['density']

		# Render core
		render_args = dict(
			background_rgb=background_rgb,
			background_alpha=background_alpha,
			background_sampled_color=background_sampled_color,
			background_density=background_density,
			cos_anneal_ratio=cos_anneal_ratio,
			proj_params=proj_params,
			illum_params=illum_params,
			compute_curvature_loss=compute_curvature_loss,
			compute_temporal_smoothness_error=compute_temporal_smoothness_error,
			shadow_field_ratio=shadow_field_ratio,
			disable_shadow=disable_shadow,
			visualize_shadow=visualize_shadow
		)

		if isinstance(indices, torch.Tensor):
			indices = int(indices.item())
		ret_fine = self.render_core(rays_o,	rays_d, indices, z_vals, sample_dist, 
									self.sdf_network, self.deviation_network, self.color_network, 
									is_reference_frame=is_reference_frame, **render_args)

		color_fine = ret_fine['color']
		weights = ret_fine['weights']
		weights_sum = weights.sum(dim=-1, keepdim=True)
		opacity = ret_fine['opacity']
		transmittance = ret_fine['transmittance']
		visibility = ret_fine['visibility']
		gradients = ret_fine['gradients']
		s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

		ret = {
			'color_fine': color_fine,
			's_val': s_val,
			'weight_sum': weights_sum,
			'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
			'gradients': gradients,
			'weights': weights,
			'opacity': opacity,
			'transmittance': transmittance,
			'visibility': visibility,
			'sdf': ret_fine['sdf'],
			'pts': ret_fine['pts'],
			'gradient_error': ret_fine['gradient_error'],
			'curvature_error': ret_fine['curvature_error'],
			'unimodality_error': ret_fine['unimodality_error'],
			'temporal_smoothness_error': ret_fine['temporal_smoothness_error'],
			'inside_sphere': ret_fine['inside_sphere'],
			'shadow_camera': ret_fine['shadow_camera'],
		}

		return ret


	def extract_fields(self, bound_min, bound_max, resolution, swap_time_axis=None):
		return extract_fields(bound_min,
							bound_max,
							resolution=resolution,
							query_func=lambda pts: -self.sdf_network.sdf(self.to_timed_pts(pts, 0), 0),
							flat_axis=swap_time_axis if swap_time_axis is not None else 0)

	def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0, time=0, swap_time_axis=None):
		network = self.sdf_network

		def query_func(pts):
			if swap_time_axis is not None:
				time_ = (pts[...,swap_time_axis].mean() - bound_min[swap_time_axis]) / (bound_max[swap_time_axis] - bound_min[swap_time_axis]) * self.n_images
				time_ = min(max(time_, 0), self.n_images - 1)
				pts[...,swap_time_axis] = 0
			else:
				time_ = time
			return -network.sdf(self.to_timed_pts(pts, time_), time_)

		return extract_geometry(bound_min,
								bound_max,
								resolution=resolution,
								threshold=threshold,
								query_func=query_func,
								flat_axis=swap_time_axis if swap_time_axis is not None else 0)


	def train(self):
		self.nerf.train()
		self.sdf_network.train()
		self.deviation_network.train()
		self.color_network.train()
		if self.projection_network is not None:
			self.projection_network.train()
			if self.shadow_field is not None:
				self.shadow_field.train()


	def eval(self):
		self.nerf.eval()
		self.sdf_network.eval()
		self.deviation_network.eval()
		self.color_network.eval()
		if self.projection_network is not None:
			self.projection_network.eval()
			if self.shadow_field is not None:
				self.shadow_field.eval()
