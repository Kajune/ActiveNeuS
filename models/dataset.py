import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

#from pytorch3d.transforms import so3_log_map, so3_exp_map
from models import camera
from .utils import GradientScaler


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



class IlluminationParams(nn.Module):
	def __init__(self, ambient, diffuse, emissive):
		super().__init__()
#		self.ambient = nn.Parameter(torch.tensor(np.clip(ambient, 0, 1) + 1e-3, dtype=torch.float))
		self.diffuse_denom = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
		self.emissive_denom = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
#		self.shadow = nn.Parameter(torch.tensor(1e-3, dtype=torch.float))

		self.register_buffer('ambient', torch.tensor(ambient, dtype=torch.float))
#		self.register_buffer('ambient', torch.tensor(1.0, dtype=torch.float))
		self.register_buffer('diffuse_base', torch.tensor(diffuse, dtype=torch.float))
		self.register_buffer('emissive_base', torch.tensor(emissive, dtype=torch.float))


	def forward(self):
		return {
			"ambient": self.ambient,
			"diffuse": self.diffuse_base / (F.relu(self.diffuse_denom) + 1e-8),
			"emissive": self.emissive_base / (F.relu(self.emissive_denom) + 1e-8),
#			"shadow": F.relu(self.shadow),
		}



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



class CameraParams(nn.Module):
	def __init__(self, n_images, camera_dict, pose_noise, scene_scale=1.0, dynamic=False):
		super().__init__()
		self.n_images = n_images
		self.dynamic = dynamic

		# world_mat is a projection matrix from world to image
#		self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

		# scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
		self.scale_mats_np = np.array([camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)])
		self.scale_mats_np[...,:3,:3] *= scene_scale
		self.cam_mats_np = np.array([camera_dict['cam_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)])
		self.consistent_images = np.allclose(np.std(self.cam_mats_np, axis=0), 0)
		if self.consistent_images:
			self.cam_mats_np = self.cam_mats_np[0]
		self.cam_poses_np = np.array([camera_dict['cam_pose_%d' % idx].astype(np.float32) for idx in range(self.n_images)])
		self.cam_poses_np_inv = np.linalg.inv(self.cam_poses_np)
		self.cam_rots = self.cam_poses_np[:,:3,:3]
		self.cam_tvecs = self.cam_poses_np[:,:3,3]
		self.cam_rots_inv = self.cam_poses_np_inv[:,:3,:3]
		self.cam_tvecs_inv = self.cam_poses_np_inv[:,:3,3]

		scale_mats = torch.from_numpy(self.scale_mats_np)   # [n_images, 4, 4]
		scale_mats_inv = torch.inverse(scale_mats)   # [n_images, 4, 4]
		cam_mats = torch.from_numpy(self.cam_mats_np)   # [n_images, 4, 4]
		cam_mats_inv = torch.inverse(cam_mats)  # [n_images, 4, 4]
		cam_rots = self.cam_rots_inv
		cam_tvecs = self.cam_tvecs_inv

		max_trans = np.mean(np.linalg.norm(self.cam_tvecs_inv[1:] - self.cam_tvecs_inv[:-1], axis=-1))
#		max_trans = np.mean(np.min(np.linalg.norm(self.cam_tvecs[np.newaxis,:,:] - self.cam_tvecs[:,np.newaxis,:], axis=-1), axis=-1))

		if pose_noise is not None:
			if "rotation" in pose_noise:
				rot_noise = (np.random.rand(len(self.cam_rots_inv), 3) * 2 - 1) * pose_noise["rotation"]
				rot_noise_matrix = Rot.from_euler('zxy', rot_noise, degrees=True).as_matrix()
				cam_rots = rot_noise_matrix @ self.cam_rots_inv
			if "translation" in pose_noise:
				trans_noise = (np.random.rand(*cam_tvecs.shape) * 2 - 1) * pose_noise["translation"] * max_trans / 100
				cam_tvecs = cam_tvecs + trans_noise
			if "consective" in pose_noise:
				rot_noise = (np.random.rand(len(self.cam_rots_inv), 3) * 2 - 1) * pose_noise["consective"]["rotation"]
				rot_noise_matrix = Rot.from_euler('zxy', rot_noise, degrees=True).as_matrix()
				trans_noise = (np.random.rand(*cam_tvecs.shape) * 2 - 1) * pose_noise["consective"]["translation"] * max_trans / 100

				current_rot = self.cam_rots_inv[0]
				current_tvec = self.cam_tvecs_inv[0]
				rot_list = [current_rot]
				tvec_list = [current_tvec]
				for i in range(len(cam_tvecs)-1):
					current_rot = rot_noise_matrix[i] @ self.cam_rots_inv[i+1] @ np.linalg.inv(self.cam_rots_inv[i]) @ current_rot
					current_tvec += (current_rot @ np.linalg.inv(self.cam_rots_inv[i])) @ (trans_noise[i] + self.cam_tvecs_inv[i+1] - self.cam_tvecs_inv[i])
					rot_list.append(current_rot.copy())
					tvec_list.append(current_tvec.copy())
				cam_rots = np.array(rot_list)
				cam_tvecs = np.array(tvec_list)

#		pose_tmp = self.cam_poses_np_inv.copy()
#		pose_tmp[:,:3,:3] = cam_rots
#		pose_tmp[:,:3,3] = cam_tvecs
#		pose_tmp = np.linalg.inv(pose_tmp)
#		cam_rots = torch.from_numpy(np.array([cv2.Rodrigues(rmat)[0][:,0] for rmat in pose_tmp[:,:3,:3]]))
#		cam_rots = camera.Lie().SO3_to_so3(torch.from_numpy(pose_tmp[:,:3,:3]))
#		cam_tvecs = torch.from_numpy(pose_tmp[:,:3,3])
#		cam_rots_gt = torch.from_numpy(np.array([cv2.Rodrigues(rmat)[0][:,0] for rmat in self.cam_rots]))
#		cam_rots_gt = camera.Lie().SO3_to_so3(torch.from_numpy(self.cam_rots))
#		cam_tvecs_gt = torch.from_numpy(self.cam_tvecs)

		cam_rots = torch.from_numpy(np.float32([cv2.Rodrigues(rot)[0][:,0] for rot in cam_rots]))
		cam_rots_gt = torch.from_numpy(np.float32([cv2.Rodrigues(rot)[0][:,0] for rot in self.cam_rots_inv]))
		cam_tvecs = torch.from_numpy(cam_tvecs.astype(np.float32))
		cam_tvecs_gt = torch.from_numpy(self.cam_tvecs_inv.astype(np.float32))
		if self.dynamic:
			for i in range(len(cam_rots)):
				cam_rots[i] = cam_rots[0]
				cam_tvecs[i] = cam_tvecs[0]

		self.cam_rots = nn.Parameter(cam_rots)
		self.cam_tvecs = nn.Parameter(cam_tvecs)
		self.register_buffer('scale_mats', scale_mats)
		self.register_buffer('scale_mats_inv', scale_mats_inv)
		self.register_buffer('cam_mats', cam_mats)
		self.register_buffer('cam_mats_inv', cam_mats_inv)
		self.register_buffer('cam_rots_gt', cam_rots_gt)
		self.register_buffer('cam_tvecs_gt', cam_tvecs_gt)

	#
	# Intrinsic matrix
	#

	def get_cam_mat(self, img_idx):
		if self.consistent_images:
			return self.cam_mats
		else:
			return self.cam_mats[img_idx]


	def get_cam_mat_inv(self, img_idx):
		if self.consistent_images:
			return self.cam_mats_inv
		else:
			return self.cam_mats_inv[img_idx]

	#
	# Camera extrincis matrix
	#

	def get_scale_mat(self, img_idx):
		if self.consistent_images:
			return self.scale_mats[0]
		else:
			return self.scale_mats[img_idx]


	def get_scale_mat_inv(self, img_idx):
		if self.consistent_images:
			return self.scale_mats_inv[0]
		else:
			return self.scale_mats_inv[img_idx]


	def get_cam_pose(self, img_idx):
		pose = torch.cat([self.get_cam_rot(img_idx), self.get_cam_tvec(img_idx).unsqueeze(-1)], dim=-1)
		return to_homo(pose)


	def get_cam_pose_gt(self, img_idx):
		pose = torch.cat([self.get_cam_rot_gt(img_idx), self.get_cam_tvec_gt(img_idx).unsqueeze(-1)], dim=-1)
		return to_homo(pose)


	def get_cam_pose_inv(self, img_idx):
		pose = torch.cat([self.get_cam_rot_inv(img_idx), self.get_cam_tvec_inv(img_idx).unsqueeze(-1)], dim=-1)
		return to_homo(pose)


	def get_cam_pose_inv_gt(self, img_idx):
		pose = torch.cat([self.get_cam_rot_inv_gt(img_idx), self.get_cam_tvec_inv_gt(img_idx).unsqueeze(-1)], dim=-1)
		return to_homo(pose)


	def get_cam_rot(self, img_idx):
		return self.get_cam_rot_inv(img_idx).transpose(-1, -2)


	def get_cam_rot_inv(self, img_idx):
		if isinstance(img_idx, int) or img_idx.dim() == 0:
			return camera.Lie().so3_to_SO3(self.cam_rots[img_idx].unsqueeze(0))[0]
		else:
			return gather_unique(img_idx, lambda indices: camera.Lie().so3_to_SO3(self.cam_rots[indices]))


	def get_cam_rot_gt(self, img_idx):
		return self.get_cam_rot_inv_gt(img_idx).transpose(-1, -2)


	def get_cam_rot_inv_gt(self, img_idx):
		if isinstance(img_idx, int) or img_idx.dim() == 0:
			return camera.Lie().so3_to_SO3(self.cam_rots_gt[img_idx].unsqueeze(0))[0]
		else:
			return gather_unique(img_idx, lambda indices: camera.Lie().so3_to_SO3(self.cam_rots_gt[indices]))


	def get_cam_tvec(self, img_idx):
		return -(self.get_cam_rot(img_idx) @ self.cam_tvecs[img_idx].unsqueeze(-1))[...,0]


	def get_cam_tvec_inv(self, img_idx):
		return (self.scale_mats_inv[img_idx,:3,:3] @ self.cam_tvecs[img_idx].unsqueeze(-1))[...,0] + self.scale_mats_inv[img_idx,:3,3]
#		return self.scale_mats[img_idx,:3,:3] @ self.cam_tvecs[img_idx] + self.scale_mats[img_idx,:3,3]


	def get_cam_tvec_gt(self, img_idx):
		return -(self.get_cam_rot_gt(img_idx) @ self.cam_tvecs_gt[img_idx].unsqueeze(-1))[...,0]


	def get_cam_tvec_inv_gt(self, img_idx):
		return (self.scale_mats_inv[img_idx,:3,:3] @ self.cam_tvecs_gt[img_idx].unsqueeze(-1))[...,0] + self.scale_mats_inv[img_idx,:3,3]
#		return self.scale_mats[img_idx,:3,:3] @ self.cam_tvecs[img_idx] + self.scale_mats[img_idx,:3,3]



class ProjectorParams(nn.Module):
	def __init__(self, n_images, camera_dict, pose_noise, consistent_images, consistent_proj_noise):
		super().__init__()
		self.n_images = n_images
		self.consistent_images = consistent_images
		self.consistent_proj_noise = consistent_proj_noise

		# proj_mat
		self.proj_mats_np = np.array([camera_dict['proj_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)])
		self.proj_poses_np = np.array([np.linalg.inv(camera_dict['cam_proj_pose_%d' % idx].astype(np.float32)) for idx in range(self.n_images)])
		self.proj_rots_np = self.proj_poses_np[:,:,:3,:3]
		self.proj_tvecs_np = self.proj_poses_np[:,:,:3,3]
		proj_rots_gt = proj_rots = camera.Lie().SO3_to_so3(torch.from_numpy(self.proj_rots_np).view(-1,3,3)).view(self.proj_rots_np.shape[0],self.proj_rots_np.shape[1],3)
		proj_tvecs_gt = proj_tvecs = torch.from_numpy(self.proj_tvecs_np)
		if self.consistent_images:
			self.proj_mats_np = self.proj_mats_np[0]
			self.proj_rots_np = self.proj_rots_np[0]
			self.proj_tvecs_np = self.proj_tvecs_np[0]
			proj_rots = proj_rots[0]
			proj_tvecs = proj_tvecs[0]


		self.max_trans = np.max(np.linalg.norm(self.proj_tvecs_np[np.newaxis,:,:] - self.proj_tvecs_np[:,np.newaxis,:], axis=-1))

		if pose_noise is not None:
			if "rotation" in pose_noise:
				rot_noise = (np.random.rand(len(self.proj_rots_np), 3) * 2 - 1) * pose_noise["rotation"]
				rot_noise_matrix = Rot.from_euler('zxy', rot_noise, degrees=True).as_matrix()
#				proj_rots = camera.Lie().SO3_to_so3(torch.from_numpy(rot_noise_matrix @ self.proj_rots)).float().to(proj_rots_gt)
				proj_rots_noise = camera.Lie().SO3_to_so3(torch.from_numpy(rot_noise_matrix)).float().to(proj_rots_gt)
			if "translation" in pose_noise:
				trans_noise = (torch.rand(*proj_tvecs.shape).float().to(proj_tvecs) * 2 - 1) * pose_noise["translation"] * self.max_trans / 100
#				proj_tvecs = proj_tvecs + trans_noise
				proj_tvecs_noise = trans_noise

			if self.consistent_proj_noise:
				proj_rots_noise = proj_rots_noise[...,0,:]
				proj_tvecs_noise = proj_tvecs_noise[...,0,:]

		else:
			proj_rots_noise = torch.from_numpy(np.zeros_like(proj_rots))
			proj_tvecs_noise = torch.from_numpy(np.zeros_like(proj_tvecs))

		proj_mats = torch.from_numpy(self.proj_mats_np)
		self.proj_rots_noise = nn.Parameter(proj_rots_noise)
		self.proj_tvecs_noise = nn.Parameter(proj_tvecs_noise)
#		self.register_buffer('proj_tvecs_noise', proj_tvecs_noise)
		self.register_buffer('proj_mats', proj_mats)
		self.register_buffer('proj_rots', proj_rots)
		self.register_buffer('proj_tvecs', proj_tvecs)
		self.register_buffer('proj_rots_gt', proj_rots_gt)
		self.register_buffer('proj_tvecs_gt', proj_tvecs_gt)

	#
	# Intrinsic matrix
	#

	def get_proj_mat(self, img_idx):
		if self.consistent_images:
			return self.proj_mats.unsqueeze(0)
		else:
			return self.proj_mats[img_idx]

	#
	# Projector extrinsic matrix
	#

	def get_proj_pose(self, img_idx):
		rot = self.get_proj_rot(img_idx).transpose(-1,-2)
		tvec = rot @ -self.get_proj_tvec(img_idx).unsqueeze(-1)
		pose = torch.cat([rot, tvec], dim=-1)
		return to_homo(pose)


	def get_proj_pose_gt(self, img_idx):
		rot = self.get_proj_rot_gt(img_idx).transpose(-1,-2)
		tvec = rot @ -self.get_proj_tvec_gt(img_idx).unsqueeze(-1)
		pose = torch.cat([rot, tvec], dim=-1)
		return to_homo(pose)


	def get_proj_tvec_noise(self):
		max_noise = self.max_trans / len(self.proj_mats_np)
		return GradientScaler(1e-3)(self.proj_tvecs_noise).clamp(-max_noise, max_noise)


	def get_proj_rot(self, img_idx):
		if self.consistent_images:
			if self.consistent_proj_noise:
				return (camera.Lie().so3_to_SO3(self.proj_rots_noise) @ camera.Lie().so3_to_SO3(self.proj_rots)).unsqueeze(0)
			else:
				return (camera.Lie().so3_to_SO3(self.proj_rots_noise) @ camera.Lie().so3_to_SO3(self.proj_rots)).unsqueeze(0)
		else:
			if self.consistent_proj_noise:
				return camera.Lie().so3_to_SO3(self.proj_rots_noise[img_idx]) @ camera.Lie().so3_to_SO3(self.proj_rots[img_idx])
			else:
				return camera.Lie().so3_to_SO3(self.proj_rots_noise[img_idx]) @ camera.Lie().so3_to_SO3(self.proj_rots[img_idx])


	def get_proj_tvec(self, img_idx):
		if self.consistent_images:
			if self.consistent_proj_noise:
				return (self.proj_tvecs + self.get_proj_tvec_noise()).unsqueeze(0)
			else:
				return (self.proj_tvecs + self.get_proj_tvec_noise()).unsqueeze(0)
		else:
			if self.consistent_proj_noise:
				return self.proj_tvecs[img_idx] + self.get_proj_tvec_noise()[img_idx]
			else:
				return self.proj_tvecs[img_idx] + self.get_proj_tvec_noise()[img_idx]


	def get_proj_rot_gt(self, img_idx):
		if self.consistent_images:
			return camera.Lie().so3_to_SO3(self.proj_rots_gt[img_idx]).unsqueeze(0)
		else:
			return camera.Lie().so3_to_SO3(self.proj_rots_gt[img_idx])


	def get_proj_tvec_gt(self, img_idx):
		if self.consistent_images:
			return self.proj_tvecs_gt[img_idx].unsqueeze(0)
		else:
			return self.proj_tvecs_gt[img_idx]



class ProjectionPattern(nn.Module):
	def __init__(self, patterns, pattern_noise, consistent_images, consistent_projectors):
		super().__init__()
		self.n_images = patterns.shape[0]
		self.n_projectors = patterns.shape[1]
		height = patterns.shape[2]
		width = patterns.shape[3]
		self.consistent_images = consistent_images
		self.consistent_projectors = consistent_projectors

		if consistent_images:
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

#		self.register_buffer('patterns', torch.from_numpy(patterns))
		self.patterns = nn.Parameter(torch.from_numpy(patterns))


	def correct_pattern(self):
		return self.patterns
#		return torch.tanh(self.patterns)
#		mask = torch.any(self.patterns > 0, dim=-1)
#		pattern = self.patterns + (self.color_correction[...,np.newaxis,np.newaxis,:] * mask[...,np.newaxis])
#		affine_grid = F.affine_grid(self.affine_correction, pattern.view(-1,*pattern.shape[-3:]).shape, align_corners=None)
#		pattern = F.grid_sample(pattern.view(-1,*pattern.shape[-3:]), affine_grid).view(*pattern.shape)
#		return pattern


	def get_pattern(self, img_idx):
		if self.consistent_images:
			if self.consistent_projectors:
				return torch.stack([self.correct_pattern()] * self.n_projectors).unsqueeze(0)
			else:
				return self.correct_pattern().unsqueeze(0)
		else:
			if self.consistent_projectors:
				return torch.stack([self.correct_pattern()[img_idx]] * self.n_projectors)
			else:
				return self.correct_pattern()[img_idx]


class Dataset:
	def __init__(self, conf, pose_noise={}, proj_pose_noise={}, pattern_noise={}, 
				cam_refractive_interface_params=None, proj_refractive_interface_params=None, 
				event_mode=None, scene_scale=1.0, dynamic=False):
		super(Dataset, self).__init__()
		print('Load data: Begin')
		self.device = torch.device('cuda')
		self.conf = conf
		self.event_mode = event_mode
		self.dynamic = dynamic

		self.data_dir = conf.get_string('data_dir')
		self.render_cameras_name = conf.get_string('render_cameras_name')
		self.object_cameras_name = conf.get_string('object_cameras_name')

		self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
		self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

		self.with_projection = conf.get_bool('with_projection', default=False)
		self.with_scatter = conf.get_bool('with_scatter', default=False)

		camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name), allow_pickle=True)
		self.camera_dict = camera_dict
		self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
		self.n_images = len(self.images_lis)
		if self.event_mode is not None:
			# Quantinaztion error handling
			self.images_np = (np.stack([cv2.imread(im_name) for im_name in self.images_lis]) - 0.5) / 255.0

			if self.event_mode == "accumulated":
				img_tmp = np.ones_like(self.images_np[0]) * 0.5
				img_list = [img_tmp]
				for i in range(len(self.images_np) - 1):
					img_tmp = (img_tmp - (self.images_np[i] - 0.5)).clip(0, 1)
					img_list.append(img_tmp)
				self.images_np = np.array(img_list)

			elif self.event_mode == "interval":
				img_list = []
				for i in range(len(self.images_np)):
					img_tmp = np.ones_like(self.images_np[i]) * 0.5
					for t in range(min(10, self.n_images - i - 1)):
						img_tmp = (img_tmp + (self.images_np[i+t] - 0.5)).clip(0, 1)
					img_list.append(img_tmp)
				self.images_np = np.array(img_list)

			elif self.event_mode == "mixed":
#				img_tmp = np.ones_like(self.images_np[0]) * 0.5
#				img_list = [img_tmp]
#				for i in range(len(self.images_np) - 1):
#					img_tmp = (img_tmp - (self.images_np[i] - 0.5)).clip(0, 1)
#					img_list.append(img_tmp)

#				for i in range(len(self.images_np)):
#					if i % 3 == 2:
#						img_tmp = np.ones_like(self.images_np[i]) * 0.5
#						for t in range(10):
#							img_tmp = (img_tmp + (self.images_np[(i+t) % self.n_images] - 0.5)).clip(0, 1)
#						img_list[i] = img_tmp
#					elif i % 3 == 0:
#						img_list[i] = self.images_np[i]

				img_list = []
				for i in range(len(self.images_np)):
					img_tmp = np.ones_like(self.images_np[i]) * 0.5
					if i % 3 == 2:
						for t in range(min(10, self.n_images - i - 1)):
							img_tmp = (img_tmp + (self.images_np[i+t] - 0.5)).clip(0, 1)
					elif i % 3 == 1:
						for t in range(min(5, self.n_images - i - 1)):
							img_tmp = (img_tmp + (self.images_np[i+t] - 0.5)).clip(0, 1)
					else:
						img_tmp = self.images_np[i]

					img_list.append(img_tmp)

				self.images_np = np.array(img_list)


		else:
			self.images_np = np.stack([cv2.imread(im_name) for im_name in self.images_lis]) / 255.0

		self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
		self.masks_np = np.stack([cv2.imread(im_name) for im_name in self.masks_lis]) / 255.0
		self.with_cam_refraction = conf.get_bool('with_cam_refraction', default="camera_refraction" in camera_dict)
		self.with_proj_refraction = conf.get_bool('with_proj_refraction', default="projector_refraction" in camera_dict)

		if self.with_projection:
			self.patterns_lis = []
			self.patterns_np = []
			for image_path in self.images_lis:
				frame = os.path.splitext(os.path.basename(image_path))[0]
				patterns_lis_tmp = sorted(glob(os.path.join(self.data_dir, "pattern/%s*.png" % frame)))
				patterns_np_tmp = np.stack([cv2.imread(im_name) for im_name in patterns_lis_tmp]) / 255.0
				self.patterns_lis.append(patterns_lis_tmp)
				self.patterns_np.append(patterns_np_tmp)
			self.patterns_np = np.stack(self.patterns_np)

		self.camera_params = CameraParams(self.n_images, camera_dict, pose_noise, scene_scale, self.dynamic).to(self.device)
		if self.with_projection:
			self.projector_params = ProjectorParams(self.n_images, camera_dict, proj_pose_noise, 
													conf.get_bool('consistent_images'), conf.get_bool('consistent_proj_noise', default=False)).to(self.device)

		self._images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
		self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]

		if self.with_projection:
			self.projection_pattern = ProjectionPattern(self.patterns_np, pattern_noise,
				conf.get_bool('consistent_images'), conf.get_bool('consistent_projectors')).to(self.device)
			self.illumination_params = IlluminationParams(**camera_dict["illumination_params"].item()).to(self.device)

		if self.with_cam_refraction:
			params = cam_refractive_interface_params if cam_refractive_interface_params is not None else camera_dict["camera_refraction"].item()
			self.cam_refractive_interface = RefractiveInterface(**params, scale=self.camera_params.get_scale_mat(0)[0,0]).to(self.device)

		if self.with_proj_refraction:
			params_list = proj_refractive_interface_params if proj_refractive_interface_params is not None else camera_dict["projector_refraction"]
			self.proj_refractive_interface = nn.ModuleList([RefractiveInterface(**params, scale=self.camera_params.get_scale_mat(0)[0,0]) for params in params_list]).to(self.device)

		self.H, self.W = self._images.shape[1], self._images.shape[2]
		self.image_pixels = self.H * self.W

		self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
		self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])

		print('Load data: End')


	def get_proj_params(self, img_indices):
		patterns = self.projection_pattern.get_pattern(img_indices).transpose(0,1)
		scale_mats = self.camera_params.get_scale_mat(img_indices)
		scale_mats = scale_mats.unsqueeze(0).expand(patterns.shape[0],*scale_mats.shape)
		cam_poses = self.camera_params.get_cam_pose(img_indices).view(1,-1,4,4).expand(patterns.shape[0],-1,4,4)
		proj_mats = self.projector_params.get_proj_mat(img_indices).transpose(0,1)
		proj_poses = self.projector_params.get_proj_pose(img_indices).transpose(0,1)
		proj_params = []
		for pattern, proj_mat, cam_pose, proj_pose, scale_mat in zip(patterns, proj_mats, cam_poses, proj_poses, scale_mats):
			proj_params.append({"pattern": pattern, "proj_mat": proj_mat, "cam_pose": cam_pose, "proj_pose": proj_pose, "scale_mat": scale_mat})

		if self.with_proj_refraction:
			for i in range(len(proj_params)):
				proj_params[i]["refractive_interface"] = self.proj_refractive_interface[i]

		return proj_params


	def interpolate_pose(self, pose_0, pose_1, ratio):
		pose_0 = pose_0.cpu().detach().numpy()
		pose_1 = pose_1.cpu().detach().numpy()
		rots = Rot.from_matrix(np.stack([pose_0[:3, :3], pose_1[:3, :3]]))
		pose = np.eye(4).astype(np.float32)
		pose[:3, :3] = Slerp([0, 1], rots)(ratio).as_matrix()
		pose[:3, 3] = (1.0 - ratio) * pose_0[:3, 3] + ratio * pose_1[:3, 3]
		return torch.from_numpy(pose).to(self.device)


	def gen_rays_at(self, img_idx, resolution_level=1):
		"""
		Generate rays at world space from one camera.
		"""
		l = resolution_level
		tx = torch.linspace(0, self.W - 1, int(self.W // l))
		ty = torch.linspace(0, self.H - 1, int(self.H // l))
		pixels_x, pixels_y = torch.meshgrid(tx, ty)
		p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
		p = torch.matmul(self.camera_params.get_cam_mat_inv(img_idx)[None, None, :3, :3], p[:, :, :, None])[...,0]  # W, H, 3
		rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
		rays_o = torch.zeros_like(rays_v)

		if self.with_cam_refraction:
			rays_o, rays_v = self.cam_refractive_interface.backward_projection(rays_o, rays_v)

		rot = self.camera_params.get_cam_rot_inv(img_idx)[None, None]
		rays_v = torch.matmul(rot, rays_v[:, :, :, None])[...,0]  # W, H, 3
		rays_o = torch.matmul(rot, rays_o[:, :, :, None])[...,0]  # W, H, 3
		rays_o += self.camera_params.get_cam_tvec_inv(img_idx)[None, None].expand(rays_v.shape)  # W, H, 3
		return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


	def gen_random_pixels(self, batch_size):
		pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()
		pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()
		return pixels_x, pixels_y


	@property
	def images(self):
		return self._images


	def gen_random_rays_multi(self, img_indices, batch_size, pixels=None):
		"""
		Generate random rays at world space from multiple camera.
		"""
		indices = img_indices[torch.randint(low=0, high=len(img_indices), size=[batch_size])].cpu()
		if pixels is not None:
			pixels_x, pixels_y = pixels
		else:
			pixels_x, pixels_y = self.gen_random_pixels(batch_size)
		color = self.images[(indices, pixels_y, pixels_x)].to(self.device)    # batch_size, 3
		mask = self.masks[(indices, pixels_y, pixels_x)].to(self.device)      # batch_size, 3
		p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)  # batch_size, 3
		p = torch.matmul(self.camera_params.get_cam_mat_inv(indices)[..., :3, :3], p[:, :, None])[...,0] # batch_size, 3
		rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
		rays_o = torch.zeros_like(rays_v)

		if self.with_cam_refraction:
			rays_o, rays_v = self.cam_refractive_interface.backward_projection(rays_o, rays_v)

		rot = self.camera_params.get_cam_rot_inv(indices)
		rays_v = torch.matmul(rot, rays_v[:, :, None])[...,0]  # batch_size, 3
		rays_o = torch.matmul(rot, rays_o[:, :, None])[...,0]  # batch_size, 3
		rays_o += self.camera_params.get_cam_tvec_inv(indices).expand(rays_v.shape) # batch_size, 3
		return indices, torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1)    # batch_size, 10


	def gen_random_rays_at(self, img_idx, batch_size, pixels=None):
		"""
		Generate random rays at world space from one camera.
		"""
		if pixels is not None:
			pixels_x, pixels_y = pixels
		else:
			pixels_x, pixels_y = self.gen_random_pixels(batch_size)

		color = self.images[img_idx][(pixels_y, pixels_x)].to(self.device)    # batch_size, 3
		mask = self.masks[img_idx][(pixels_y, pixels_x)].to(self.device)      # batch_size, 3
		p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)  # batch_size, 3
		p = torch.matmul(self.camera_params.get_cam_mat_inv(img_idx)[None, :3, :3], p[:, :, None])[...,0] # batch_size, 3
		rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
		rays_o = torch.zeros_like(rays_v)

		if self.with_cam_refraction:
			rays_o, rays_v = self.cam_refractive_interface.backward_projection(rays_o, rays_v)

		rot = self.camera_params.get_cam_rot_inv(img_idx)[None]
		rays_v = torch.matmul(rot, rays_v[:, :, None])[...,0]  # batch_size, 3
		rays_o = torch.matmul(rot, rays_o[:, :, None])[...,0]  # batch_size, 3
		rays_o += self.camera_params.get_cam_tvec_inv(img_idx)[None].expand(rays_v.shape) # batch_size, 3
		return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1)    # batch_size, 10


	def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
		assert isinstance(idx_0, int) and isinstance(idx_1, int), "idx_0 and idx_1 must be int"
		"""
		Interpolate pose between two cameras.
		"""
		l = resolution_level
		tx = torch.linspace(0, self.W - 1, self.W // l)
		ty = torch.linspace(0, self.H - 1, self.H // l)
		pixels_x, pixels_y = torch.meshgrid(tx, ty)
		p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
		p = torch.matmul(self.camera_params.get_cam_mat_inv(0)[None, None, :3, :3], p[:, :, :, None])[...,0]  # W, H, 3
		rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
		rays_o = torch.zeros_like(rays_v)

		if self.with_cam_refraction:
			rays_o, rays_v = self.cam_refractive_interface.backward_projection(rays_o, rays_v)

		pose_0 = torch.linalg.inv(self.camera_params.get_cam_pose_inv(idx_0))
		pose_1 = torch.linalg.inv(self.camera_params.get_cam_pose_inv(idx_1))
		pose = torch.linalg.inv(self.interpolate_pose(pose_0, pose_1, ratio))
		rot = pose[:3, :3]
		trans = pose[:3, 3]

		rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None])[...,0]  # W, H, 3
		rays_o = torch.matmul(rot[None, None, :3, :3], rays_o[:, :, :, None])[...,0]  # W, H, 3
		rays_o += trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
		return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


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

		return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


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

