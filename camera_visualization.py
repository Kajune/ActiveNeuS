# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import open3d as o3d


def make_pcd_from_points(points, colors=None, estimate_normals=False):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3))
	if colors is not None:
		if colors.shape == points.shape:
			pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1,3))
		elif colors.shape == (3,):
			pcd.colors = o3d.utility.Vector3dVector(np.float32(colors * np.ones_like(points)).reshape(-1,3))
		else:
			raise NotImplementedError()

	if estimate_normals:
		pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

	return pcd


def get_camera_wireframe(scale: float = 0.3):
	D = 3
	a = 0.5 * torch.tensor([-2, 1.5, D])
	up1 = 0.5 * torch.tensor([0, 1.5, D])
	up2 = 0.5 * torch.tensor([0, 2, D])
	b = 0.5 * torch.tensor([2, 1.5, D])
	c = 0.5 * torch.tensor([-2, -1.5, D])
	d = 0.5 * torch.tensor([2, -1.5, D])
	C = torch.zeros(3)
	F = torch.tensor([0, 0, D])
	camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
	lines = torch.stack([x.float() for x in camera_points]) * scale
	return lines


def plot_cameras(ax, cam_poses, max_trans = None, color: str = "blue", scale=0.05):
	"""
	Plots a set of `cameras` objects into the maplotlib axis `ax` with
	color `color`.
	"""
	if max_trans is None:
		cam_tvecs = cam_poses[:,:3,3].cpu().numpy()
		max_trans = np.max(np.linalg.norm(cam_tvecs[np.newaxis,:,:] - cam_tvecs[:,np.newaxis,:], axis=-1))

	cam_wires_canonical = get_camera_wireframe(max_trans * scale).cuda()[None]
#    cam_trans = cameras.get_world_to_view_transform().inverse()
#    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
	cam_wires_canonical = cam_wires_canonical.expand(cam_poses.shape[0], cam_wires_canonical.shape[1], 3)
	cam_wires_canonical = torch.cat([cam_wires_canonical, torch.ones_like(cam_wires_canonical[...,:1])], dim=-1)
	cam_wires_trans = torch.matmul(cam_poses.cuda(), cam_wires_canonical.permute(0,2,1)).permute(0,2,1)[...,:3]
	plot_handles = []
	for wire in cam_wires_trans:
		x_, y_, z_ = wire.detach().cpu().numpy().T.astype(float)
		(h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
		plot_handles.append(h)

		indices = [0,4,5,6]
		x1 = x_[indices]
		y1 = y_[indices]
		z1 = z_[indices]
		verts = [list(zip(x1, y1, z1))]
		srf = Poly3DCollection(verts, alpha=.25, facecolor=color)
		ax.add_collection3d(srf)

	# h = ax.scatter(cam_tvecs[:,0], cam_tvecs[:,2], cam_tvecs[:,1], color=color)
	# plot_handles.append(h)
	# trans = cam_wires_trans.cpu().detach().numpy()[:,-1]
	# for i in range(len(cam_poses)):
	# 	(h,) = ax.plot([cam_tvecs[i,0], trans[i,0]], [cam_tvecs[i,2], trans[i,2]], [cam_tvecs[i,1], trans[i,1]], color=color)
	# 	plot_handles.append(h)

	return plot_handles


def plot_camera_scene(cameras, cameras_gt, status: str, scale=0.05, with_icp=True, legend=True):
	"""
	Plots a set of predicted cameras `cameras` and their corresponding
	ground truth locations `cameras_gt`. The plot is named with
	a string passed inside the `status` argument.
	"""
	cameras = cameras.clone()
	cameras_gt = cameras_gt.clone()

	fig = plt.figure(tight_layout=True)
	ax = fig.add_subplot(projection="3d")
	ax.clear()
	if status is not None:
		ax.set_title(status)

	if with_icp:
		threshold = 0.02
		trans_init = np.eye(4)
		source = make_pcd_from_points(cameras[:,:3,3].cpu().detach().numpy())
		target = make_pcd_from_points(cameras_gt[:,:3,3].cpu().detach().numpy())
		reg_p2p = o3d.pipelines.registration.registration_icp(
			source, target, threshold, trans_init,
			o3d.pipelines.registration.TransformationEstimationPointToPoint())
		cameras[:,:3,3] = torch.tensor(source.transform(reg_p2p.transformation).points).to(cameras)

	mean_tvec = (cameras[:,:3,3].mean(dim=0) + cameras_gt[:,:3,3].mean(dim=0)) / 2
	cameras[:,:3,3] -= mean_tvec
	cameras_gt[:,:3,3] -= mean_tvec

	cam_tvecs = cameras[:,:3,3].cpu().numpy()
	cam_tvecs_gt = cameras_gt[:,:3,3].cpu().numpy()
	max_trans = max(
		np.max(np.linalg.norm(cam_tvecs[np.newaxis,:,:] - cam_tvecs[:,np.newaxis,:], axis=-1)),
		np.max(np.linalg.norm(cam_tvecs_gt[np.newaxis,:,:] - cam_tvecs_gt[:,np.newaxis,:], axis=-1))
	)
	if max_trans == 0:
		max_trans = 1

	handle_cam = plot_cameras(ax, cameras, max_trans=max_trans, color="#FF7D1E", scale=scale)
	handle_cam_gt = plot_cameras(ax, cameras_gt, max_trans=max_trans, color="#812CE5", scale=scale)
#	handle_cam = plot_cameras(ax, cameras, color="red", scale=scale)
#	handle_cam_gt = plot_cameras(ax, cameras_gt, color="green", scale=scale)

	tvec = cameras[:,:3,3].cpu().numpy()
	tvec_gt = cameras_gt[:,:3,3].cpu().numpy()
	for i in range(len(tvec)):
		x1, y1, z1 = tvec[i]
		x2, y2, z2 = tvec_gt[i]
		ax.plot([x1, x2], [y1, y2], [z1, z2], color="red", linewidth=0.5)

	plot_radius = max(
		np.max(np.linalg.norm(tvec[np.newaxis,:] - tvec[:,np.newaxis], axis=-1)),
		np.max(np.linalg.norm(tvec_gt[np.newaxis,:] - tvec_gt[:,np.newaxis], axis=-1)),
	) / 2 * 0.8
	if plot_radius == 0:
		plot_radius = 0.1

	ax.set_xlim3d([-plot_radius, plot_radius])
	ax.set_ylim3d([-plot_radius, plot_radius])
	ax.set_zlim3d([-plot_radius, plot_radius])
#	ax.set_xlabel("x")
#	ax.set_ylabel("z")
#	ax.set_zlabel("y")
	labels_handles = {
		"Estimated cameras": handle_cam[0],
		"GT cameras": handle_cam_gt[0],
	}

	if legend:
		ax.legend(
			labels_handles.values(),
			labels_handles.keys(),
			loc="upper center",
			bbox_to_anchor=(0.5, 0),
		)
#	plt.show()
	return fig
