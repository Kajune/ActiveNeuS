import torch
import torch.nn as nn
import numpy as np


class GradientScaleFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input_forward: torch.Tensor, scale: torch.Tensor):
		ctx.save_for_backward(scale)
		return input_forward
 
	@staticmethod
	def backward(ctx, grad_backward: torch.Tensor):
		scale, = ctx.saved_tensors
		return scale * grad_backward, None



class GradientScaler(nn.Module): 
	def __init__(self, scale: float):
		super().__init__()
		self.scale = torch.tensor(scale)
 
	def forward(self, x: torch.Tensor):
		return GradientScaleFunction.apply(x, self.scale)



class QuadraticLayer(nn.Module):
	def __init__(self, d_in, d_out):
		super().__init__()
		self.lin1 = nn.Linear(d_in, d_out)
		self.lin2 = nn.Linear(d_in, d_out)
		self.lin3 = nn.Linear(d_in, d_out)


	def forward(self, x):
		return torch.mul(self.lin1(x), self.lin2(x)) + self.lin3(torch.square(x))



def init_lin2_lin3(m):
	nn.init.normal_(m.lin2.weight, mean=0.0, std=1e-5)
	nn.init.ones_(m.lin2.bias)
	nn.init.normal_(m.lin3.weight, mean=0.0, std=1e-5)
	nn.init.zeros_(m.lin3.bias)


def sine_init(m):
	with torch.no_grad():
		if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
			num_input = m.lin1.weight.size(-1)
			# See SIREN paper supplement Sec. 1.5 for discussion of factor 30
			m.lin1.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
			init_lin2_lin3(m)


def create_positional_encoding(W, H):
	x = torch.linspace(0, 1, W).repeat(H, 1)
	y = torch.linspace(0, 1, H).view(H, 1).repeat(1, W)
	pos_encoding = torch.stack([x, y], dim=-1)
	return pos_encoding


def create_gaussian_kernel(kernel_size, sigma):
	ax = torch.arange(kernel_size) - kernel_size // 2
	xx, yy = torch.meshgrid(ax, ax, indexing='ij')
	kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
	kernel = kernel / kernel.sum()
	return kernel


def find_peak_interval(img):
	unique_intensities = np.unique(img)

	min_cost = np.inf
	best_interval = 255

	for i in range(len(unique_intensities),255):
		intervals = np.linspace(0, 255, i)
		cost = np.sum(np.min(np.abs(intervals[:,None] - unique_intensities[None,:]), axis=0)) + len(intervals) - len(unique_intensities)

		if cost < min_cost:
			min_cost = cost
			best_interval = i

	return best_interval


class SingleConv(nn.Module):
	def __init__(
		self,
		kernel_size,
		stride = 1,
		padding = 0,
		dilation = 1,
		bias: bool = True,
		padding_mode: str = "zeros",
		device=None,
		dtype=None,
	) -> None:
		super().__init__()
		self.conv = nn.Conv2d(1, 1, kernel_size, stride, padding, dilation, 1, bias, padding_mode, device, dtype)

	@property
	def weight(self):
		return self.conv.weight
	
	@property
	def bias(self):
		return self.conv.bias

	def forward(self, x):
		return self.conv(x.view(-1,1,*x.shape[2:])).view(*x.shape)
