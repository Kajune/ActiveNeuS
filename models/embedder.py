import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.create_embedding_fn()

	def create_embedding_fn(self):
		embed_fns = []
		d = self.kwargs['input_dims']
		out_dim = 0
		if self.kwargs['include_input']:
			embed_fns.append(lambda x: x)
			out_dim += d

		max_freq = self.kwargs['max_freq_log2']
		N_freqs = self.kwargs['num_freqs']

		if self.kwargs['log_sampling']:
			self.freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
		else:
			self.freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

		for freq in self.freq_bands:
			for p_fn in self.kwargs['periodic_fns']:
				embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
				out_dim += d

		self.embed_fns = embed_fns
		self.out_dim = out_dim

	def embed(self, inputs):
#		return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

		outputs = [inputs]
		for p_fn in self.kwargs['periodic_fns']:
			outputs.append(p_fn(inputs[...,None,:] * self.freq_bands[:,None]).view(*inputs.shape[:-1],-1))
		return torch.cat(outputs, dim=-1)



class ProgressiveEncoder(nn.Module):
	def __init__(self, encoder, multires, progressive_until, init_level=2):
		super().__init__()
		self.encoder = encoder
		self.progress = 0
		self.multires = multires
		self.progressive_until = progressive_until
		self.init_level = init_level
		self.cur_level = 0


	def forward(self, x):
		embed = self.encoder(x)
		if self.progressive_until > 0:
			self.cur_level = max(int(min(self.progress / self.progressive_until, 1) * self.multires), self.init_level)
			detach_dim = embed.shape[-1] // self.multires * self.cur_level
			if detach_dim > 0:
				embed = torch.cat([embed[...,:detach_dim], embed[...,detach_dim:].detach()], dim=-1)
			else:
				embed = embed.detach()

		return embed


	def update(self, progress):
		self.progress = progress
		if hasattr(self.encoder, "update"):
			return self.encoder.update()
		else:
			return {}



class WithInput(ProgressiveEncoder):
	def __init__(self, encoder, enc_dim, input_dim, multires, progressive_until=0):
		super().__init__(encoder, multires, progressive_until)
		self.encoding_dim = enc_dim + input_dim


	def forward(self, x):
		embed = super().forward(x)
		return torch.cat([embed, x], dim=-1)



class WithDefaultEncoder(ProgressiveEncoder):
	def __init__(self, encoder, enc_dim, multires, input_dims, dropout_ratio=0.0, progressive_until=0):
		super().__init__(encoder, multires, progressive_until)
		self.fourier, enc_dim_fourier = make_default_encoding(multires, input_dims)
		self.encoding_dim = enc_dim + enc_dim_fourier
		self.dropout_ratio = dropout_ratio


	def forward(self, x):
		embed = super().forward(x)

		if self.dropout_ratio > 0:
			embed = F.dropout(embed, p=self.dropout_ratio * (1 - self.progress), training=self.training)

		return torch.cat([self.fourier(x), embed], dim=-1)



def make_default_encoding(multires, input_dims):
	embed_kwargs = {
		'include_input': True,
		'input_dims': input_dims,
		'max_freq_log2': multires-1,
		'num_freqs': multires,
		'log_sampling': True,
		'periodic_fns': [torch.sin, torch.cos],
	}

	embedder_obj = Embedder(**embed_kwargs)
	def embed(x, eo=embedder_obj): return eo.embed(x)
	return embed, embedder_obj.out_dim


def make_fourier_encoding(multires, input_dims):
	import tinycudann as tcnn

	config = dict(
		otype="Frequency",
		n_frequencies=multires,
	)
	encoding = tcnn.Encoding(input_dims, config)
	encoding_dim = input_dims * 2 * multires
	return encoding, encoding_dim


def make_hashgrid_encoding_base(multires, input_dims):
	import tinycudann as tcnn

	min_logres = 5
	max_logres = 11
	dict_size = 20
	dim = 4
	r_min, r_max = 2 ** min_logres, 2 ** max_logres
	growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (multires - 1))

	config = dict(
		otype="HashGrid",
		n_levels=multires,
		n_features_per_level=dim,
		log2_hashmap_size=dict_size,
		base_resolution=2 ** min_logres,
		per_level_scale=growth_rate,
		interpolation="Smoothstep",
	)
	encoding = tcnn.Encoding(input_dims, config)
	encoding_dim = dim * multires
	return encoding, encoding_dim


def make_hashgrid_encoding(multires, input_dims, progressive_until=0):
	hash_encoding, encoding_dim = make_hashgrid_encoding_base(multires, input_dims)
	encoding = WithInput(hash_encoding, encoding_dim, 3, multires=multires, progressive_until=progressive_until)
	return encoding, encoding.encoding_dim


def make_hybrid_encoding(multires, input_dims, dropout_ratio=0.0, progressive_until=0):
	hash_encoding, encoding_dim = make_hashgrid_encoding_base(multires, input_dims)
	encoding = WithDefaultEncoder(hash_encoding, encoding_dim, multires, input_dims, dropout_ratio=dropout_ratio, progressive_until=progressive_until)
	return encoding, encoding.encoding_dim


def make_sh_encoding(multires, input_dims):
	import tinycudann as tcnn

	config = dict(
		otype="SphericalHarmonics",
		degree=multires,
	)
	encoding = tcnn.Encoding(input_dims, config)
	encoding_dim = multires ** 2

	def embed(x, eo=encoding):
		x = F.normalize(x, dim=-1)
		return eo(x)
	return embed, encoding_dim


def make_tensorf_encoding(multires, input_dims, dropout_ratio=0.0):
	from .tensorf import TensorVMSplit

	aabb = np.float32([[-1,-1,-1], [1,1,1]])
	gridSize = [2**multires, 2**multires, 2**multires]

	tensorf = TensorVMSplit(aabb, gridSize)
	encoding = WithDefaultEncoder(tensorf, tensorf.encoding_dim, multires, input_dims, dropout_ratio=dropout_ratio)

	return encoding, encoding.encoding_dim


def make_kplanes_encoding(multires, input_dims, dropout_ratio=0.0):
	from .field_components.encodings import KPlanesEncoding

	enc_dim = 16
	res = 128
	encoding = WithDefaultEncoder(KPlanesEncoding(resolution=[res,res,res], num_components=enc_dim, num_levels=multires), 
		enc_dim * multires, multires, input_dims, detach_enc=True, dropout_ratio=dropout_ratio)

	return encoding, encoding.encoding_dim


def make_ff_encoding(multires, input_dims):
	from .field_components.encodings import FFEncoding

#	basis = torch.normal(mean=0.0, std=1.0, size=(input_dims, multires))
	basis = torch.eye(input_dims, dtype=torch.float32)
	embed_kwargs = {
		'include_input': True,
		'in_dim': input_dims,
		'min_freq_exp': 0,
		'max_freq_exp': multires-1,
		'num_frequencies': multires,
		'basis': basis,
	}

	encoding = FFEncoding(**embed_kwargs)

	return encoding, encoding.get_out_dim()


def make_lff_encoding(multires, input_dims):
	from .field_components.encodings import FFEncoding

#	basis = torch.normal(mean=0.0, std=1.0, size=(input_dims, multires))
	basis = torch.eye(input_dims, dtype=torch.float32)
	embed_kwargs = {
		'include_input': True,
		'in_dim': input_dims,
		'min_freq_exp': 0,
		'max_freq_exp': multires-1,
		'num_frequencies': multires,
		'basis': basis,
		'learnable_basis': True,
	}

	encoding = FFEncoding(**embed_kwargs)

	return encoding, encoding.get_out_dim()


def make_nlff_encoding(multires, input_dims):
	from .field_components.encodings import FFEncoding

	def transform_fn(x):
		return torch.cat([x, x**3, 1/(torch.abs(x)+1e-6), 1/(x**2+1e-6), (x[...,None] * x[...,None,:]).view(*x.shape[:-1],-1)], dim=-1)

#	basis = torch.normal(mean=0.0, std=1.0, size=(input_dims, multires))
	basis = torch.zeros((input_dims * 4 + input_dims ** 2, multires))
	basis[:input_dims, :input_dims] = torch.eye(input_dims, dtype=torch.float32)
	embed_kwargs = {
		'include_input': True,
		'in_dim': input_dims,
		'min_freq_exp': 0,
		'max_freq_exp': multires-1,
		'num_frequencies': multires,
		'basis': basis,
		'learnable_basis': False,
		'transform': transform_fn,
	}

	encoding = FFEncoding(**embed_kwargs)

	return encoding, encoding.get_out_dim()


def make_rff_encoding(multires, input_dims):
	from .field_components.encodings import FFEncoding

	basis = torch.normal(mean=0.0, std=1.0, size=(input_dims, input_dims * 4))
#	basis = torch.eye(input_dims, dtype=torch.float32)
	embed_kwargs = {
		'include_input': True,
		'in_dim': input_dims,
		'min_freq_exp': 0,
		'max_freq_exp': multires-1,
		'num_frequencies': multires,
		'basis': basis,
		'learnable_basis': False,
	}

	encoding = FFEncoding(**embed_kwargs)

	return encoding, encoding.get_out_dim()


def make_hybnerf_encoding(multires, input_dims, dropout_ratio=0.0, progressive_until=0):
	ff_encoding, ff_dim = make_default_encoding(multires, input_dims)
	hash_encoding, hash_dim = make_hashgrid_encoding(multires, input_dims, progressive_until=progressive_until)

	class HybNeRFEncoding(nn.Module):
		def __init__(self):
			super().__init__()
			self.progress = 0
			self.ff_encoding = ff_encoding
			self.hash_encoding = hash_encoding
			self.encoding_dim = ff_dim
			self.basis_function = nn.Linear(ff_dim + hash_dim, ff_dim)
			init.uniform_(self.basis_function.weight, -1e-3, 1e-3)
			with torch.no_grad():
				self.basis_function.bias.copy_(torch.normal(mean=0.0, std=1e-3, size=self.basis_function.bias.shape))

		def forward(self, x):
			ff_feat = self.ff_encoding(x)
			hash_feat = self.hash_encoding(x)
			alpha = torch.sigmoid(self.basis_function(torch.cat([ff_feat, hash_feat], dim=-1))) * 2
			if self.progress < 0.1:
				alpha = alpha.detach()
			integrated_feature = ff_feat * alpha
			return integrated_feature

		def update(self, progress):
			self.progress = progress
			return {}

	return HybNeRFEncoding(), ff_dim


def get_embedder(encoding, multires, input_dims=3, dropout_ratio=0.0, progressive_until=0):
	if encoding == "default":
		return make_default_encoding(multires, input_dims)

	elif encoding == "fourier":
		return make_fourier_encoding(multires, input_dims)

	elif encoding == "ff":
		return make_ff_encoding(multires, input_dims)

	elif encoding == "lff":
		return make_lff_encoding(multires, input_dims)

	elif encoding == "nlff":
		return make_nlff_encoding(multires, input_dims)

	elif encoding == "rff":
		return make_rff_encoding(multires, input_dims)

	elif encoding == "hashgrid":
		return make_hashgrid_encoding(multires, input_dims, progressive_until=progressive_until)

	elif encoding == "sh":
		return make_sh_encoding(multires, input_dims)

	elif encoding == "hybrid":
		return make_hybrid_encoding(multires, input_dims, dropout_ratio=dropout_ratio, progressive_until=progressive_until)

	elif encoding == "hybnerf":
		return make_hybnerf_encoding(multires, input_dims, dropout_ratio=dropout_ratio, progressive_until=progressive_until)

	elif encoding == "tensorf":
		return make_tensorf_encoding(multires, input_dims, dropout_ratio=dropout_ratio)

	elif encoding == "kplanes":
		return make_kplanes_encoding(multires, input_dims, dropout_ratio=dropout_ratio)

	else:
		raise NotImplementedError()
