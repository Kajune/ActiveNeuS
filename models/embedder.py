import torch
import torch.nn as nn
import torch.nn.functional as F
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
			freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
		else:
			freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

		for freq in freq_bands:
			for p_fn in self.kwargs['periodic_fns']:
				embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
				out_dim += d

		self.embed_fns = embed_fns
		self.out_dim = out_dim

	def embed(self, inputs):
		return torch.cat([fn(inputs) for fn in self.embed_fns], -1)



class WithInput(nn.Module):
	def __init__(self, encoder, enc_dim, input_dim):
		super().__init__()
		self.encoder = encoder
		self.encoding_dim = enc_dim + input_dim
		self.progress = 0


	def forward(self, x):
		return torch.cat([self.encoder(x), x], dim=-1)


	def update(self, progress):
		self.progress = progress
		if hasattr(self.encoder, "update"):
			return self.encoder.update()
		else:
			return {}



class WithDefaultEncoder(nn.Module):
	def __init__(self, encoder, enc_dim, multires, input_dims, 
		detach_enc=False, dropout_ratio=0.0):
		super().__init__()
		self.input_dims = input_dims
		self.enc_dim = enc_dim
		self.encoder = encoder
		self.fourier, enc_dim_fourier = make_default_encoding(multires, input_dims)
		self.encoding_dim = enc_dim + enc_dim_fourier
		self.detach_enc = detach_enc
		self.dropout_ratio = dropout_ratio
		self.progress = 0


	def forward(self, x):
		embed = self.encoder(x.reshape(-1, self.input_dims)).reshape(*x.shape[:-1], self.enc_dim)
		if self.detach_enc:
			embed = embed.detach()
		if self.dropout_ratio > 0:
			embed = F.dropout(embed, p=self.dropout_ratio * (1 - self.progress), training=self.training)
		return torch.cat([self.fourier(x), embed], dim=-1)


	def update(self, progress):
		self.progress = progress
		if hasattr(self.encoder, "update"):
			return self.encoder.update()
		else:
			return {}



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


def make_hashgrid_encoding(multires, input_dims):
	hash_encoding, encoding_dim = make_hashgrid_encoding_base(multires, input_dims)
	encoding = WithInput(hash_encoding, encoding_dim, 3)
	return encoding, encoding.encoding_dim


def make_hybrid_encoding(multires, input_dims, dropout_ratio=0.0):
	hash_encoding, encoding_dim = make_hashgrid_encoding_base(multires, input_dims)
	encoding = WithDefaultEncoder(hash_encoding, encoding_dim, multires, input_dims, dropout_ratio=dropout_ratio)
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


def get_embedder(encoding, multires, input_dims=3, dropout_ratio=0.0):
	if encoding == "default":
		return make_default_encoding(multires, input_dims)

	elif encoding == "fourier":
		return make_fourier_encoding(multires, input_dims)

	elif encoding == "hashgrid":
		return make_hashgrid_encoding(multires, input_dims)

	elif encoding == "sh":
		return make_sh_encoding(multires, input_dims)

	elif encoding == "hybrid":
		return make_hybrid_encoding(multires, input_dims, dropout_ratio=dropout_ratio)

	elif encoding == "tensorf":
		return make_tensorf_encoding(multires, input_dims, dropout_ratio=dropout_ratio)

	elif encoding == "kplanes":
		return make_kplanes_encoding(multires, input_dims, dropout_ratio=dropout_ratio)

	else:
		raise NotImplementedError()
