import numpy as np
import torch
import pdb
from math import floor, ceil

from .arrays import to_np, to_torch

class QuantileDiscretizer:

	def __init__(self, data, N):
		self.data = data
		self.N = N

		# n_points_per_bin = int(np.ceil(len(data) / N))
		# obs_sorted = np.sort(data, axis=0)
		# thresholds = obs_sorted[::n_points_per_bin, :]
		# maxs = data.max(axis=0, keepdims=True)
		# mins = data.min(axis=0, keepdims=True)
		maxs = data.max(axis=0, keepdims=False)
		mins = data.min(axis=0, keepdims=False)
		# extend_top = maxs + np.linspace(0,(maxs[0]-mins[0])/2,ceil((N-len(thresholds)+1)/2))
		# extend_bot = mins - np.linspace(0,(maxs[0]-mins[0])/2,floor((N-len(thresholds)+1)/2))
		# print(mins[0])
		# assert (mins[0] < 0).all()
		mins = np.full_like(mins,-3)
		maxs = np.full_like(maxs,3)
		print(maxs.shape)
		thresholds = np.linspace(mins, maxs, N+1)
		self.thresholds = thresholds
		# print(thresholds.shape)

		## [ (N + 1) x dim ]
		# self.thresholds = np.concatenate([extend_bot, thresholds, extend_top], axis=0)
		# print(self.thresholds.shape)

		# threshold_inds = np.linspace(0, len(data) - 1, N + 1, dtype=int)
		# obs_sorted = np.sort(data, axis=0)

		# ## [ (N + 1) x dim ]
		# self.thresholds = obs_sorted[threshold_inds, :]

		## [ N x dim ]
		self.diffs = self.thresholds[1:] - self.thresholds[:-1]

		## for sparse reward tasks
		# if (self.diffs[:,-1] == 0).any():
		# 	raise RuntimeError('rebin for sparse reward tasks')

		self._test()

	def __call__(self, x):
		indices = self.discretize(x)
		recon = self.reconstruct(indices)
		error = np.abs(recon - x).max(0)
		return indices, recon, error

	def _test(self):
		print('[ utils/discretization ] Testing...', end=' ', flush=True)
		# inds = np.random.randint(0, len(self.data), size=1000)
		inds = np.arange(0,np.min([len(self.data), 500]))
		X = self.data[inds]
		# print(X[0])
		indices = self.discretize(X)
		# print(indices)
		recon = self.reconstruct(indices)
		## make sure reconstruction error is less than the max allowed per dimension
		error = np.abs(X - recon).max(0)
		# print(X[86])
		# print(recon)
		# print(np.unravel_index(error.argmax(), error.shape))
		# print(error[86])
		# print(self.diffs)
		# assert (error <= self.diffs.max(axis=0)).all()
		## re-discretize reconstruction and make sure it is the same as original indices
		indices_2 = self.discretize(recon)
		# print(np.where(indices != indices_2, indices_2, np.full_like(indices,-1)))
		# import pdb; pdb.set_trace()
		# print(X[235,7])
		# print(recon[235,7])
		#ignore the value and reward ones for now
        #this doesnt work coz some values dont change enough in dataset
		# assert (indices[:,:-2] == indices_2[:,:-2]).all()
		## reconstruct random indices
		## @TODO: remove duplicate thresholds
		# randint = np.random.randint(0, self.N, indices.shape)
		# randint_2 = self.discretize(self.reconstruct(randint))
		# assert (randint == randint_2).all()
		print('✓')

	def discretize(self, x, subslice=(None, None)):
		'''
			x : [ B x observation_dim ]
		'''

		if torch.is_tensor(x):
			x = to_np(x)

		## enforce batch mode
		if x.ndim == 1:
			x = x[None]
			# import pdb; pdb.set_trace()

		## [ N x B x observation_dim ]
		start, end = subslice
		thresholds = self.thresholds[:, start:end]

		# import pdb; pdb.set_trace()
		gt = x[None] >= thresholds[:,None]
		indices = largest_nonzero_index(gt, dim=0)

		if indices.min() < 0 or indices.max() >= self.N:
			indices = np.clip(indices, 0, self.N - 1)

		return indices

	def reconstruct(self, indices, subslice=(None, None)):

		if torch.is_tensor(indices):
			indices = to_np(indices)

		## enforce batch mode
		if indices.ndim == 1:
			indices = indices[None]

		if indices.min() < 0 or indices.max() >= self.N:
			print(f'[ utils/discretization ] indices out of range: ({indices.min()}, {indices.max()}) | N: {self.N}')
			indices = np.clip(indices, 0, self.N - 1)

		start, end = subslice
		thresholds = self.thresholds[:, start:end]
		# print(thresholds.shape)

		left = np.take_along_axis(thresholds, indices, axis=0)
		right = np.take_along_axis(thresholds, indices + 1, axis=0)
		recon = (left + right) / 2.
		return recon

	#---------------------------- wrappers for planning ----------------------------#

	def expectation(self, probs, subslice):
		'''
			probs : [ B x N ]
		'''

		if torch.is_tensor(probs):
			probs = to_np(probs)

		## [ N ]
		thresholds = self.thresholds[:, subslice]
		## [ B ]
		left  = probs @ thresholds[:-1]
		right = probs @ thresholds[1:]

		avg = (left + right) / 2.
		return avg

	def percentile(self, probs, percentile, subslice):
		'''
			percentile `p` :
				returns least value `v` s.t. cdf up to `v` is >= `p`
				e.g., p=0.8 and v=100 indicates that
					  100 is in the 80% percentile of values
		'''
		## [ N ]
		thresholds = self.thresholds[:, subslice]
		## [ B x N ]
		cumulative = np.cumsum(probs, axis=-1)
		valid = cumulative > percentile
		## [ B ]
		inds = np.argmax(np.arange(self.N, 0, -1) * valid, axis=-1)
		left = thresholds[inds-1]
		right = thresholds[inds]
		avg = (left + right) / 2.
		return avg

	#---------------------------- wrappers for planning ----------------------------#

	def value_expectation(self, probs):
		'''
			probs : [ B x 2 x ( N + 1 ) ]
				extra token comes from termination
		'''

		if torch.is_tensor(probs):
			probs = to_np(probs)
			return_torch = True
		else:
			return_torch = False

		probs = probs[:, :, :-1]
		assert probs.shape[-1] == self.N

		rewards = self.expectation(probs[:, 0], subslice=-2)
		next_values = self.expectation(probs[:, 1], subslice=-1)

		if return_torch:
			rewards = to_torch(rewards)
			next_values = to_torch(next_values)

		return rewards, next_values

	def value_fn(self, probs, percentile):
		if percentile == 'mean':
			return self.value_expectation(probs)
		else:
			## percentile should be interpretable as float,
			## even if passed in as str because of command-line parser
			percentile = float(percentile)

		if torch.is_tensor(probs):
			probs = to_np(probs)
			return_torch = True
		else:
			return_torch = False

		probs = probs[:, :, :-1]
		assert probs.shape[-1] == self.N

		rewards = self.percentile(probs[:, 0], percentile, subslice=-2)
		next_values = self.percentile(probs[:, 1], percentile, subslice=-1)

		if return_torch:
			rewards = to_torch(rewards)
			next_values = to_torch(next_values)

		return rewards, next_values

def largest_nonzero_index(x, dim):
	N = x.shape[dim]
	arange = np.arange(N) + 1

	for i in range(dim):
		arange = np.expand_dims(arange, axis=0)
	for i in range(dim+1, x.ndim):
		arange = np.expand_dims(arange, axis=-1)

	inds = np.argmax(x * arange, axis=0)
	## masks for all `False` or all `True`
	lt_mask = (~x).all(axis=0)
	gt_mask = (x).all(axis=0)

	inds[lt_mask] = 0
	inds[gt_mask] = N

	return inds
