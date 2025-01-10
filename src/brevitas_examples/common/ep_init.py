# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from torch import Tensor


def _euclidean_projection_onto_positive_simplex(vec: Tensor, radius: float = 1.):
    assert radius > 0, "Error: radius needs to be strictly positive."
    assert vec.ndim == 1, "Error: projection assumes a vector, not a matrix."
    assert vec.min() >= 0, "Error: assuming a vector of non-negative numbers."
    n_elems = vec.shape[0]
    # if we are already within the simplex, then the best projection is itself
    if vec.sum() <= radius:
        return vec
    # using algorithm derived in `Efficient Projections onto the L1-Ball for
    # Learning in High Dimensions`
    v = vec.cpu().detach().numpy()
    u = np.sort(v)[::-1]
    cumsum_u = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n_elems + 1) > (cumsum_u - radius))[0][-1]
    theta = float(cumsum_u[rho] - radius) / (rho + 1)
    w = np.clip(v - theta, 0, np.inf)
    vec.data = torch.tensor(w, dtype=vec.dtype, device=vec.device)
    return vec


def euclidean_projection_onto_l1_ball(vec: Tensor, radius: float):
    assert radius > 0, "Error: radius needs to be strictly positive."
    assert vec.ndim == 1, "Error: projection assumes a vector, not a matrix."
    vec_dir = vec.sign()
    vec_mag = _euclidean_projection_onto_positive_simplex(vec.abs(), radius)
    new_vec = vec_dir * vec_mag
    assert vec.shape == new_vec.shape, "Error: shape changed."
    return new_vec


def l1_proj_matrix_per_channel(weights: Tensor, radius: Tensor):
    assert isinstance(weights, Tensor), "Error: weights is assumed to be a Tensor."
    assert isinstance(radius, Tensor), "Error: radius is assumed to be a Tensor."
    assert weights.ndim == 2, "Error: assuming a matrix with ndim=2."
    # if defined per-tensor
    if radius.ndim == 0:
        radius = torch.ones(weights.shape[0]) * radius
    # if defined per-channel
    else:
        radius = radius.flatten()
        assert radius.nelement() == weights.shape[0], "Error: shape mismatch."
    # project each channel independently
    for i in range(weights.shape[0]):
        w = weights[i]
        z = radius[i].item()
        v = euclidean_projection_onto_l1_ball(w, z)
        weights[i] = v
    return weights
