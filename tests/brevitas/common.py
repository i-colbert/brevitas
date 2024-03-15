# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

RTOL = 0
ATOL = 1e-23

FP32_BIT_WIDTH = 32
MIN_INT_BIT_WIDTH = 2
MAX_INT_BIT_WIDTH = 8
INT_BIT_WIDTH_TO_TEST = range(MIN_INT_BIT_WIDTH, MAX_INT_BIT_WIDTH + 1)
BOOLS = [True, False]


def assert_allclose(generated, reference):
    assert torch.allclose(generated, reference, RTOL, ATOL)


def assert_zero_or_none(value):
    if isinstance(value, torch.Tensor):
        assert (value == torch.tensor(0.)).all()
    else:
        assert value is None


def get_weight_per_layer(state_dict):
    weight_per_layer = {}
    for key, value in state_dict.items():
        key_split = key.split('.')
        if 'weight' in key_split:
            weight_per_layer[key_split[0]] = value

    return weight_per_layer


def get_l1_norm_per_oc_layer(weight_per_layer):
    l1_per_oc_per_layer = {}
    for key, value in weight_per_layer.items():
        # flatten
        value = value.reshape(value.size(0), -1)
        l1_per_oc_per_layer[key] = torch.linalg.norm(value, ord=1, dim=1)  # 1 dim for OC

    return l1_per_oc_per_layer
