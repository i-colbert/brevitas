# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.function.ops import get_upper_bound_on_l1_norm
from brevitas.function.shape import over_output_channels
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
from brevitas_examples.common.ep_init import l1_proj_matrix_per_channel


@torch.no_grad()
def apply_ep_init(model, dataloader, max_accumulator_bit_width):
    """Euclidean projection-based weight initialization (EP-init) for accumulator-aware
    quantization as proposed in `A2Q+: Improving Accumulator-Aware Weight Quantization`"""
    model.eval()

    for mod in model.modules():
        if isinstance(mod, ActQuantProxyFromInjectorBase):
            mod.cache_inference_quant_act = True  # get input_quant metadata

    with torch.no_grad():
        model(**dataloader[0])

    module_stats = {}

    def get_module_stats(module: QuantWBIOL):

        weights: Tensor = module.weight.data
        scales: Tensor = module.weight_quant.scale()
        input_bit_width = module.input_quant.bit_width()
        input_is_signed = module.input_quant.is_signed

        max_norm: Tensor = get_upper_bound_on_l1_norm(
            accumulator_bit_width=max_accumulator_bit_width,
            input_bit_width=input_bit_width,
            input_is_signed=input_is_signed)

        shape = over_output_channels(weights)
        s = scales.reshape(shape)
        w = weights.reshape(shape)

        z: Tensor = s * max_norm  # radius
        return w, z

    # get module stats
    for name, mod in model.named_modules():
        if isinstance(mod, QuantWBIOL):
            module_stats[name] = get_module_stats(mod)

    # project weights onto the l1-ball
    for name, mod in model.named_modules():
        if name in module_stats and isinstance(mod, nn.Linear):
            (weights, radius) = module_stats[name]
            weights = l1_proj_matrix_per_channel(weights, radius)
            weights = weights.reshape(mod.weight.shape)
            mod.weight.data = weights

    return model
