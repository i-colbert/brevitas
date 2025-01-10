# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.core.scaling import AccumulatorAwareParameterPreScaling
from brevitas.function.shape import over_output_channels
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas_examples.common.ep_init import l1_proj_matrix_per_channel

__all__ = ["apply_bias_correction", "apply_ep_init"]


def get_a2q_module(module: nn.Module):
    for submod in module.modules():
        if isinstance(submod, AccumulatorAwareParameterPreScaling):
            return submod
    return None


def apply_ep_init(model: nn.Module, inp: Tensor):
    """Euclidean projection-based weight initialization (EP-init) for accumulator-aware
    quantization as proposed in `A2Q+: Improving Accumulator-Aware Weight Quantization`"""
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    module_stats = {}
    hook_list = list()

    def register_upper_bound(module: AccumulatorAwareParameterPreScaling, inp, output, name):
        """Accumulate the regularization penalty across constrained layers"""
        nonlocal module_stats

        (weights, input_bit_width, input_is_signed) = inp
        scales: Tensor = module.scaling_impl(weights)
        max_norm: Tensor = module.calc_max_l1_norm(input_bit_width, input_is_signed)

        shape = over_output_channels(weights)
        s = scales.reshape(shape)
        w = weights.reshape(shape)

        z: Tensor = s * max_norm  # radius
        module_stats[name] = (w.detach(), z.detach())  # no gradients

        restrict_value_impl = module.restrict_clamp_scaling.restrict_value_impl
        pre_scaling_init: Tensor = restrict_value_impl.restrict_init_tensor(scales * max_norm)
        assert pre_scaling_init.shape == module.value.shape, "Error: shape mismatch."
        module.value.data = torch.where(
            module.value.data <= pre_scaling_init, module.value.data, pre_scaling_init)

        return output

    # add hooks to each of the A2Q pre-scaling modules
    for name, mod in model.named_modules():
        if isinstance(mod, QuantWBIOL):
            submod = get_a2q_module(mod)
            if submod is not None:
                hook_fn = partial(register_upper_bound, name=name)
                hook = submod.register_forward_hook(hook_fn)
                hook_list.append(hook)

    inp = inp.to(device=device, dtype=dtype)
    model(inp)  # register the scaled upper bounds

    # project weights onto the l1-ball
    for name, mod in model.named_modules():
        if name in module_stats and isinstance(mod, (nn.Conv2d, nn.Linear)):
            (weights, radius) = module_stats[name]
            weights = l1_proj_matrix_per_channel(weights, radius)
            weights = weights.reshape(mod.weight.shape)
            mod.weight.data = weights

    for hook in hook_list:
        hook.remove()

    return model
