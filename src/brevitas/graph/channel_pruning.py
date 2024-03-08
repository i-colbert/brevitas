# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Set, Tuple, Union

import torch
import torch.nn as nn

from brevitas.fx import GraphModule
from brevitas.graph.base import GraphTransform
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _get_input_axis
from brevitas.graph.equalize import _get_output_axis
from brevitas.graph.equalize import Region
from brevitas.graph.equalize import transpose
from brevitas.nn.mixin.base import QuantLayerMixin

from .channel_splitting import _clean_regions
from .channel_splitting import _conv
from .channel_splitting import _is_mha
from .channel_splitting import _unwrap_mha

__all__ = ['GraphChannelPruning']


# decorator is needed to modify the weights in-place using a view
@torch.no_grad()
def _delete_channels(module: nn.Module, channels_to_keep: torch.Tensor, del_input: bool) -> None:
    """
    Given a QuantModule, this method splits the weight channels and scales in case of per_channel
    quantization. It differs from _split_quantized_channels as the actual splitting of channels and scales
    might needs access to the quantization methods and parameters.
    """
    weight = module.weight.data
    bias = module.bias.data if module.bias is not None else None

    _get_axis = _get_input_axis if del_input else _get_output_axis
    axis = _get_axis(module)
    # save shape of the module weights
    orig_shape = list(weight.shape)
    weight_t = transpose(weight, axis)
    # flatten to 2d
    weight_t = weight_t.reshape(weight_t.size(0), -1)

    # check for per_channel quantization
    is_per_channel_quant = False
    is_asym_quant = False
    # check if quant layer and reshape scales and zero_points
    if isinstance(module, QuantLayerMixin) and not del_input and module.weight_quant.scale(
    ).shape[axis] == orig_shape[axis]:
        is_per_channel_quant = True
        # if scales are in the log domain, then arithmetic manipulations need to take that into account
        scales = module.weight_quant.tensor_quant.scaling_impl(weight_t)
        orig_scales_shape = scales.shape
        # get view of scales
        scales_t = transpose(scales, axis)
        # flatten to 2d
        scales_t = scales_t.reshape(scales_t.size(0), -1)
        try:
            # get zero_points, for sym quantization the zero point is 1D
            if module.weight_quant.tensor_quant.zero_point_impl.value.shape == orig_scales_shape:
                is_asym_quant = True
                zero_points = module.weight_quant.tensor_quant.zero_point_impl.value.data
                # get view of scales
                zero_points_t = transpose(zero_points, axis)
                # flatten to 2d
                zero_points_t = zero_points_t.reshape(zero_points_t.size(0), -1)
        except AttributeError:
            # nothing to do with the zero points, that's 0 anyways
            pass

    # delete channels, i.e. keep only the channels to keep
    weight_t = weight_t.index_select(0, channels_to_keep)
    if is_per_channel_quant:
        scales_t = scales_t.index_select(0, channels_to_keep)
    if is_asym_quant:
        zero_points_t = zero_points_t.index_select(0, channels_to_keep)

    if bias is not None and not del_input:
        bias = bias.index_select(0, channels_to_keep)

    # reshape weight_t back to orig shape with the added channels
    del orig_shape[axis]
    weight_t = weight_t.reshape(weight_t.size(0), *orig_shape)
    weight_t = transpose(weight_t, axis)
    module.weight.data = weight_t.clone().contiguous()

    if bias is not None:
        module.bias.data = bias

    if is_per_channel_quant:
        # we don't care about the axis dim, we need all other dim for reshaping later
        orig_scales_shape = list(orig_scales_shape)
        del orig_scales_shape[axis]
        # reshape scale back to orig shape, with adapted dim for split scales
        scales_t = scales_t.reshape(scales_t.size(0), *orig_scales_shape)
        scales_t = transpose(scales_t, axis)
        # set value for scaling_impl to scales
        scaling_impl = module.weight_quant.tensor_quant.scaling_impl
        try:
            scales_t = scaling_impl.restrict_clamp_scaling.restrict_value_impl.restrict_init_tensor(
                scales_t)
        except AttributeError:
            # no restrict_clamp_scaling, so pass
            pass
        finally:
            module.weight_quant.tensor_quant.scaling_impl.value.data = scales_t

    if is_asym_quant:
        # set zero_points to module, reshape using scales shape as it's the same
        zero_points_t = zero_points_t.reshape(zero_points_t.size(0), *orig_scales_shape)
        zero_points_t = transpose(zero_points_t, axis)
        module.weight_quant.tensor_quant.zero_point_impl.value.data = zero_points_t

    if isinstance(module, _conv):
        if del_input:
            module.in_channels = len(channels_to_keep)
        else:
            module.out_channels = len(channels_to_keep)
    elif isinstance(module, nn.Linear):
        if del_input:
            module.in_features = len(channels_to_keep)
        else:
            module.out_features = len(channels_to_keep)


def _get_channels_to_keep(sources: List[nn.Module]) -> torch.Tensor:
    all_channels_to_keep = torch.empty(0)
    # first, go through all sources and get channels to keep
    for source in sources:
        # reshape weight
        weight = source.weight.data
        # we're always looking at the output channels as we're only checking OCs for zero channels
        axis = _get_output_axis(source)
        weight_t = transpose(weight, axis)
        # flatten to 2d
        weight_t = weight_t.reshape(weight_t.size(0), -1)

        # get channels to keep, basically summing across output channels and see which ones are not 0
        # this is easier than looking at all values and checking if they're 0, if they sum up to 0,
        # they all have to be 0 anyways
        channels_to_keep = torch.nonzero(weight_t.abs().sum(dim=1) != 0).flatten()

        # union with other channels to keep
        all_channels_to_keep = torch.cat([all_channels_to_keep, channels_to_keep]).unique()

    return all_channels_to_keep.to(torch.int)


def _prune_channels(
        sources: List[nn.Module], sinks: List[nn.Module], channels_to_keep: torch.Tensor):
    # delete output channels in sources
    for source in sources:
        # do preprocessing and get all weights in shape [OC, everything else]
        _delete_channels(source, channels_to_keep, del_input=False)

    # delete input channels for sinks
    for sink in sinks:
        _delete_channels(sink, channels_to_keep, del_input=True)


def _prune(model: GraphModule, regions: List[Region]) -> GraphModule:
    for i, region in enumerate(regions):
        sources = [region.get_module_from_name(src) for src in region.srcs_names]
        sinks = [region.get_module_from_name(sink) for sink in region.sinks_names]

        # check for mha in sources and unwrap it for out_proj
        if any(map(_is_mha, sources)):
            sources = _unwrap_mha(sources)

        # only look at the OCs of the sources for 0 channels, as we are pruning 0 channels in the source,
        # and then delete the according input channel in the sink
        channels_to_keep = _get_channels_to_keep(sources)
        _prune_channels(sources, sinks, channels_to_keep)

    return model


class GraphChannelPruning(GraphTransform):

    def __init__(self):
        super(GraphChannelPruning, self).__init__()

    def apply(
            self,
            model: GraphModule,
            return_regions: bool = False
    ) -> Union[Tuple[GraphModule, Set[Tuple[str]]], GraphModule]:
        regions = _extract_regions(model)
        regions = _clean_regions(regions)
        if len(regions) > 0:
            model = _prune(model=model, regions=regions)
        if return_regions:
            return model, regions
        else:
            return model
