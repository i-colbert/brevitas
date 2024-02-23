# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Dict, List, Set, Tuple, Union

import torch
import torch.nn as nn

from brevitas.fx import GraphModule
from brevitas.graph.base import GraphTransform
from brevitas.graph.equalize import _channel_maxabs
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _get_input_axis
from brevitas.graph.equalize import _get_output_axis
from brevitas.graph.equalize import Region
from brevitas.graph.equalize import transpose
from brevitas.nn.mixin.base import QuantLayerMixin

__all__ = ['GraphChannelSplitting']

_conv = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)

_unsupported_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)


def _channels_to_split(
        sources: Dict[str, nn.Module],
        sinks: Dict[str, nn.Module],
        split_criterion: str,
        split_ratio: float,
        split_input: bool) -> Dict[nn.Module, List[torch.Tensor]]:
    """
    This method computes the channels that will be split based on `split_criterion`.
    """
    modules = sinks if split_input else sources
    _get_axis = _get_input_axis if split_input else _get_output_axis
    # the modules are all of the same shape so we can just take the first one
    single_module = next(iter(modules))
    num_channels = single_module.weight.shape[_get_axis(single_module)]
    splits_per_layer = int(math.ceil(split_ratio * num_channels))

    all_channels = []
    if split_criterion == 'maxabs':
        for module in modules:
            # get input/output axis of module
            axis = _get_axis(module)
            # transpose to have axis as first dimension
            weight_t = transpose(module.weight, axis)
            # flatten all but first dimension and get max per channel
            max_per_channel = _channel_maxabs(weight_t.reshape(weight_t.size(0), -1))
            channels_sorted = torch.argsort(max_per_channel, descending=True)
            all_channels.append(channels_sorted[:splits_per_layer])

    # return tensor with the unique indices to split
    channels_to_split = torch.cat(all_channels)
    return torch.unique(channels_to_split)


# decorator is needed to modify the weights in-place using a view
@torch.no_grad()
def _split_channels(
        module: nn.Module,
        channels_to_split: torch.Tensor,
        split_input: bool = False,
        split_factor: float = 0.5,
        use_quant_error: bool = False) -> None:
    """
    Given a module, this method splits the weight channels as proposed in https://arxiv.org/abs/1901.09504.
    `split_factor` determines how to split the channels, `channels_to_split` is a list of channel indices.
    If `split_input=True`, the input channels of the module are split, otherwise the output channels.
    """
    weight = module.weight.data
    bias = module.bias.data if module.bias is not None else None
    num_added_channels = len(channels_to_split)

    _get_axis = _get_input_axis if split_input else _get_output_axis
    axis = _get_axis(module)
    # save shape of the module weights
    orig_shape = list(weight.shape)
    weight_t = transpose(weight, axis)
    # flatten to 2d
    weight_t = weight_t.reshape(weight_t.size(0), -1)

    # check for quantized layer and get scales if per channel quant
    is_per_channel_quant = False
    is_asym_quant = False
    if isinstance(module,
                  QuantLayerMixin) and module.weight_quant.scale().shape[axis] == orig_shape[axis]:
        is_per_channel_quant = True
        orig_scales_shape = module.weight_quant.tensor_quant.scaling_impl.value.shape
        # get view of scales
        scales = module.weight_quant.tensor_quant.scaling_impl.value.data.flatten()
        # get zero_points, for sym quantization the zero point is 1D
        try:
            if module.weight_quant.tensor_quant.zero_point_impl.value.shape == orig_scales_shape:
                is_asym_quant = True
                zero_points = module.weight_quant.tensor_quant.zero_point_impl.value.data.flatten()
        except AttributeError:
            # nothing to do with the zero points, that's 0 anyways
            print('sym quant')
    for id in channels_to_split:
        # just adding a new channel with the quantization error
        if isinstance(module, QuantLayerMixin) and use_quant_error:
            # adding a new channel with the quantization error
            # get channel to split and expand that channel for concatenating
            split_channel = weight_t[id, :]
            # get scale, zero_point, and bit width
            scale_for_split = scales[id]
            zero_point_for_split = zero_points[id] if is_asym_quant else 0.
            bit_width = module.weight_quant.bit_width()
            split_channel = split_channel - module.weight_quant.tensor_quant.int_quant(
                scale_for_split, zero_point_for_split, bit_width, split_channel)
        else:
            # split and get channel to stack
            weight_t[id, :] *= split_factor
            split_channel = weight_t[id, :]
        # expand so we can stack
        split_channel = split_channel.expand(1, split_channel.size(0))
        weight_t = torch.cat([weight_t, split_channel], dim=0)

        if bias is not None and not split_input:
            bias[id] *= split_factor
            split_bias = bias[id:id + 1]
            bias = torch.cat((bias, split_bias))

        if is_per_channel_quant:
            # TODO: add option to keep it or half it or set it to 0
            scales[id] *= split_factor
            split_scale = scales[id:id + 1]
            scales = torch.cat([scales, split_scale], dim=0)

        if is_asym_quant:
            # duplicate zero_point, no point of splitting it
            zero_point_channel = zero_points[id:id + 1]
            zero_points = torch.cat([zero_points, zero_point_channel], dim=0)

    # reshape weight_t back to orig shape with the added channels
    del orig_shape[axis]
    weight_t = weight_t.reshape(weight_t.size(0), *orig_shape)
    weight_t = transpose(weight_t, axis)
    module.weight.data = weight_t

    if bias is not None:
        module.bias.data = bias

    if is_per_channel_quant:
        # TODO: add option to initialize scale for the new channel differently
        # we don't care about the axis dim, we need all other dim for reshaping later
        orig_scales_shape = list(orig_scales_shape)
        del orig_scales_shape[axis]
        # reshape scale back to orig shape, with adapted dim for split scales
        scales = scales.reshape(-1, *orig_scales_shape)
        # set value for scaling_impl to scales
        module.weight_quant.tensor_quant.scaling_impl.value.data = scales

    if is_asym_quant:
        # set zero_points to module
        zero_points = zero_points.reshape(-1, *orig_scales_shape)  # same shape as scales
        module.weight_quant.tensor_quant.zero_point_impl.value.data = zero_points

    if isinstance(module, _conv):
        if split_input:
            module.in_channels += num_added_channels
        else:
            module.out_channels += num_added_channels
    elif isinstance(module, nn.Linear):
        if split_input:
            module.in_features += num_added_channels
        else:
            module.out_features += num_added_channels


def _split_channels_region(
        sources: Dict[str, nn.Module],
        sinks: Dict[str, nn.Module],
        channels_to_split: torch.tensor,
        split_input: bool,
        use_quant_error: bool) -> None:
    if not split_input:
        # splitting output channels
        for module in sources:
            _split_channels(
                module, channels_to_split, split_input=False, use_quant_error=use_quant_error)
        for module in sinks:
            # duplicating input_channels for all modules in the sink
            _split_channels(module, channels_to_split, split_factor=1, split_input=True)
    else:
        # input channels are split in half, output channels duplicated
        for module in sinks:
            _split_channels(
                module, channels_to_split, split_input=True, use_quant_error=use_quant_error)

        for module in sources:
            # duplicating output_channels for all modules in the source
            _split_channels(module, channels_to_split, split_factor=1, split_input=False)


def _is_groupwise(module: nn.Module) -> bool:
    # only Conv layers can be groupwise
    return isinstance(module, _conv) and module.groups > 1


def _is_unsupported(module: nn.Module) -> bool:
    return isinstance(module, _unsupported_layers)


def _is_mha(module: nn.Module) -> bool:
    return isinstance(module, nn.MultiheadAttention)


def _is_supported(srcs: List[nn.Module], sinks: List[nn.Module]) -> bool:
    # groupwise convolutions are not supported so filter them out
    if any(map(_is_groupwise, srcs + sinks)):
        return False

    # filter out unsupported layers
    if any(map(_is_unsupported, sinks + srcs)):
        return False

    # mha can only be in the sources
    if any(map(_is_mha, sinks)):
        return False
    elif any(map(_is_mha, srcs)):
        # we need to access the weights of the out_proj layers in mha, therefore unwrap
        srcs = _unwrap_mha(srcs)

    # check if OCs of sources are all equal
    srcs_ocs = set(module.weight.shape[_get_output_axis(module)] for module in srcs)
    if len(srcs_ocs) > 1:
        return False

    # check if ICs of sinks are all equal
    sinks_ics = set(module.weight.shape[_get_input_axis(module)] for module in sinks)
    if len(sinks_ics) > 1:
        return False

    return srcs_ocs == sinks_ics


def _unwrap_mha(sources: List[nn.Module]) -> List[nn.Module]:
    for i, source in enumerate(sources):
        if _is_mha(source):
            sources[i] = source.out_proj
    return sources


def _split(
        model: GraphModule,
        regions: List[Region],
        split_ratio: float,
        split_input: bool,
        split_criterion: str = 'maxabs',
        use_quant_error: bool = False) -> GraphModule:
    for i, region in enumerate(regions):
        sources = [region.get_module_from_name(src) for src in region.srcs_names]
        sinks = [region.get_module_from_name(sink) for sink in region.sinks_names]

        # check for mha in sources and unwrap it for out_proj
        if any(map(_is_mha, sources)):
            sources = _unwrap_mha(sources)

        # get channels to split
        channels_to_split = _channels_to_split(
            sources=sources,
            sinks=sinks,
            split_criterion=split_criterion,
            split_ratio=split_ratio,
            split_input=split_input)
        # splitting/duplicating channels
        _split_channels_region(
            sources=sources,
            sinks=sinks,
            channels_to_split=channels_to_split,
            split_input=split_input,
            use_quant_error=use_quant_error)

    return model


def _clean_regions(regions: List[Region]) -> List[Region]:
    """
    Given a list of regions, this method removes all regions that are not compatible with channel splitting.
    """
    # idea: map modules to their regions and check whether it appears in multiple regions
    regions_to_del = set()
    source_modules = dict()
    sink_modules = dict()
    for i, region in enumerate(regions):
        sources = [region.get_module_from_name(src) for src in region.srcs_names]
        sinks = [region.get_module_from_name(sink) for sink in region.sinks_names]

        # a module cannot be in the sources (or sinks) of multiple regions
        for src in sources:
            # if not yet in the dict, instantiate new list for keeping track
            if src not in source_modules:
                source_modules[src] = [i]
            else:
                # we know the module has been in sources before, so region needs to be deleted
                source_modules[src].append(i)
                regions_to_del.update({*source_modules[src]})
        for sink in sinks:
            if sink not in sink_modules:
                sink_modules[sink] = [i]
            else:
                sink_modules[sink].append(i)
                regions_to_del.update({*sink_modules[sink]})

        # check for other unsupported
        if not _is_supported(srcs=sources, sinks=sinks):
            # add region to be deleted
            regions_to_del.add(i)

    regions = [regions[i] for i, _ in enumerate(regions) if i not in regions_to_del]
    return regions


class GraphChannelSplitting(GraphTransform):

    def __init__(
            self,
            split_ratio: float = 0.02,
            split_criterion: str = 'maxabs',
            split_input: bool = True,
            use_quant_error: bool = False):
        super(GraphChannelSplitting, self).__init__()

        self.split_ratio = split_ratio
        self.split_criterion = split_criterion
        self.split_input = split_input
        self.use_quant_error = use_quant_error

    def apply(
            self,
            model: GraphModule,
            return_regions: bool = False
    ) -> Union[Tuple[GraphModule, Set[Tuple[str]]], GraphModule]:
        regions = _extract_regions(model)
        regions = _clean_regions(regions)
        if len(regions) > 0:
            model = _split(
                model=model,
                regions=regions,
                split_ratio=self.split_ratio,
                split_criterion=self.split_criterion,
                split_input=self.split_input,
                use_quant_error=self.use_quant_error)
        if return_regions:
            return model, regions
        else:
            return model
