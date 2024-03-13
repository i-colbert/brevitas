# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm

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


# example quant split function
def quant_split_evenly(channel, scale, zero_point, module):
    return channel / 2., channel / 2., scale, scale, zero_point, zero_point


def compressibility_loss(inp: torch.Tensor, dim: int = 1) -> torch.Tensor:
    out = torch.norm(inp, dim=dim, p=1) / torch.norm(inp, dim=dim, p=2)
    return out


def quant_split_quant_error(channel, scale, zero_point, bias, module):
    bit_width = module.weight_quant.bit_width()
    int_threshold = module.weight_quant.tensor_quant.int_scaling_impl(bit_width)
    # TODO: insert assertion about the int_quant
    split_channel = channel - module.weight_quant.tensor_quant.int_quant(
        scale / int_threshold, zero_point, bit_width, channel)
    # leaving scales untouched and initializing bias 1:0
    device = bias.device if bias is not None else 'cpu'
    return channel.clone(), split_channel, scale, scale, zero_point, zero_point, bias, torch.tensor(0.0, device=device)


def quant_duplicate_input(channel, scale, zero_point, bias, module):
    # no need to return anything else
    return channel, channel, scale, scale, zero_point, zero_point, bias, bias


def _channels_to_split(
        sources: Dict[str, nn.Module],
        sinks: Dict[str, nn.Module],
        layer_split_perc_func: Callable,
        split_input: bool,
        split_criterion_func: Callable) -> Dict[nn.Module, List[torch.Tensor]]:
    """
    This method computes the channels that will be split based on `split_criterion`.
    """
    modules = sinks if split_input else sources
    _get_axis = _get_input_axis if split_input else _get_output_axis
    # the modules are all of the same shape so we can just take the first one
    single_module = next(iter(modules))
    num_channels = single_module.weight.shape[_get_axis(single_module)]
    split_perc = layer_split_perc_func(single_module)
    splits_per_layer = int(math.ceil(split_perc * num_channels))

    all_channels = []
    for module in modules:
        # get input/output axis of module
        axis = _get_axis(module)
        # transpose to have axis as first dimension
        weight_t = transpose(module.weight, axis)
        # flatten all but first dimension and get max per channel
        max_per_channel = split_criterion_func(weight_t.reshape(weight_t.size(0), -1))
        channels_sorted = torch.argsort(max_per_channel, descending=True)
        all_channels.append(channels_sorted[:splits_per_layer])

    # return tensor with the unique indices to split
    channels_to_split = torch.cat(all_channels)
    return torch.unique(channels_to_split)


# decorator is needed to modify the weights in-place using a view
@torch.no_grad()
def _split_quantized_channels(
        module: nn.Module, channels_to_split: torch.Tensor, split_input: bool,
        split_func: Callable) -> None:
    """
    Given a QuantModule, this method splits the weight channels and scales in case of per_channel
    quantization. It differs from _split_quantized_channels as the actual splitting of channels and scales
    might needs access to the quantization methods and parameters.
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

    # check for per_channel quantization
    is_per_channel_quant = False
    is_asym_quant = False
    # we can only split scales etc. for output channels, so check if we are splitting output
    if not split_input and module.weight_quant.scale().shape[axis] == orig_shape[axis]:
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
    for id in channels_to_split:
        # get channel to split
        channel_to_split = weight_t[id, :]
        # get scale for channel
        scale_to_split = scales_t[id] if is_per_channel_quant else module.weight_quant.scale()
        # get zero point
        zero_point_to_split = zero_points[id] if is_asym_quant else torch.tensor(0.)
        # get bias
        if bias is not None and not split_input:
            bias_to_split = bias[id]
        else:
            bias_to_split = None
        # split channel/scale/zero_point/bias based on user defined method, i.e. halfing it, duplicating it
        split_values = split_func(
            channel_to_split, scale_to_split, zero_point_to_split, bias_to_split, module)
        assert len(split_values) == 8, 'split method needs to return 8 values: 2x channel, 2x scale, 2x zero_point, 2x bias'
        # unpack all split_values
        split_channel_0, split_channel_1, split_scale_0, split_scale_1, zero_point_0, zero_point_1, split_bias_0, split_bias_1 = split_values
        # set original channel to split_channel_0 and then stack the second channel
        weight_t[id, :] = split_channel_0
        # expand so we can stack
        split_channel_1 = split_channel_1.expand(1, split_channel_1.size(0))
        weight_t = torch.cat([weight_t, split_channel_1], dim=0)

        # if per_channel quant, we need to create a new scale for the added channel
        if is_per_channel_quant:
            scales_t[id] = split_scale_0
            split_scale_1 = split_scale_1.unsqueeze(0)
            # stacking the newly created scale
            scales_t = torch.cat([scales_t, split_scale_1], dim=0)

        # zero points in case of asym
        if is_asym_quant:
            zero_points[id] = zero_point_0
            zero_point_1 = zero_point_1.unsqueeze(0)
            zero_points = torch.cat([zero_points, zero_point_1], dim=0)

        if bias is not None and not split_input:
            bias[id] = split_bias_0
            split_bias_1 = split_bias_1.unsqueeze(0)
            bias = torch.cat([bias, split_bias_1], dim=0)

    # reshape weight_t back to orig shape with the added channels
    del orig_shape[axis]
    weight_t = weight_t.reshape(weight_t.size(0), *orig_shape)
    weight_t = transpose(weight_t, axis)
    module.weight.data = weight_t.clone().contiguous()

    if bias is not None:
        module.bias.data = bias.clone().contiguous()

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
            # TODO: this is the wrong attribute to set, ask Giuseppe for the right place to set them to
            module.weight_quant.tensor_quant.scaling_impl.value.data = scales_t.clone().contiguous()

    if is_asym_quant:
        # set zero_points to module, reshape using scales shape as it's the same
        zero_points_t = zero_points_t.reshape(zero_points_t.size(0), *orig_scales_shape)
        zero_points_t = transpose(zero_points_t, axis)
        module.weight_quant.tensor_quant.zero_point_impl.value.data = zero_points_t.clone(
        ).contiguous()

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


# decorator is needed to modify the weights in-place using a view
@torch.no_grad()
def _split_unquantized_channels(
        module: nn.Module,
        channels_to_split: torch.Tensor,
        split_input: bool = False,
        split_factor: float = 0.5) -> None:
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
    for id in channels_to_split:
        # split and get channel to stack
        weight_t[id, :] *= split_factor
        split_channel = weight_t[id, :]
        # expand so we can stack
        split_channel = split_channel.expand(1, split_channel.size(0))
        weight_t = torch.cat([weight_t, split_channel], dim=0)

        if bias is not None and not split_input:
            bias[id] *= split_factor
            split_channel = bias[id:id + 1]
            bias = torch.cat((bias, split_channel))

    # reshape weight_t back to orig shape with the added channels
    del orig_shape[axis]
    weight_t = weight_t.reshape(weight_t.size(0), *orig_shape)
    weight_t = transpose(weight_t, axis)
    module.weight.data = weight_t
    if bias is not None:
        module.bias.data = bias

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


def _duplicate_channels(
        module, channels_to_split, duplicate_input, quant_duplicating_func=quant_duplicate_input):
    # wrapper to simply use duplicating functions
    if isinstance(module, QuantLayerMixin):
        # duplicate using duplicate func
        _split_quantized_channels(
            module,
            channels_to_split,
            split_input=duplicate_input,
            split_func=quant_duplicating_func)
    else:
        # duplicate the channels as before
        _split_unquantized_channels(
            module, channels_to_split, split_input=duplicate_input, split_factor=1)


def _split_channels(module, channels_to_split, split_input, quant_split_func):
    # wrapper for splitting channels in quant/unquant modules
    if isinstance(module, QuantLayerMixin):
        # split quantized channels using the specified splitting mechanism
        _split_quantized_channels(
            module, channels_to_split, split_input, split_func=quant_split_func)
    else:
        # split channels regularly
        _split_unquantized_channels(module, channels_to_split, split_input, split_factor=0.5)


def _split_channels_region(
        sources: Dict[str, nn.Module],
        sinks: Dict[str, nn.Module],
        channels_to_split: torch.tensor,
        split_input: bool,
        quant_split_func: Callable) -> None:
    if not split_input:
        # splitting output channels
        for module in sources:
            _split_channels(
                module, channels_to_split, split_input=False, quant_split_func=quant_split_func)
        for module in sinks:
            # duplicating input_channels for all modules in the sink
            _duplicate_channels(module, channels_to_split, duplicate_input=True)
    else:
        # input channels are split in half, output channels duplicated
        for module in sinks:
            _split_channels(
                module, channels_to_split, split_input=True, quant_split_func=quant_split_func)
        for module in sources:
            # duplicating output_channels for all modules in the source
            _duplicate_channels(module, channels_to_split, duplicate_input=False)


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
        layer_split_perc_func: Callable,
        split_input: bool,
        quant_split_func: Callable = quant_split_quant_error,
        split_criterion_func: Callable = _channel_maxabs) -> GraphModule:
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
            split_criterion_func=split_criterion_func,
            layer_split_perc_func=layer_split_perc_func,
            split_input=split_input)
        # splitting/duplicating channels
        _split_channels_region(
            sources=sources,
            sinks=sinks,
            channels_to_split=channels_to_split,
            split_input=split_input,
            quant_split_func=quant_split_func)

    return model


def _clean_regions(regions: List[Region], region_filter_func: Callable) -> List[Region]:
    """
    Given a list of regions, this method removes all regions that are not compatible with channel splitting.
    """
    # idea: map modules to their regions and check whether it appears in multiple regions
    regions_to_del = set()
    source_modules = dict()
    sink_modules = dict()
    for i, region in enumerate(tqdm(regions)):
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
        # check if user filters out this region
        if not region_filter_func(sources, sinks):
            # user doesn't want to split this region
            regions_to_del.add(i)

    regions = [regions[i] for i, _ in enumerate(regions) if i not in regions_to_del]
    return regions


class GraphChannelSplitting(GraphTransform):

    def __init__(
        self,
        split_input: bool = True,
        split_criterion_func: Callable = compressibility_loss,
        quant_split_func: Callable = quant_split_quant_error,
        layer_split_perc_func: Optional[Callable[[nn.Module], float]] = lambda x: 0.02,
        region_filter_func: Optional[Callable[[List[nn.Module], List[nn.Module]],
                                              bool]] = lambda sources,
        sinks: True):
        super(GraphChannelSplitting, self).__init__()

        self.split_input = split_input
        self.layer_split_perc_func = layer_split_perc_func
        self.split_criterion_func = split_criterion_func
        self.quant_split_func = quant_split_func
        self.region_filter_func = region_filter_func

    def apply(
            self,
            model: GraphModule,
            return_regions: bool = False
    ) -> Union[Tuple[GraphModule, Set[Tuple[str]]], GraphModule]:
        regions = _extract_regions(model)
        regions = _clean_regions(regions, region_filter_func=self.region_filter_func)
        if len(regions) > 0:
            model = _split(
                model=model,
                regions=regions,
                layer_split_perc_func=self.layer_split_perc_func,
                split_criterion_func=self.split_criterion_func,
                split_input=self.split_input,
                quant_split_func=self.quant_split_func)
        if return_regions:
            return model, regions
        else:
            return model
