# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import partial
import math
from operator import attrgetter
from typing import List, Optional, Set
import warnings

import torch
from torch.fx import GraphModule as TorchGraphModule

from brevitas.fx import GraphModule
from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.graph.calibrate import restore_return_quant_tensor
import brevitas.nn as qnn

SUPPORTED_CONV_OP = (
    qnn.QuantConv1d,
    qnn.QuantConv2d,
    qnn.QuantConv3d,
    qnn.QuantConvTranspose1d,
    qnn.QuantConvTranspose2d,
    qnn.QuantConvTranspose3d)


class StopFwdException(Exception):
    pass


@dataclass
class LayerHandler:
    layer_names: Set = field(default_factory=set)
    forward_count: int = 0


class gpxq_mode(ABC):
    """
    Apply GPxQ algorithm.

    Args:
        model (Module): The model to quantize with GPxQ
        group_of_parallel_layers (Optional, List[str]): .List of lists where each inner list is a group
            of layer names that can be optimized in parallel. Default: None
        inplace (bool): Wheter to apply GPFQ inplace or perform a deepcopy. Default: True
        create_weight_orig (bool): If True, store the original floating point weights before applying
            gpxq. These weights will be used anytime quantization is disabled. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPxQ. Default: False
        act_order (bool): Whether to order greedy path following by Hessian approximation. Default: False
        return_forward_output (bool): If True, returns the output of the forward pass. Otherwise the
            forward call inside the context manager returns None. Default: False

    Example:
        >>> with torch.no_grad():
        >>>     with gpxq_mode(model) as gpxq:
        >>>         gpxq_mode = gpxq.model
        >>>         for i in tqdm(range(gpxq.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 gpxq_mode(img)
        >>>             gpxq.update()
    """

    def __init__(
            self,
            model,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            act_order: bool = False,
            return_forward_output: bool = False,
            use_random_proj: bool = False,
            use_random_sampling: bool = False,
            target_dim: int = 4096) -> None:

        if not inplace:
            model = deepcopy(model)
        self.model = model
        self.create_weight_orig = create_weight_orig
        self.use_quant_activations = use_quant_activations
        self.hook_dict = dict()
        self.gpxq_layers = dict()
        # reference for each layer to update
        self.current_layer = LayerHandler()
        # How many layer to optimize
        self.num_layers = 0
        # Quantize following magnitude of activation
        self.act_order = act_order
        # How many subblock to use during GPTQ for each layer

        self.disable_quant_inference = DisableEnableQuantization()
        self.return_quant_tensor_state = dict()

        self.group_of_parallel_layers = group_of_parallel_layers
        self.return_forward_output = return_forward_output

        # random proj/sample and target_dim
        self.use_random_proj = use_random_proj
        self.use_random_sampling = use_random_sampling
        self.target_dim = target_dim

    def _is_module_supported(self, module):
        if isinstance(module, SUPPORTED_CONV_OP):
            return True
        elif isinstance(module, qnn.QuantLinear):
            return True
        else:
            return False

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback):
        pass

    def setup_gpxq_layers(self):
        # The user can specify on which layers to apply gptq in parallel.
        # All the others will be executed sequentially
        self.dict_of_layers = {
            name: [(name, module)] for name,
            module in self.model.named_modules() if self._is_module_supported(module)}
        if self.group_of_parallel_layers is not None:
            for parallel_layers in self.group_of_parallel_layers:
                for name in parallel_layers:
                    if name not in self.dict_of_layers:
                        raise ValueError(
                            "The layer {} is not present in the model or it is not supported for GPTQ"
                            .format(name))
                    del self.dict_of_layers[name]
                names = '_'.join(parallel_layers)
                self.dict_of_layers[names] = [
                    (name, attrgetter(name)(self.model)) for name in parallel_layers]

        # Print warning if hooks are attached to any module, since the normal forward flow of the
        # network is highly disrupted during GPxQ
        for _, parallel_layers in self.dict_of_layers.items():
            for name, module in parallel_layers:
                if len(module._forward_hooks) > 0 or len(module._forward_pre_hooks):
                    warnings.warn(
                        f'Hooks detected during setup for GPxQ. '
                        f'Behaviour might deviate from what expected.')

                # initialize GPxQ
                if self._is_module_supported(module):
                    gpxq_module_optimizer = self.initialize_module_optimizer(
                        module,
                        name,
                        act_order=self.act_order,
                        len_parallel_layers=len(parallel_layers),
                        create_weight_orig=self.create_weight_orig)
                    self.gpxq_layers[name] = gpxq_module_optimizer

    def setup_gpxq_hooks(self):
        for name, module in self.gpxq_layers.items():
            # Attach hooks for GPxQ
            hook_fn = partial(module.update_batch, current_layer=self.current_layer)
            self.hook_dict[name] = module.layer.register_forward_pre_hook(hook_fn)

        if not self.use_quant_activations:
            self.return_quant_tensor_state = disable_return_quant_tensor(self.model)
            self.disable_quant_inference.disable_act_quantization(
                self.model, is_training=self.model.training)
            self.disable_quant_inference.disable_bias_quantization(
                self.model, is_training=self.model.training)

        self.num_layers = len(self.dict_of_layers)
        return self

    def exit(self):
        if isinstance(self.model, (GraphModule, TorchGraphModule)):
            self.model.__class__.forward = self.orig_forward
        else:
            self.model.forward = self.orig_forward

        if not self.use_quant_activations:
            self.disable_quant_inference.enable_act_quantization(
                self.model, is_training=self.model.training)
            self.disable_quant_inference.enable_bias_quantization(
                self.model, is_training=self.model.training)
            restore_return_quant_tensor(self.model, self.return_quant_tensor_state)

    def update(self):
        for name in self.current_layer.layer_names:
            self.gpxq_layers[name].single_layer_update()
            self.hook_dict[name].remove()
        self.current_layer.layer_names.clear()

    @abstractmethod
    def catch_stopfwd(self, *args, **kwargs):
        pass


class GPxQ(ABC):

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers=1,
            create_weight_orig=True,
            use_random_proj=False,
            use_random_sampling=False,
            target_dim=4096) -> None:
        self.layer = layer
        self.name = name
        self.act_order = act_order

        weight = layer.weight.data

        if create_weight_orig and not hasattr(self.layer, 'weight_orig'):
            self.layer.register_buffer('weight_orig', layer.weight.detach().clone())

        # By default, use groups = 1
        self.groups = 1
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(
                    self.layer,
                (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d, qnn.QuantConvTranspose3d)):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
            self.groups = self.layer.groups

        # Number of rows is equal to the output channels (OC)
        self.rows = weight.shape[0]
        # Number of columns is equal to the input channels (IC)
        self.columns = weight.shape[1]
        self.len_parallel_layers = len_parallel_layers

        self.disable_pre_forward_hook = False
        # Some layers require knowledge from quant inputs to compute quant weights
        self.quant_input = None

        self.use_random_proj = use_random_proj
        self.use_random_sampling = use_random_sampling
        self.target_dim = target_dim

    def process_input(self, inp):
        # Input is a tuple, so we take first element
        inp = inp[0]

        # If input is unbatched, add batch_size = 1
        if len(inp.shape) == 1:
            warnings.warn("Found unbatched input, adding batch dimension equal to 1")
            inp = inp.unsqueeze(0)

        # Define batch size before re-organizing the input
        if hasattr(inp, 'names') and 'N' in inp.names:
            batch_dim = inp.names.index('N')
            inp.rename_(None)
            inp = inp.transpose(0, batch_dim)

        inp = self.layer.input_quant(inp)
        return inp

    @abstractmethod
    def update_batch(self):
        pass

    @abstractmethod
    def single_layer_update(self):
        pass

    def get_quant_weights(self, i, i1, permutation_list):
        # We need to recompute quant weights at runtime since our float weights are being updated
        # Add offset in case of blockwise computation
        i = i1 + i
        # For QuantLinear and for some QuantConvolutional layers, we exploit the possibility
        # of quantizing only a subset of the entire matrix speeding up the computation of GPxQ
        if isinstance(self.layer, qnn.QuantLinear):
            index = permutation_list[0][i]
            subtensor_slice_list = [None, (index, index + 1)]
            q = self.layer.quant_weight(
                subtensor_slice_list=subtensor_slice_list,
                quant_input=self.quant_input).value.unsqueeze(0)  # [1, OC, 1]
        elif isinstance(self.layer, SUPPORTED_CONV_OP):
            # For depthwise and ConvTranspose we fall back to quantizing the entire martix.
            # For all other cases, we create a mask that represent the slicing we will perform on the weight matrix
            # and we quantize only the selected dimensions.
            if self.groups > 1 or (self.groups == 1 and isinstance(
                    self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d))):

                quant_weight = self.layer.quant_weight(quant_input=self.quant_input)
                quant_weight = quant_weight.value

                if isinstance(self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)):
                    quant_weight = quant_weight.transpose(1, 0)  # This performs a view
                quant_weight = quant_weight.flatten(1)
                quant_weight = quant_weight.view(self.groups, -1, quant_weight.shape[-1])

                if self.act_order:
                    for ii, perm in enumerate(permutation_list):
                        quant_weight[ii, :, :] = quant_weight[ii, :, perm]

                q = quant_weight[:, :, i:i + 1]  # [groups, OC/groups, 1]
            else:
                index = permutation_list[0][i]
                shapes = self.layer.weight.shape[1:]
                index_2d_to_nd = []
                residual_index = index.item()
                for shape in shapes[::-1]:
                    index_2d_to_nd.append((residual_index % shape, residual_index % shape + 1))
                    residual_index = residual_index // shape
                index_2d_to_nd = index_2d_to_nd[::-1]
                index_2d_to_nd.insert(0, None)
                q = self.layer.quant_weight(
                    subtensor_slice_list=index_2d_to_nd,
                    quant_input=self.quant_input).value.flatten(1)  # [OC, 1]
                q = q.unsqueeze(0)  # [1, OC, 1]
        # We need to remove the last dim
        q = q.squeeze(2)  # [groups, OC/groups] or [1, OC]
        return q


def random_projection(
        float_input: torch.Tensor,
        quantized_input: torch.Tensor,
        target_dim: int,
        batch_size: int = 2048):
    # use random projection to reduce dimensionality
    n = quantized_input.size(1)
    dev = float_input.device
    # use batching if target_dim is greater than 8000 to avoid memory issues
    if target_dim > 8000:
        accumulated_batches = 0
        first_batch = True
        quantized_input = torch.transpose(quantized_input, 1, 2)
        float_input = torch.transpose(float_input, 1, 2)
        while accumulated_batches < target_dim:
            # cur_target_dim makes sure to fully use batch_size unless we're too close to target_dim
            cur_target_dim = min(batch_size, target_dim - accumulated_batches)
            accumulated_batches += cur_target_dim
            R = torch.normal(mean=0.0, std=1. / math.sqrt(n), size=(cur_target_dim, n), device=dev)
            if first_batch:
                quantized_input_proj = (quantized_input @ R.T).cpu()
                float_input_proj = (float_input @ R.T).cpu()
                first_batch = False
            else:
                # concatenate projected input along last dimension
                quantized_input_proj = torch.cat(
                    [quantized_input_proj, (quantized_input @ R.T).cpu()], dim=-1)
                float_input_proj = torch.cat([float_input_proj, (float_input @ R.T).cpu()], dim=-1)
    else:
        # create gaussian random matrix
        R = torch.normal(mean=0.0, std=1. / math.sqrt(n), size=(target_dim, n), device=dev)
        quantized_input_proj = torch.transpose(quantized_input, 1, 2) @ R.T
        float_input_proj = torch.transpose(float_input, 1, 2) @ R.T
        del R
    # reshape back
    quantized_input_proj = torch.transpose(quantized_input_proj, 1, 2)
    float_input_proj = torch.transpose(float_input_proj, 1, 2)
    del quantized_input
    del float_input

    return float_input_proj, quantized_input_proj
