# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
import math
from typing import Callable, List, Optional
import warnings

from packaging import version
import torch
import torch.nn as nn

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError
from torch.fx import GraphModule as TorchGraphModule
import unfoldNd

from brevitas import torch_version
from brevitas.function import get_upper_bound_on_l1_norm
from brevitas.fx import GraphModule
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import StopFwdException
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
import brevitas.nn as qnn
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor


class GPTQ(GPxQ):
    """
    Adapted from https://github.com/IST-DASLab/gptq, released under the following LICENSE:

    Copyright 2023 IST-DASLab

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            num_blocks,
            use_random_proj=False,
            use_random_sampling=False,
            target_dim=10000) -> None:
        super().__init__(
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            use_random_proj,
            use_random_sampling,
            target_dim)

        # Define how many columns to update in each mini-block
        self.blocksize = math.ceil(self.columns / num_blocks)

        # Initialize Hessian matrix and counter. We need it in float32 to compute the inverse
        self.H = torch.zeros((self.groups, self.columns, self.columns),
                             device='cpu',
                             dtype=torch.float32)
        self.nsamples = 0

        assert torch_version >= version.parse('1.10'), "GPTQ requires torch 1.10 or higher"

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input

        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        inp = self.process_input(input)
        inp = _unpack_quant_tensor(inp)
        batch_size = inp.shape[0]

        # Preprocess the input to compute the Hessian
        if isinstance(self.layer, qnn.QuantLinear):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, sum(inp.shape[2:])))
            inp = inp.t()
            # For QuantLinear layer, groups will be 1
            inp_processed = inp.unsqueeze(0)

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            # Pick the correct unfoldNd class
            if isinstance(
                    self.layer,
                (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d, qnn.QuantConvTranspose3d)):
                unfold_impl = unfoldNd.UnfoldTransposeNd
            else:
                unfold_impl = unfoldNd.UnfoldNd

            unfold = unfold_impl(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride)

            # Split input based on how many groups in convolution
            inp_by_group = torch.chunk(inp, self.groups, 1)
            inp_processed = []
            # Preprocess input by group
            for i, inp in enumerate(inp_by_group):
                inp = unfold(inp)
                inp = inp.transpose(1, 0)
                inp = inp.flatten(1)
                inp_processed.append(inp)
            inp_processed = torch.stack(inp_processed)

        dev = inp_processed.device
        n = inp_processed.shape[-1]
        if self.use_random_proj:
            # use batching if target_dim is greater than 8000 to avoid memory issues
            if self.target_dim > 8000:
                batch_size_proj = 4096
                accumulated_batches = 0
                first_batch = True
                while accumulated_batches < self.target_dim:
                    # cur_target_dim makes sure to fully use batch_size unless we're too close to target_dim
                    cur_target_dim = min(batch_size_proj, self.target_dim - accumulated_batches)
                    accumulated_batches += cur_target_dim
                    R = torch.normal(
                        mean=0.0, std=1. / math.sqrt(n), size=(n, cur_target_dim), device=dev)
                    if first_batch:
                        inp_processed_proj = inp_processed @ R
                        first_batch = False
                    else:
                        # concatenate projected input along last dimension
                        inp_processed_proj = torch.cat([inp_processed_proj, (inp_processed @ R)],
                                                       dim=-1)
                # finally setting inp_processed to projected one, del proj afterwards
                inp_processed = inp_processed_proj
                del inp_processed_proj
            else:
                R = torch.normal(mean=0.0, std=1. / math.sqrt(n), size=(n, self.target_dim))
                # projecting the input data
                inp_processed = inp_processed @ R.to(inp_processed.device)
            del R
        elif self.use_random_sampling:
            # choose random indices and take TARGET_DIM many
            ind = torch.randint(n, (self.target_dim,))
            inp_processed = inp_processed.index_select(-1, ind.to(dev))
            del ind

        # Hessian computation
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp_processed = math.sqrt(2 / self.nsamples) * inp_processed.to(torch.float32)
        self.H += (inp_processed.bmm(inp_processed.transpose(2, 1))).to(self.H.device)
        # If we are executing GPTQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            raise StopFwdException

    def single_layer_update(self, percdamp=.01):
        assert not self.layer.weight_quant.requires_quant_input, "Error: GPTQ does not support weight quantizers that require quantized inputs."
        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        weight = self.layer.weight.data
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(
                    self.layer,
                (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d, qnn.QuantConvTranspose3d)):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)

        # List with permutation tensors for the Hessian and Weight matrix.
        # If act_order is False, the tensors will be ordered indexes.
        # For groupwise convolution, we have one tensor per group,
        # thus len(permutation_list) is always equal to self.groups.
        # We do not explicity permute the weight matrix, only the Hessian.
        permutation_list = []
        weight = weight.view(self.groups, -1, weight.shape[-1])
        # For groupwise convolution, these operations are groupwise so we iterate
        for i in range(self.groups):
            # If a diagonal element on the Hessian is zero, we can set to 0 the corresponding
            # column in the weight matrix.
            # The diagonal element is set to 1 to avoid division-by-zero
            dead = torch.diag(self.H[i, :, :]) == 0
            self.H[i, dead, dead] = 1
            # If the diagonal of activations is zero, we set the weight to zero
            weight[i, :, dead] = 0
            if self.act_order:
                # Re-order Hessian so that weights associated to
                # higher magnitude activations are quantized first
                perm = torch.argsort(torch.diag(self.H[i, :, :]), descending=True)
                self.H[i, :, :] = self.H[i, perm, :][:, perm]
            else:
                # No permutation, permutation tensor is a ordered index
                perm = torch.tensor(range(self.H.shape[-1]), device=dev)
            permutation_list.append(perm)

        # Try/Except in case the inverse Hessian cannot be computed
        try:
            for i in range(self.groups):
                damp = percdamp * torch.mean(torch.diag(self.H[i, :, :]))
                diag = torch.arange(self.columns, device='cpu')
                self.H[i, diag, diag] += damp
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :])
                self.H[i, :, :] = torch.cholesky_inverse(self.H[i, :, :])
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :], upper=True)
            h_inv = self.H
        except LinAlgError as e:
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, perm[i1:i2]], dtype=torch.float32)  # [groups, OC/groups, i2-i1]

            h_inv_block = h_inv[:, i1:i2, i1:i2]
            for i in range(count):
                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [groups, OC/groups]
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    q = q_groups[group_index]  # [OC/groups]
                    w = weight[group_index, :, perm[i1:i2][i]].to(torch.float32)  # [OC/groups]
                    d = h_inv_block[group_index, i, i]  # [1]
                    error = (w - q) / d  # [OC/groups]
                    error_block[group_index, :, i] = error
                    # We need to update the original weights
                    weight[group_index, :, perm[i1:i2][i:]] -= (
                        error.unsqueeze(1).matmul(
                            h_inv_block[group_index, i, i:].unsqueeze(0).to(dev))).to(dtype)

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(h_inv[group_index, i1:i2,
                                                          i2:].to(dev))).to(dtype)
        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)


class A2GPTQ(GPTQ):
    """
    Accumulator-Aware GPTQ (A2GPTQ) based on `A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance`
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            num_blocks,
            accumulator_bit_width) -> None:
        super().__init__(
            layer, name, act_order, len_parallel_layers, create_weight_orig, num_blocks)
        self.accumulator_bit_width = accumulator_bit_width
        assert self.accumulator_bit_width is not None

    def process_input(self, inp):
        inp = super().process_input(inp)

        is_quant_enabled = self.layer.weight_quant.is_quant_enabled

        # If using quantized activations, inp could be QuantTensor. In
        # this case, we overwrite the metadata.
        if isinstance(inp, QuantTensor):
            if is_quant_enabled and self.quant_input is None:
                self.quant_input = QuantTensor(
                    value=torch.empty(
                        1, dtype=self.layer.weight.dtype, device=self.layer.weight.device),
                    scale=inp.scale,
                    zero_point=inp.zero_point,
                    bit_width=inp.bit_width,
                    signed=inp.signed,
                    training=inp.training)
            inp = inp.value

        return inp

    def single_layer_update(self, percdamp=.01):
        # raise error in case no quant-input is here
        if self.quant_input is None:
            raise ValueError('Expected self.quant_input to calculate L1-norm upper bound, but recevied None. ' + \
                'Make sure that either the input to the model is a QuantTensor or the layer has an input quant enabled. ' \
                'Also, check if `use_quant_activations=True` in `gptq_mode` when `accumulator_bit_width` is specified. ' + \
                'Alternatively, provide a custom `a2q_layer_filter_fnc` to `gptq_mode` to filter out layers without a quantized input.')

        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        weight = self.layer.weight.data
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(
                    self.layer,
                (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d, qnn.QuantConvTranspose3d)):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)

        # List with permutation tensors for the Hessian and Weight matrix.
        # If act_order is False, the tensors will be ordered indexes.
        # For groupwise convolution, we have one tensor per group,
        # thus len(permutation_list) is always equal to self.groups.
        # We do not explicity permute the weight matrix, only the Hessian.
        permutation_list = []
        weight = weight.view(self.groups, -1, weight.shape[-1])
        # For groupwise convolution, these operations are groupwise so we iterate
        for i in range(self.groups):
            # If a diagonal element on the Hessian is zero, we can set to 0 the corresponding
            # column in the weight matrix.
            # The diagonal element is set to 1 to avoid division-by-zero
            dead = torch.diag(self.H[i, :, :]) == 0
            self.H[i, dead, dead] = 1
            # If the diagonal of activations is zero, we set the weight to zero
            weight[i, :, dead] = 0
            if self.act_order:
                # Re-order Hessian so that weights associated to
                # higher magnitude activations are quantized first
                perm = torch.argsort(torch.diag(self.H[i, :, :]), descending=True)
                self.H[i, :, :] = self.H[i, perm, :][:, perm]
            else:
                # No permutation, permutation tensor is a ordered index
                perm = torch.tensor(range(self.H.shape[-1]), device=dev)
            permutation_list.append(perm)

        # Try/Except in case the inverse Hessian cannot be computed
        try:
            for i in range(self.groups):
                damp = percdamp * torch.mean(torch.diag(self.H[i, :, :]))
                diag = torch.arange(self.columns, device='cpu')
                self.H[i, diag, diag] += damp
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :])
                self.H[i, :, :] = torch.cholesky_inverse(self.H[i, :, :])
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :], upper=True)
            h_inv = self.H
        except LinAlgError:
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H

        # get upper bound
        input_bit_width = self.quant_input.bit_width
        input_is_signed = self.quant_input.signed
        T = get_upper_bound_on_l1_norm(
            torch.tensor(self.accumulator_bit_width), input_bit_width, input_is_signed)
        s = self.layer.weight_quant.scale()
        if s.ndim > 1:
            s = s.view(self.groups, -1)  # [Groups, OC/Groups]

        # initialize cumulative l1-norm
        z = torch.zeros(weight.shape[:-1], device=dev)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, perm[i1:i2]], dtype=torch.float32)  # [groups, OC/groups, i2-i1]

            h_inv_block = h_inv[:, i1:i2, i1:i2]
            for i in range(count):
                # TODO: need to handle upwards rounding errors properly
                q_lim = s * torch.clamp_min(T - z - 0.5, 0.0)  # [groups, OC/groups]
                # NOTE: need to clamp before quantization
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    weight[:, :, perm[i1:i2][i]].clamp_(-q_lim, q_lim)
                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [groups, OC/groups]
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    q = q_groups[group_index]  # [OC/groups]
                    w = weight[group_index, :, perm[i1:i2][i]].to(torch.float32)  # [OC/groups]
                    d = h_inv_block[group_index, i, i]  # [1]
                    error = (w - q) / d  # [OC/groups]
                    error_block[group_index, :, i] = error
                    # We need to update the original weights
                    weight[group_index, :, perm[i1:i2][i:]] -= (
                        error.unsqueeze(1).matmul(
                            h_inv_block[group_index, i, i:].unsqueeze(0).to(dev))).to(dtype)
                # TODO: need to handle non-zero zero points properly
                z += q_groups.abs() / s  # increment cumulative l1-norm

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(h_inv[group_index, i1:i2,
                                                          i2:].to(dev))).to(dtype)
        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)


class gptq_mode(gpxq_mode):
    """
    Apply GPTQ algorithm https://arxiv.org/abs/2210.17323.

    Args:
        model (Module): The model to quantize with GPTQ
        group_of_parallel_layers (Optional, List[str]): .List of lists where each inner list is a group
            of layer names that can be optimized in parallel. Default: None
        inplace (bool): Wheter to apply GPTQ inplace or perform a deepcopy. Default: True
        create_weight_orig (bool): If True, store the original floating point weights before applying
            gptq. These weights will be used anytime quantization is disabled. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPTQ. Default: False
        num_blocks (int): The number of sub-blocks to use to speed-up GPTQ computation. Default: 100
        act_order (bool): Whether to order greedy path following by Hessian approximation. Default: False
        return_forward_output (bool): If True, returns the output of the forward pass. Otherwise the
            forward call inside the context manager returns None. Default: False

    Example:
        >>> with torch.no_grad():
        >>>     with gptq_mode(model) as gptq:
        >>>         gptq_model = gptq.model
        >>>         for i in tqdm(range(gptq.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 gptq_model(img)
        >>>             gptq.update()
    """

    def __init__(
            self,
            model,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            num_blocks: int = 100,
            return_forward_output: bool = False,
            act_order: bool = False,
            accumulator_bit_width: Optional[int] = None,
            a2q_layer_filter_fnc: Optional[Callable[[nn.Module], bool]] = lambda x: False,
            a2q_gptq_class: Optional[A2GPTQ] = A2GPTQ,
            use_random_proj: bool = False,
            use_random_sampling: bool = False,
            target_dim: int = 4096) -> None:
        if not inplace:
            model = deepcopy(model)
        super().__init__(
            model,
            group_of_parallel_layers,
            inplace,
            create_weight_orig,
            use_quant_activations,
            act_order,
            return_forward_output,
            use_random_proj,
            use_random_sampling,
            target_dim)

        # How many subblock to use during GPTQ for each layer
        self.num_blocks = num_blocks

        # A2GPTQ params
        self.accumulator_bit_width = accumulator_bit_width
        self.a2q_layer_filter_fnc = a2q_layer_filter_fnc  # returns true when to use A2GPTQ
        self.a2q_gptq_class = a2q_gptq_class

    def __enter__(self):
        self.orig_forward = self.model.forward
        if isinstance(self.model, (GraphModule, TorchGraphModule)):
            self.model.__class__.forward = self.catch_stopfwd
        else:
            self.model.forward = self.catch_stopfwd
        self.setup_gpxq_layers()
        return self.setup_gpxq_hooks()

    def __exit__(self, type, value, traceback):
        return self.exit()

    def catch_stopfwd(self, *args, **kwargs):
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass
        finally:
            if self.return_forward_output:
                # If we want to return the output of the network, we need to disable all hooks
                for name, gpxq_class in self.gpxq_layers.items():
                    gpxq_class.disable_pre_forward_hook = True
                out = self.orig_forward(*args, **kwargs)
                for name, gpxq_class in self.gpxq_layers.items():
                    gpxq_class.disable_pre_forward_hook = False
                return out

    def initialize_module_optimizer(
            self, layer, name, act_order, len_parallel_layers, create_weight_orig):
        if self.accumulator_bit_width is not None and self.a2q_layer_filter_fnc(layer):
            return self.a2q_gptq_class(
                layer=layer,
                name=name,
                act_order=act_order,
                len_parallel_layers=len_parallel_layers,
                create_weight_orig=create_weight_orig,
                num_blocks=self.num_blocks,
                accumulator_bit_width=self.accumulator_bit_width)
        return GPTQ(
            layer=layer,
            name=name,
            act_order=act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig,
            num_blocks=self.num_blocks,
            use_random_proj=self.use_random_proj,
            use_random_sampling=self.use_random_sampling,
            target_dim=self.target_dim)
