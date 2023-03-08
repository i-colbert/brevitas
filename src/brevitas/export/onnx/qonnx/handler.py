# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from abc import ABC
from copy import copy

import torch
import torch.nn as nn
from torch import Tensor

from brevitas.quant_tensor import QuantTensor
from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.onnx.handler import QuantLSTMLayerHandler
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy import TruncQuantProxyFromInjector
from brevitas.proxy import InputBitWidthDecoupledWeightQuantProxyFromInjector as InputBitWidthDWQPFI

from .function import BrevitasBinaryQuantFn
from .function import BrevitasQuantFn
from .function import BrevitasQuantLSTMCellFn
from .function import BrevitasTruncFn


class BrevitasQuantProxyHandler(ONNXBaseHandler, ABC):

    def validate(self, module):
        if module.bit_width() == 1:
            assert module.zero_point() == 0, "Zero-point not supported for binary quant."

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs = {
                'scale': module.scale(),
                'zero_point': module.zero_point(),
                'bit_width': module.bit_width(),
                'narrow_range': module.is_narrow_range,
                'signed': module.is_signed,
                'rounding_mode': module.rounding_mode}

    def symbolic_execution(self, x: Tensor):
        scale = self.symbolic_kwargs['scale']
        zero_point = self.symbolic_kwargs['zero_point']
        bit_width = self.symbolic_kwargs['bit_width']
        if bit_width == 1:
            x = BrevitasBinaryQuantFn.apply(x, *self.symbolic_kwargs.values())
        else:
            x = BrevitasQuantFn.apply(x, *self.symbolic_kwargs.values())
        return x, scale, zero_point, bit_width


class BrevitasWeightQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.quant_weights = None

    def prepare_for_export(self, module: WeightQuantProxyFromInjector):
        super().prepare_for_export(module)
        quant_weights = {
            tm.weight.data_ptr(): tm.quant_weight().value for tm in module.tracked_module_list}
        self.quant_weights = quant_weights
        # override rounding mode since quantization has been pre-applied
        self.symbolic_kwargs['rounding_mode'] = 'ROUND'

    def symbolic_execution(self, x: Tensor):
        quant_weight = self.quant_weights[x.data_ptr()]
        return super().symbolic_execution(quant_weight)


class BrevitasDecoupledWeightQuantProxyHandler(BrevitasWeightQuantProxyHandler):
    handled_layer = DecoupledWeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.extra_kwargs = {}

    def prepare_for_export(self, module: DecoupledWeightQuantProxyFromInjector):
        super().prepare_for_export(module)
        self.extra_kwargs['pre_scale'] = module.pre_scale()
        self.extra_kwargs['pre_zero_point'] = module.pre_zero_point()

    def symbolic_execution(self, x: Tensor):
        out, scale, zero_point, bit_width = super().symbolic_execution(x)
        pre_scale = self.extra_kwargs['pre_scale']
        pre_zero_point = self.extra_kwargs['pre_zero_point']
        return out, pre_scale, pre_zero_point, scale, zero_point, bit_width

class BrevitasInputBitWidthDecoupledWeightQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = InputBitWidthDWQPFI

    def __init__(self):
        super().__init__()
        self.quant_weight = None

    def validate(self, module):
        """Validate im"""
        quant_weight: QuantTensor = module.quant_weight()
        if quant_weight.bit_width == 1:
            assert quant_weight.zero_point() == 0, "Zero-point not supported for binary quant."
        input_bit_width = module.quant_input_bit_width() # N
        accumulator_bit_width = module.quant_accumulator_bit_width() # P
        if isinstance(module, nn.Conv2d):
            l1_norm = quant_weight.int().float().norm(p=1, dim=(1,2,3)).max()
        elif isinstance(module, nn.Linear):
            l1_norm = quant_weight.int().float().norm(p=1, dim=(1,)).max()
        else:
            raise NotImplementedError("Module not yet supported for accumulator-aware quantization")
        alpha: Tensor = input_bit_width - int(module.is_quant_input_signed) + l1_norm.log2().squeeze()
        phi = lambda x: torch.log2(1. + pow(2, -x))
        assert (alpha + phi(alpha) + 1. <= accumulator_bit_width).all()

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            # Quantizer sharing is not supported for export 
            assert len(module.tracked_module_list) == 1
            self.validate(module.tracked_module_list[0])
            quant_weight = module.tracked_module_list[0].quant_weight()
            self.quant_weight = quant_weight.value
            self.symbolic_kwargs = {
                'scale': quant_weight.scale,
                'zero_point': quant_weight.zero_point,
                'bit_width': quant_weight.bit_width,
                # narrow range is not a property of the QuantTensor take it from the proxy instead
                'narrow_range': module.is_narrow_range,
                'signed': quant_weight.signed,
                # override rounding mode since quantization has been pre-applied
                'rounding_mode': 'ROUND'}

    def symbolic_execution(self, x: Tensor, input_bit_width: Tensor, input_is_signed: bool):
        # Quant weight bit width is retrieved through a cache, so we can just ignore the input_bit_width and input_is_signed
        # We can't use x.data_ptr() as key to a dict because it points to the output of a computation
        return super().symbolic_execution(self.quant_weight)


class BrevitasActQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = ActQuantProxyFromInjector


class BrevitasBiasQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = BiasQuantProxyFromInjector

    def symbolic_execution(self, x: Tensor, input_scale=None, input_bit_width=None):
        # avoid in place pop in case the proxy is shared
        symbolic_kwargs = copy(self.symbolic_kwargs)
        scale = symbolic_kwargs.pop('scale')
        bit_width = symbolic_kwargs.pop('bit_width')
        zero_point = symbolic_kwargs.pop('zero_point')
        if scale is None:
            assert input_scale is not None, 'Input scale required for bias export'
            scale = input_scale
        if bit_width is None:
            assert input_bit_width is not None, 'Input bit_width required for bias export'
            bit_width = input_bit_width
        y = BrevitasQuantFn.apply(
            x, scale, zero_point, bit_width, *symbolic_kwargs.values())
        return y, scale, zero_point, bit_width


class BrevitasTruncQuantProxyHandler(ONNXBaseHandler):
    handled_layer = TruncQuantProxyFromInjector

    def prepare_for_export(self, module: TruncQuantProxyFromInjector):
        self.symbolic_kwargs = {
                'output_bit_width': module.bit_width(),
                'rounding_mode': module.rounding_mode}

    def symbolic_execution(
            self, x: Tensor, scale: Tensor, zero_point: Tensor, input_bit_width: Tensor, signed: Tensor):
        y = BrevitasTruncFn.apply(
            x, scale, zero_point, input_bit_width, *self.symbolic_kwargs.values())
        return y, scale, zero_point, self.symbolic_kwargs['output_bit_width']


class BrevitasQuantLSTMLayerHandler(QuantLSTMLayerHandler):

    def quantized_cell_symbolic_execution(
        self,
        quant_input,
        quant_hidden_state,
        quant_cell_state,
        quant_weight_ii,
        quant_weight_if,
        quant_weight_ic,
        quant_weight_io,
        quant_weight_hi,
        quant_weight_hf,
        quant_weight_hc,
        quant_weight_ho,
        quant_bias_input,
        quant_bias_forget,
        quant_bias_cell,
        quant_bias_output):
        return BrevitasQuantLSTMCellFn.apply(
            quant_input,
            quant_hidden_state,
            quant_cell_state,
            quant_weight_ii,
            quant_weight_if,
            quant_weight_ic,
            quant_weight_io,
            quant_weight_hi,
            quant_weight_hf,
            quant_weight_hc,
            quant_weight_ho,
            quant_bias_input,
            quant_bias_forget,
            quant_bias_cell,
            quant_bias_output,
            *self.symbolic_kwargs.values())
