# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from brevitas.fx import symbolic_trace
from brevitas.graph.channel_pruning import GraphChannelPruning
from brevitas.graph.fixed_point import MergeBatchNorm

from .equalization_fixtures import *


@pytest.mark.parametrize('prune_percentage', [0.1, 0.2, 0.5])
def test_pruning(toy_model, prune_percentage, request):
    # prune_percentage determines how many channels we want to set to 0
    model_name = request.node.callspec.id.split('-')[0]

    torch.manual_seed(SEED)

    model_class = toy_model
    model = model_class()
    if 'mha' in model_name:
        inp = torch.randn(IN_SIZE_LINEAR)
    else:
        inp = torch.randn(IN_SIZE_CONV)

    # set random channels for weights to 0

    model.eval()

    model = symbolic_trace(model)
    # merge BN before applying channel splitting
    model = MergeBatchNorm().apply(model)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # randomly set some output channels to 0
            print('shape of the weights:', module.weight.shape)
            ocs = module.out_channels if isinstance(
                module, torch.nn.Conv2d) else module.out_features
            rand_indices = torch.randperm(ocs)[:math.floor(ocs * prune_percentage)]
            for i in rand_indices:
                module.weight.data[i, :] = 0.0

    expected_out = model(inp)

    pruned_model = GraphChannelPruning().apply(model)

    # get pruned output
    out = pruned_model(inp)

    assert torch.allclose(out, expected_out, atol=0.01)
