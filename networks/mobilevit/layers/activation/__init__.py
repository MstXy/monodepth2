#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse

SUPPORTED_ACT_FNS = []


def register_act_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_ACT_FNS:
            raise ValueError("Cannot register duplicate activation function ({})".format(name))
        SUPPORTED_ACT_FNS.append(name)
        return fn
    return register_fn


# automatically import different activation functions
act_dir = os.path.dirname(__file__)
for file in os.listdir(act_dir):
    path = os.path.join(act_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("networks.mobilevit.layers.activation." + model_name)


def arguments_activation_fn(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Non-linear functions", description="Non-linear functions")

    group.add_argument('--model.activation.name', default='relu', type=str, help='Non-linear function type')
    group.add_argument('--model.activation.inplace', action='store_true', help='Inplace non-linear functions')
    group.add_argument('--model.activation.neg-slope', default=0.1, type=float, help='Negative slope in leaky relu')

    return parser


# import later to avoid circular loop
from networks.mobilevit.layers.activation.gelu import GELU
from networks.mobilevit.layers.activation.hard_sigmoid import Hardsigmoid
from networks.mobilevit.layers.activation.hard_swish import Hardswish
from networks.mobilevit.layers.activation.leaky_relu import LeakyReLU
from networks.mobilevit.layers.activation.prelu import PReLU
from networks.mobilevit.layers.activation.relu import ReLU
from networks.mobilevit.layers.activation.relu6 import ReLU6
from networks.mobilevit.layers.activation.sigmoid import Sigmoid
from networks.mobilevit.layers.activation.swish import Swish


__all__ = [
    'GELU',
    'Hardsigmoid',
    'Hardswish',
    'LeakyReLU',
    'PReLU',
    'ReLU',
    'ReLU6',
    'Sigmoid',
    'Swish',
]
