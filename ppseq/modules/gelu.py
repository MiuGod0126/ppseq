# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
See "Gaussian Error Linear Units (GELUs)" by Dan Hendrycks and Kevin Gimpel with
the corresponding GitHub repo: https://github.com/hendrycks/GELUs
"""

import math

import paddle
import paddle.nn as nn


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + paddle.tanh(gelu_accurate._a * (x + 0.044715 * paddle.pow(x, 3))))
    )


def gelu(x: paddle.Tensor) -> paddle.Tensor:
    return paddle.nn.functional.gelu(x.float()).type_as(x)
