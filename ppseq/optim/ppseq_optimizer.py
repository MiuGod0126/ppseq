# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import paddle
from ppseq import utils
from ppseq.dataclass.utils import gen_parser_from_dataclass
from paddle.optimizer import Optimizer

class PaddleseqOptimizer(Optimizer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def add_args(cls, parser):
        """Add optimizer-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())


class LegacyPaddleseqOptimizer(PaddleseqOptimizer):
    def __init__(self, args):
        self.args = args
