# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

from ppseq.dataclass.utils import gen_parser_from_dataclass
from paddle.optimizer.lr import LRScheduler


class PaddleseqLRScheduler(LRScheduler):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.best = None

    @classmethod
    def add_args(cls, parser):
        """Add arguments to the parser for this LR scheduler."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

class LegacyPaddleseqLRScheduler(PaddleseqLRScheduler):
    def __init__(self, args: Namespace):
        self.args = args
        self.best = None
