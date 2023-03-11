#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

# We need to setup root logger before importing any ppseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ppseq_cli.train")

import numpy as np
import paddle
from ppseq import (
    checkpoint_utils,
    options,
    tasks,
    utils,
)
# from paddleseq.data import iterators, data_utils
# from paddleseq.data.plasma_utils import PlasmaStore
from ppseq.dataclass.configs import PaddleseqConfig
from ppseq.dataclass.utils import convert_namespace_to_omegaconf
# from paddleseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from ppseq.file_io import PathManager
# from paddleseq.logging import meters, metrics, progress_bar
# from paddleseq.model_parallel.megatron_trainer import MegatronTrainer
# from paddleseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf




from ppseq import options
def cli_main() -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    cfg = convert_namespace_to_omegaconf(args)
    print(cfg)
    # if cfg.common.use_plasma_view:
    #     server = PlasmaStore(path=cfg.common.plasma_path)
    #     logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    # if args.profile:
    #     with torch.cuda.profiler.profile():
    #         with torch.autograd.profiler.emit_nvtx():
    #             distributed_utils.call_main(cfg, main)
    # else:
    #     distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
