# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys
from ppseq.patch_utils import *
# try:
#     from .version import __version__  # noqa
# except ImportError:
#     version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
#     with open(version_txt) as f:
#         __version__ = f.read().strip()
#
# __all__ = ["pdb"]
#
# # backwards compatibility to support `from ppseq.X import Y`
# from ppseq.distributed import utils as distributed_utils
# from ppseq.logging import meters, metrics, progress_bar  # noqa
#
# sys.modules["ppseq.distributed_utils"] = distributed_utils
# sys.modules["ppseq.meters"] = meters
# sys.modules["ppseq.metrics"] = metrics
# sys.modules["ppseq.progress_bar"] = progress_bar
#
# # initialize hydra
# from ppseq.dataclass.initialize import hydra_init
#
# hydra_init()

import ppseq.criterions  # noqa
import ppseq.distributed  # noqa
import ppseq.models  # noqa
import ppseq.modules  # noqa
import ppseq.optim  # noqa
import ppseq.optim.lr_scheduler  # noqa
import ppseq.pdb  # noqa
import ppseq.scoring  # noqa
import ppseq.tasks  # noqa
import ppseq.token_generation_constraints  # noqa

import ppseq.benchmark  # noqa
import ppseq.model_parallel  # noqa
