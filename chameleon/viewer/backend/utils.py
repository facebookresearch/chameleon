# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import logging
import types

from rich.logging import RichHandler


def configure_rich_logging():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
        format=FORMAT,
        force=True,
    )


configure_rich_logging()


def get_logger(module: types.ModuleType) -> logging.Logger:
    """This forces logging.basicConfig to be called first."""
    logger = logging.getLogger(module)
    return logger
