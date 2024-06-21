# Copyright (c) Meta Platforms, Inc. and affiliates
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torch
from omegaconf import DictConfig

from chameleon.inference import loader
from chameleon.viewer.backend.models.chameleon_distributed import (
    ChameleonDistributedGenerator,
)
from chameleon.viewer.backend.models.chameleon_local import ChameleonLocalGenerator
from chameleon.viewer.backend.models.service import serve
from chameleon.viewer.backend.utils import configure_rich_logging, get_logger

logger = get_logger(__name__)

VERSION = "2.0"
SEED = 42


def create_chameleon_generator(cfg: DictConfig):
    world_size = loader.detect_shard_count(cfg.model_path)
    if world_size > 1:
        torch.multiprocessing.set_start_method("spawn")
        generator = ChameleonDistributedGenerator(
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            vqgan_config_path=cfg.vqgan_config_path,
            vqgan_ckpt_path=cfg.vqgan_ckpt_path,
            additional_eos_tokens=cfg.additional_eos_tokens,
            world_size=world_size,
            master_address=cfg.distributed.master_address,
            master_port=cfg.distributed.master_port,
            redis_port=cfg.redis_port,
        )
    else:
        generator = ChameleonLocalGenerator(
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            vqgan_config_path=cfg.vqgan_config_path,
            vqgan_ckpt_path=cfg.vqgan_ckpt_path,
            additional_eos_tokens=cfg.additional_eos_tokens,
        )
    return generator


@hydra.main("../../../config", config_name="model_viewer", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    configure_rich_logging()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    logger.info("Starting viewer server with hydra cfg: %s", cfg)

    serve(
        create_chameleon_generator(cfg),
        cfg.host,
        cfg.port,
        debug=cfg.debug,
        redis_port=cfg.redis_port,
    )


if __name__ == "__main__":
    main()
