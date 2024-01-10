# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import glob
import inspect
import json
from pathlib import Path

import torch

from chameleon.inference.transformer import ModelArgs, Transformer


def _convert(model_args: ModelArgs, consolidated_path: Path) -> Transformer:
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    model = Transformer(model_args)

    transfer_results = model.load_state_dict(
        torch.load(str(consolidated_path)),
        strict=False,
    )

    # TODO: More generally, assert missing or unexpected keys are buffers.
    assert transfer_results.missing_keys == []
    assert transfer_results.unexpected_keys == ["rope.freqs"]

    model.eval()

    torch.set_default_dtype(old_default_dtype)
    return model


def _get_checkpoint_path(src_dir: Path, rank: int | None) -> Path:
    base_path = src_dir / "consolidated.pth"
    if not rank and base_path.exists():
        return base_path

    alt_path = src_dir / f"consolidated.{rank:02}.pth"
    if alt_path.exists():
        return alt_path

    raise ValueError("Consolidated checkpoint not found.")


def load_model(path: str, rank: int | None = None) -> Transformer:
    src_dir = Path(path)

    with open(src_dir / "params.json", "r") as f:
        params = json.loads(f.read())
    with open(src_dir / "consolidate_params.json", "r") as f:
        consolidate_params = json.loads(f.read())
    params = {**params, **params["model"], **consolidate_params}

    known_param = inspect.signature(ModelArgs.__init__).parameters
    filtered_params = {k: v for k, v in params.items() if k in known_param}

    return _convert(
        ModelArgs(**filtered_params),
        _get_checkpoint_path(src_dir, rank),
    )


def detect_shard_count(path: str) -> int:
    src_dir = Path(path)
    if (src_dir / "consolidated.pth").exists():
        return 1
    return len(glob.glob(str(src_dir / "consolidated.*.pth")))
