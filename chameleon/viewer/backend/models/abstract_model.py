# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass
from typing import Generator

import PIL.Image

# images, joined retrieval queries, retrieval images
MixedTokenType = str | PIL.Image.Image
MixedSequenceType = list[MixedTokenType]


@dataclass
class StreamingImage:
    image: PIL.Image.Image
    final: bool


DEFAULT_MULTIMODAL_CFG_IMAGE = 1.2
DEFAULT_MULTIMODAL_CFG_TEXT = 3.0
DEFAULT_IMAGE_CFG_IMAGE = 3.0
DEFAULT_IMAGE_CFG_TEXT = 3.0


class AbstractMultimodalGenerator(abc.ABC):
    @abc.abstractmethod
    def generate_text_streaming(
        self,
        prompts: list[MixedSequenceType],
        temp: float = 1.0,
        top_p: float = 0.8,
        seed: int | None = None,
    ) -> Generator[list[str], None, None]:
        pass

    @abc.abstractmethod
    def generate_image_streaming(
        self,
        prompt: MixedSequenceType,
        temp: float = 1.0,
        top_p: float = 0.8,
        cfg_image_weight: float = DEFAULT_IMAGE_CFG_IMAGE,
        cfg_text_weight: float = DEFAULT_IMAGE_CFG_TEXT,
        yield_every_n: int = 32,
        seed: int | None = None,
    ) -> Generator[PIL.Image.Image, None, None]:
        pass

    @abc.abstractmethod
    def generate_multimodal_streaming(
        self,
        prompt: MixedSequenceType,
        temp: float = 1.0,
        top_p: float = 0.8,
        cfg_image_weight: float = DEFAULT_MULTIMODAL_CFG_IMAGE,
        cfg_text_weight: float = DEFAULT_MULTIMODAL_CFG_TEXT,
        yield_every_n: int = 32,
        max_gen_tokens: int = 4096,
        repetition_penalty: float = 1.2,
        suffix_tokens: list[str] | None = None,
        seed: int | None = None,
    ) -> Generator[MixedSequenceType, None, None]:
        pass
