# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Extra, Field

from chameleon.viewer.backend.models.abstract_model import (
    DEFAULT_MULTIMODAL_CFG_IMAGE,
    DEFAULT_MULTIMODAL_CFG_TEXT,
)


class WSMessageType(str, Enum):
    GENERATE_IMAGE = "GENERATE_IMAGE"
    GENERATE_TEXT = "GENERATE_TEXT"
    GENERATE_MULTIMODAL = "GENERATE_MULTIMODAL"
    PARTIAL_OUTPUT = "PARTIAL_OUTPUT"
    FULL_OUTPUT = "FULL_OUTPUT"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    QUEUE_STATUS = "QUEUE_STATUS"


class ContentType(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"


class Content(BaseModel):
    content_type: ContentType
    content: str

    class Config:
        extra = Extra.forbid


class NoOptionsForPartial(BaseModel):
    message_type: Literal[WSMessageType.PARTIAL_OUTPUT] = WSMessageType.PARTIAL_OUTPUT


class NoOptionsForFull(BaseModel):
    message_type: Literal[WSMessageType.FULL_OUTPUT] = WSMessageType.FULL_OUTPUT


class NoOptionsForComplete(BaseModel):
    message_type: Literal[WSMessageType.COMPLETE] = WSMessageType.COMPLETE


class NoOptionsForError(BaseModel):
    message_type: Literal[WSMessageType.ERROR] = WSMessageType.ERROR


class NoOptionsForQueueStatus(BaseModel):
    message_type: Literal[WSMessageType.QUEUE_STATUS] = WSMessageType.QUEUE_STATUS


class MultimodalGeneratorOptions(BaseModel):
    message_type: Literal[
        WSMessageType.GENERATE_MULTIMODAL
    ] = WSMessageType.GENERATE_MULTIMODAL
    temp: float = 0.7
    top_p: float = 0.9
    cfg_image_weight: float = DEFAULT_MULTIMODAL_CFG_IMAGE
    cfg_text_weight: float = DEFAULT_MULTIMODAL_CFG_TEXT
    yield_every_n: int = 32
    max_gen_tokens: int = 4096
    repetition_penalty: float = 1.2
    suffix_tokens: list[str] | None = None
    seed: int | None = None

    class Config:
        extra = Extra.forbid


class WSMultimodalMessage(BaseModel):
    message_type: WSMessageType
    content: list[Content]
    options: (
        MultimodalGeneratorOptions
        | NoOptionsForPartial
        | NoOptionsForFull
        | NoOptionsForError
        | NoOptionsForComplete
        | NoOptionsForQueueStatus
    ) = Field(..., discriminator="message_type")
    debug_info: dict[str, str] = {}
