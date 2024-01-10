# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod

import torch

from chameleon.inference import transformer
from chameleon.inference.alignment import (
    AlignPromptLeft,
    AlignPromptRight,
    PromptAlignment,
)
from chameleon.inference.cudagraph import cudagraph_wrap


class ModelAdapter(ABC):
    @abstractmethod
    def initialize(self, prompt_tokens: list[list[int]]):
        ...

    @abstractmethod
    def supports_alignment(self, alignment: PromptAlignment) -> bool:
        ...

    @abstractmethod
    @torch.inference_mode()
    def __call__(self, inputs: torch.LongTensor) -> torch.FloatTensor:
        ...


class ChameleonModelAdapter(ModelAdapter):
    """Adapter for Chameleon-style model that handles state, such as cache."""

    def __init__(
        self,
        model: transformer.Transformer,
        max_seq_len: int,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._args = model.args
        self._model = model
        self._max_seq_len = max_seq_len
        self._dtype = dtype or next(model.parameters()).data.dtype

    def initialize(self, prompt_tokens: list[list[int]]):
        self._prompt_lengths = [len(toks) for toks in prompt_tokens]
        batch_size = len(prompt_tokens)

        self._cache = transformer.make_cache(
            args=self._args,
            length=batch_size * self._max_seq_len,
            dtype=self._dtype,
        )

        self._local_inputs = torch.zeros([batch_size], dtype=int, device="cuda")

        self._forward = cudagraph_wrap(self._model.forward_with_attn_bias)

        self._first_pass = True

    def supports_alignment(self, alignment: PromptAlignment) -> bool:
        return isinstance(alignment, AlignPromptLeft) or isinstance(
            alignment, AlignPromptRight
        )

    def __call__(self, inputs: torch.LongTensor) -> torch.FloatTensor:
        # inputs.shape=[batch, seq-len]
        batch_size, seq_len = inputs.shape

        if self._first_pass:
            attn_seqlen = [min(pl, seq_len) for pl in self._prompt_lengths]
            self._bias = transformer.AttnBias.from_seqlens(
                q_seqlen=attn_seqlen,
                kv_seqlen=attn_seqlen,
                kv_padding=self._max_seq_len,
            )

            mask = torch.zeros_like(inputs, dtype=torch.bool)
            for i, k in enumerate(self._prompt_lengths):
                mask[i, -k:] = True

            flat_outputs: torch.Tensor = self._forward(  # type: ignore
                token_values=inputs[mask],
                attn_bias=self._bias,
                cache=self._cache,
            )
            self._local_outputs = torch.full(
                (inputs.shape[0], inputs.shape[1], flat_outputs.shape[-1]),
                -math.inf,
            )
            self._local_outputs[mask] = flat_outputs

            self._vocab_size = self._local_outputs.shape[-1]

            self._bias.q_seqinfo.seqstart.copy_(
                torch.arange(batch_size + 1, dtype=torch.int)
            )
            self._bias.q_seqinfo.max_seqlen = 1
            self._bias.q_seqinfo.seqstart_py = self._bias.q_seqinfo.seqstart.tolist()

            self._first_pass = False

        else:
            self._local_inputs.copy_(inputs[:, -1])  # type: ignore

            self._local_outputs = self._forward(  # type: ignore
                token_values=self._local_inputs,
                attn_bias=self._bias,
                cache=self._cache,
            )

        self._bias.k_seqinfo.seqlen.add_(1)
        return self._local_outputs.view(batch_size, -1, self._vocab_size)
