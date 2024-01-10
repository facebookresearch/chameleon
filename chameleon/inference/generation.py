# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers.generation.streamers import BaseStreamer

from chameleon.inference.alignment import AlignPromptLeft, PromptAlignment
from chameleon.inference.model_adapter import ModelAdapter
from chameleon.inference.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from chameleon.inference.token_selector import MultinomialTokenSelector, TokenSelector


class ChameleonGenerator:
    @dataclass
    class Token:
        id: torch.LongTensor
        logits: torch.Tensor | None

    def __init__(
        self,
        model: ModelAdapter,
        input_ids: list[list[int]],
        stopping_criteria: StoppingCriteriaList | list[StoppingCriteria] | None = None,
        logits_processors: LogitsProcessorList | list[LogitsProcessor] | None = None,
        probability_processors: LogitsProcessorList
        | list[LogitsProcessor]
        | None = None,
        token_selector: TokenSelector | None = None,
        alignment: PromptAlignment = AlignPromptLeft(),
    ):
        assert model.supports_alignment(alignment)

        self.model = model

        self.stopping_criteria = stopping_criteria
        self.logits_processors = logits_processors
        self.probability_processors = probability_processors
        self.token_selector: TokenSelector = (
            token_selector or MultinomialTokenSelector()
        )

        self.alignment = alignment

        self.model.initialize(input_ids)

        self._inputs = self.alignment.prepare_inputs(
            input_ids
        )  # inputs.shape = [batch, seq-len]

        self._idx = 0
        self._start_idx = self.alignment.start_index(input_ids)

        self._original_inputs = self._inputs.clone()
        self._inputs = self._inputs[:, : self._start_idx]

    def __iter__(self):
        return self

    @torch.inference_mode()
    def __next__(self) -> Token:
        # Are we done?
        if self.stopping_criteria(self._inputs, None):
            raise StopIteration

        # Emit initial tokens.
        # Model is not run for these.
        # If you want the logits, you can do a separate forward pass outside generation.
        if self._idx < self._start_idx:
            idx, self._idx = self._idx, self._idx + 1
            return ChameleonGenerator.Token(id=self._inputs[:, idx], logits=None)

        # Run the model for the next token.
        self._inputs = self._inputs.contiguous()
        outputs = self.model(self._inputs)  # outputs.shape = [batch, seq-len, vocab]

        # Pull out and process the logits.
        logits = outputs[:, -1, :]  # logits.shape = [batch, vocab]
        logits = self.logits_processors(self._inputs, logits)
        probs = logits.softmax(dim=1)  # probs.shape = [batch, vocab]
        probs = self.probability_processors(self._inputs, probs)

        # Select a token and add it to the inputs.
        next_tokens = self.token_selector(
            self._inputs, probs
        )  # next_tokens.shape = [batch]
        self._inputs = torch.cat([self._inputs, next_tokens[:, None]], dim=1)

        # Run alignment specific postprocessing.
        self._inputs = self.alignment.postprocess_inputs(
            self._inputs, self._original_inputs
        )

        # Return the next step result.
        return ChameleonGenerator.Token(id=self._inputs[:, -1], logits=logits)

    @property
    def stopping_criteria(self) -> StoppingCriteriaList:
        return self._stopping_criteria

    @stopping_criteria.setter
    def stopping_criteria(
        self, value: StoppingCriteriaList | list[StoppingCriteria] | None
    ):
        self._stopping_criteria = StoppingCriteriaList(value or [])

    @property
    def logits_processors(self) -> LogitsProcessorList:
        return self._logits_processors

    @logits_processors.setter
    def logits_processors(
        self, value: LogitsProcessorList | list[LogitsProcessor] | None
    ):
        self._logits_processors = LogitsProcessorList(value or [])

    @property
    def probability_processors(self) -> LogitsProcessorList:
        return self._probability_processors

    @probability_processors.setter
    def probability_processors(
        self, value: LogitsProcessorList | list[LogitsProcessor] | None
    ):
        self._probability_processors = LogitsProcessorList(value or [])


def run_generation(
    model: torch.nn.Module,
    input_ids: list[list[int]],
    stopping_criteria: StoppingCriteriaList | list[StoppingCriteria],
    logits_processors: LogitsProcessorList | list[LogitsProcessor] | None = None,
    probability_processors: LogitsProcessorList | list[LogitsProcessor] | None = None,
    token_selector: TokenSelector | None = None,
    alignment: PromptAlignment = AlignPromptLeft(),
    streamer: BaseStreamer | None = None,
) -> torch.LongTensor:
    result = torch.empty((len(input_ids), 0), dtype=int)
    for tok in ChameleonGenerator(
        model=model,
        input_ids=input_ids,
        stopping_criteria=stopping_criteria,
        logits_processors=logits_processors,
        probability_processors=probability_processors,
        token_selector=token_selector,
        alignment=alignment,
    ):
        if streamer is not None:
            streamer.put(tok.id)
        result = torch.cat([result, tok.id.view(-1, 1)], dim=1)

    if streamer is not None:
        streamer.end()

    return result
