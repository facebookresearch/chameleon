# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from transformers import LogitsProcessor


class TopPProbabilityProcessor(LogitsProcessor):
    # Modified version of TopPLogitsWarper to act on probabilities.
    # Changes:
    # * filter_value changed from -inf to 0
    # * removed softmax
    # * renormalize L1

    def __init__(
        self,
        top_p: float,
        min_tokens_to_keep: int = 1,
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(
                f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
            )

        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape=[batch, seq-len]
        # probs.shape=[batch, vocab]
        sorted_probs, sorted_indices = torch.sort(probs, descending=False)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs


class DisallowTokensInIndexRangeLogitsProcessor(LogitsProcessor):
    def __init__(
        self, token_ids: list[int], start_index: int, end_index: int | None = None
    ):
        self.token_ids = torch.tensor(token_ids)
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else math.inf

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        current_index = input_ids.shape[1]
        if self.start_index <= current_index < self.end_index:
            logits[:, self.token_ids] = -math.inf
        return logits


class DisallowTokensLogitsProcessor(DisallowTokensInIndexRangeLogitsProcessor):
    def __init__(self, token_ids: list[int]):
        super().__init__(token_ids, 0)


class DisallowTokensAtIndexLogitsProcessor(DisallowTokensInIndexRangeLogitsProcessor):
    def __init__(self, token_ids: list[int], index: int):
        super().__init__(token_ids, index, index + 1)


class DisallowTokensAfterIndexLogitsProcessor(
    DisallowTokensInIndexRangeLogitsProcessor
):
    def __init__(self, token_ids: list[int], index: int):
        super().__init__(token_ids, index + 1)


class DisallowTokensAtOrAfterIndexLogitsProcessor(
    DisallowTokensInIndexRangeLogitsProcessor
):
    def __init__(self, token_ids: list[int], index: int):
        super().__init__(token_ids, index)


class DisallowTokensInBatchIndexRangeLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        token_ids: list[int],
        start_indices: list[int],
        end_indices: list[int] | None = None,
    ):
        self.token_ids = torch.tensor(token_ids)
        self.start_indices = torch.tensor(start_indices)
        self.end_indices = (
            torch.tensor(end_indices)
            if end_indices is not None
            else torch.full_like(self.start_indices, math.inf, dtype=torch.float)
        )

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape = [batch, seq_len]
        # logits.shape = [batch, vocab]
        current_index = input_ids.shape[1]
        mask = (self.start_indices <= current_index) & (
            current_index < self.end_indices
        )
        # The following will fail if the mask is all False.
        # logits[mask, self.token_ids] = -math.inf
        logits[torch.where(mask)[0].unsqueeze(1), self.token_ids] = -math.inf
        return logits


class DisallowTokensAtBatchIndexLogitsProcessor(
    DisallowTokensInBatchIndexRangeLogitsProcessor
):
    def __init__(self, token_ids: list[int], batch_index: list[int]):
        super().__init__(token_ids, batch_index, [i + 1 for i in batch_index])


class AllowOnlyTokensInIndexRangeLogitsProcessor(LogitsProcessor):
    def __init__(
        self, token_ids: list[int], start_index: int, end_index: int | None = None
    ):
        self.token_ids = torch.tensor(token_ids)
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else math.inf

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        current_index = input_ids.shape[1]
        if self.start_index <= current_index < self.end_index:
            replacement = torch.full_like(logits, -math.inf)
            replacement[:, self.token_ids] = logits[:, self.token_ids]
            logits[:] = replacement
        return logits


class AllowOnlyTokensLogitsProcessor(AllowOnlyTokensInIndexRangeLogitsProcessor):
    def __init__(self, token_ids: list[int]):
        super().__init__(token_ids, 0)


class AllowOnlyTokensAtIndexLogitsProcessor(AllowOnlyTokensInIndexRangeLogitsProcessor):
    def __init__(self, token_ids: list[int], index: int):
        super().__init__(token_ids, index, index + 1)


class AllowOnlyTokensAfterIndexLogitsProcessor(
    AllowOnlyTokensInIndexRangeLogitsProcessor
):
    def __init__(self, token_ids: list[int], index: int):
        super().__init__(token_ids, index + 1)


class AllowOnlyTokensAtOrAfterIndexLogitsProcessor(
    AllowOnlyTokensInIndexRangeLogitsProcessor
):
    def __init__(self, token_ids: list[int], index: int):
        super().__init__(token_ids, index)


class AllowOnlyTokensInBatchIndexRangeLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        token_ids: list[int],
        start_indices: list[int],
        end_indices: list[int] | None = None,
    ):
        self.token_ids = torch.tensor(token_ids)
        self.start_indices = torch.tensor(start_indices)
        self.end_indices = (
            torch.tensor(end_indices)
            if end_indices is not None
            else torch.full_like(self.start_indices, math.inf, dtype=torch.float)
        )

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape = [batch, seq_len]
        # logits.shape = [batch, vocab]
        current_index = input_ids.shape[1]
        mask = (self.start_indices <= current_index) & (
            current_index < self.end_indices
        )

        valid_batch_indices = torch.where(mask)[0].unsqueeze(1)
        full_mask = torch.full_like(logits, -math.inf)
        full_mask[valid_batch_indices, self.token_ids] = logits[
            valid_batch_indices, self.token_ids
        ]

        logits[:] = torch.where(full_mask != -math.inf, full_mask, logits)
        return logits


class AllowOnlyTokensAtRelativeOffsetLogitsProcessor(LogitsProcessor):
    def __init__(
        self, trigger_token_id: int, subsequent_token_ids: list[int], offset: int
    ):
        self.trigger_token_id = trigger_token_id
        self.subsequent_token_ids = torch.tensor(subsequent_token_ids)
        self.offset = offset

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape=[batch, seq_len]
        # logits.shape=[batch, vocab]
        if input_ids.shape[1] < self.offset:
            return logits

        trigger_positions = (
            input_ids[:, -self.offset] == self.trigger_token_id
        ).unsqueeze(-1)

        disallowed_tokens_mask = torch.ones_like(logits, dtype=bool)
        disallowed_tokens_mask[:, self.subsequent_token_ids] = False

        return logits.masked_fill_(
            disallowed_tokens_mask & trigger_positions,
            -math.inf,
        )


class AllowOnlyTokensInRelativeWindowLogitsProcessor(LogitsProcessor):
    def __init__(self, trigger_token_id: int, allowed_token_ids: list[int], width: int):
        self.trigger_token_id = trigger_token_id
        self.allowed_token_ids = torch.tensor(allowed_token_ids).unsqueeze(
            0
        )  # shape: [1, num_allowed_tokens]
        self.width = width

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape=[batch, seq_len]
        # logits.shape=[batch, vocab]
        width = min(self.width, input_ids.shape[1])
        trigger_positions = (
            (input_ids[:, -width:] == self.trigger_token_id).any(dim=1).unsqueeze(-1)
        )

        disallowed_tokens_mask = torch.ones_like(logits, dtype=bool)
        disallowed_tokens_mask[:, self.allowed_token_ids] = False

        return logits.masked_fill_(
            disallowed_tokens_mask & trigger_positions,
            -math.inf,
        )


class CFGLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        guidance_scale: float,
        unconditional_ids: torch.LongTensor,
        model,
    ):
        self.guidance_scale = guidance_scale
        self.unconditional_ids = unconditional_ids
        self.model = model

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        conditioned_logits = logits

        self.unconditional_ids = torch.cat(
            [self.unconditional_ids, input_ids[:, -1:]], dim=1
        )
        unconditioned_outputs = self.model(self.unconditional_ids)
        unconditioned_logits = unconditioned_outputs[:, -1, :]
        return (
            self.guidance_scale * (conditioned_logits - unconditioned_logits)
            + unconditioned_logits
        )


class InBatchCFGLogitsProcessor(LogitsProcessor):
    def __init__(self, guidance_scale: float):
        self.guidance_scale = guidance_scale

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape=[2*batch, seq-len]
        # logits.shape=[2*batch, vocab]
        conditioned_logits, unconditioned_logits = torch.chunk(logits, chunks=2, dim=0)
        mixed_logits = unconditioned_logits + self.guidance_scale * (
            conditioned_logits - unconditioned_logits
        )
        return mixed_logits.repeat(2, 1)


class InBatchInstructCFGLogitsProcessor(LogitsProcessor):
    # See https://arxiv.org/abs/2211.09800

    def __init__(self, guidance_scale_text: float, guidance_scale_image: float):
        self.guidance_scale_text = guidance_scale_text
        self.guidance_scale_image = guidance_scale_image

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape=[3*batch, seq-len]
        # logits.shape=[3*batch, vocab]
        (
            full_conditioned_logits,
            image_conditioned_logits,
            unconditioned_logits,
        ) = logits.chunk(3)
        mixed_logits = (
            unconditioned_logits
            + self.guidance_scale_image
            * (image_conditioned_logits - unconditioned_logits)
            + self.guidance_scale_text
            * (full_conditioned_logits - image_conditioned_logits)
        )
        return mixed_logits.repeat(3, 1)
