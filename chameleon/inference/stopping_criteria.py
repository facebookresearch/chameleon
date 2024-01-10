# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import torch


class StoppingCriteria:
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class StoppingCriteriaList(list):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return any(criteria(input_ids, scores, **kwargs) for criteria in self)


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        cur_len = input_ids.shape[-1]
        return cur_len >= self.max_length


class StopOnEOS(StoppingCriteria):
    def __init__(self, eos_id: int):
        self._eos_id = eos_id

    def __call__(self, input_ids: torch.LongTensor, _: torch.FloatTensor) -> bool:
        # input_ids.shape=[batch, seq_len]
        return (input_ids == self._eos_id).sum(dim=1).all()


class StopOnEOSAfterBatchIndex(StoppingCriteria):
    def __init__(self, eos_id: int, batch_index: list[int]):
        self._eos_id = eos_id
        self.batch_index = torch.tensor(batch_index, dtype=torch.long).unsqueeze(1)

    def __call__(self, input_ids: torch.LongTensor, _: torch.FloatTensor) -> bool:
        # input_ids.shape=[batch, seq_len]
        eos_mask = input_ids == self._eos_id
        consider_eos_mask = (
            torch.arange(input_ids.shape[1]).unsqueeze(0) >= self.batch_index
        )
        valid_eos = eos_mask & consider_eos_mask
        return valid_eos.sum(dim=1).all()
