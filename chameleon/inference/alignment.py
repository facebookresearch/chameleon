# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch


class PromptAlignment(ABC):
    @abstractmethod
    def start_index(self, input_ids: list[list[int]]) -> int:
        ...

    @abstractmethod
    def prepare_inputs(self, input_ids: list[list[int]]) -> torch.Tensor:
        ...

    @abstractmethod
    def postprocess_inputs(
        self, inputs: torch.Tensor, original_inputs: torch.Tensor
    ) -> torch.Tensor:
        ...


class AlignPromptRight(PromptAlignment):
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def start_index(self, input_ids: list[list[int]]) -> int:
        return max(len(sublist) for sublist in input_ids)

    def prepare_inputs(self, input_ids: list[list[int]]) -> torch.LongTensor:
        max_length = max(len(sublist) for sublist in input_ids)
        return torch.tensor(
            [
                ([self.pad_id] * (max_length - len(sublist))) + sublist
                for sublist in input_ids
            ],
            requires_grad=False,
        )

    def postprocess_inputs(
        self,
        inputs: torch.Tensor,
        original_inputs: torch.Tensor,
    ) -> torch.Tensor:
        return inputs


class AlignPromptLeft(PromptAlignment):
    def __init__(self, pad_id: int = -1):
        self.pad_id = pad_id

    def start_index(self, input_ids: list[list[int]]) -> int:
        return min(len(sublist) for sublist in input_ids)

    def prepare_inputs(self, input_ids: list[list[int]]) -> torch.Tensor:
        max_length = max(len(sublist) for sublist in input_ids)
        return torch.tensor(
            [
                sublist + ([self.pad_id] * (max_length - len(sublist)))
                for sublist in input_ids
            ],
            requires_grad=False,
        )

    def postprocess_inputs(
        self,
        inputs: torch.Tensor,
        original_inputs: torch.Tensor,
    ) -> torch.Tensor:
        max_init_len = original_inputs.shape[1]
        if inputs.shape[1] <= max_init_len:
            original_inputs_limited = original_inputs[:, : inputs.shape[1]]
            mask = original_inputs_limited != self.pad_id
            inputs[mask] = original_inputs_limited[mask]
        return inputs
