# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import torch


class TokenSelector:
    def __call__(
        self, input_ids: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape=[batch, seq_len]
        # probs.shape=[batch, vocab]
        ...


class ArgmaxTokenSelector(TokenSelector):
    def __call__(
        self, _: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.LongTensor:
        # probs.shape=[batch, vocab]
        return probs.argmax(dim=1)


class MultinomialTokenSelector(TokenSelector):
    def __call__(
        self, _: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.LongTensor:
        # probs.shape=[batch, vocab]
        return probs.multinomial(num_samples=1).squeeze(1)


class ReplicatedInputTokenSelector(TokenSelector):
    def __init__(self, token_selector: TokenSelector, n: int):
        self.token_selector = token_selector
        self.n = n

    def __call__(
        self, input_ids: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.LongTensor:
        # input_ids.shape=[n*batch, seq_len]
        # probs.shape=[n*batch, vocab]
        primary_input_ids = torch.chunk(input_ids, chunks=self.n, dim=0)[0]
        primary_probs = torch.chunk(probs, chunks=self.n, dim=0)[0]
        tokens = self.token_selector(primary_input_ids, primary_probs)
        return tokens.repeat(self.n)
