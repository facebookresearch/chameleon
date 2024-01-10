# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from xformers.ops import RMSNorm, fmha, rope_padded
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)


@dataclass
class ModelArgs:
    model_parallel_size: int = 1
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None
    vocab_size: int = -1
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    qk_normalization: bool = False
    swin_norm: bool = False


LayerCache = tuple[torch.Tensor, torch.Tensor]


class Attention(nn.Module):
    def __init__(
        self,
        model_parallel_size: int,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        qk_normalization: bool = False,
    ):
        super().__init__()

        self.model_parallel_size = model_parallel_size

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = n_kv_heads // model_parallel_size

        self.wqkv = nn.Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.wo = nn.Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
            dtype=torch.bfloat16,
        )

        self.qk_normalization = qk_normalization
        if qk_normalization:
            self.q_normalization = torch.nn.LayerNorm(head_dim)
            self.k_normalization = torch.nn.LayerNorm(head_dim)

        self._register_load_state_dict_pre_hook(self.load_hook)

    # This adapter makes sure we can load vanilla
    # Llama checkpoints where wq, wk, and wv are
    # not fused in a single parameter
    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        # x.shape is (sum(seq_lens), dim)
        #
        # Since we support heterogenous sequence
        # lengths, the hidden states are all
        # concatenated together along the usual
        # sequence dimension. The attention below
        # finds out where sequences start & end
        # using the provided attention bias.
        xqkv = self.wqkv(x)
        xq = xqkv[:, : (self.n_local_heads * self.head_dim)]
        xkv = xqkv[:, (self.n_local_heads * self.head_dim) :]
        xk, xv = xkv.chunk(2, 1)

        if self.qk_normalization:
            xq = xq.view(-1, self.n_local_heads, self.head_dim)
            xq = self.q_normalization(xq)
            xq = xq.view(-1, self.n_local_heads * self.head_dim)

            xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
            xk = self.k_normalization(xk)
            xk = xk.view(-1, self.n_local_kv_heads * self.head_dim)

        output_shape = xq.shape
        xq = xq.view(1, xq.shape[0], self.n_local_heads, self.head_dim)
        xk = xk.view(1, xk.shape[0], self.n_local_kv_heads, self.head_dim)
        xv = xv.view(1, xv.shape[0], self.n_local_kv_heads, self.head_dim)
        cache_k, cache_v = cache

        xq = rope_padded(
            xq=xq,
            xk=xk,
            xv=xv,
            cache_k=cache_k,
            cache_v=cache_v,
            attn_bias=attn_bias,
            theta=self.rope_theta,
        )

        # Handle GQA
        # Q shape: [B, M, Hkv, Hq // Hkv, K]
        heads_per_group = self.n_local_heads // self.n_local_kv_heads
        cache_k = cache_k.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1)
        cache_v = cache_v.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1)
        xq = xq.reshape(
            [*xq.shape[:2], self.n_local_kv_heads, heads_per_group, xq.shape[-1]]
        )

        # rope_padded() updated the caches, so we
        # call attention directly
        output = fmha.memory_efficient_attention_forward(
            xq, cache_k, cache_v, attn_bias
        )

        output = self.wo(output.reshape(output_shape))
        if self.model_parallel_size > 1:
            dist.all_reduce(output, group=group)

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        model_parallel_size: int,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()

        self.model_parallel_size = model_parallel_size

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % model_parallel_size == 0

        self.w13 = nn.Linear(
            dim,
            2 * hidden_dim // model_parallel_size,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim // model_parallel_size,
            dim,
            bias=False,
        )
        self._register_load_state_dict_pre_hook(self.load_hook)

    # This adapter makes sure we can load vanilla
    # Llama checkpoints where w1 and w3 are not
    # fused in a single parameter
    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix + "w1.weight" in state_dict:
            w1 = state_dict.pop(prefix + "w1.weight")
            w3 = state_dict.pop(prefix + "w3.weight")
            state_dict[prefix + "w13.weight"] = torch.cat([w1, w3])

    def forward(
        self, x: torch.Tensor, group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        output = self.w2(F.silu(x1) * x3)
        if self.model_parallel_size > 1:
            dist.all_reduce(output, group=group)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        if args.n_kv_heads is not None:
            n_kv_heads = args.n_kv_heads
        else:
            n_kv_heads = args.n_heads

        model_parallel_size = args.model_parallel_size
        assert args.n_heads % n_kv_heads == 0
        assert args.n_heads % model_parallel_size == 0
        assert n_kv_heads % model_parallel_size == 0

        self.attention = Attention(
            model_parallel_size=model_parallel_size,
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            qk_normalization=args.qk_normalization,
        )
        self.feed_forward = FeedForward(
            model_parallel_size=model_parallel_size,
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.swin_norm = args.swin_norm

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        if self.swin_norm:
            h = x + self.attention_norm(
                self.attention.forward(
                    x,
                    cache,
                    attn_bias,
                    group=group,
                )
            )
            out = h + self.ffn_norm(self.feed_forward(h, group=group))
        else:
            h = x + self.attention.forward(
                self.attention_norm(x),
                cache,
                attn_bias,
                group=group,
            )
            out = h + self.feed_forward(self.ffn_norm(h), group=group)
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.model_parallel_size = args.model_parallel_size
        assert args.dim % self.model_parallel_size == 0
        assert args.vocab_size > 0
        assert args.vocab_size % self.model_parallel_size == 0

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim // self.model_parallel_size,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size // self.model_parallel_size,
            bias=False,
        )

    @torch.no_grad()
    def forward_with_attn_bias(
        self,
        token_values: torch.Tensor,
        attn_bias: AttnBias,
        cache: list[LayerCache],
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)
        if self.model_parallel_size > 1:
            gather = [torch.empty_like(h) for _ in range(self.model_parallel_size)]
            dist.all_gather(gather, h, group=group)
            h = torch.cat(gather, dim=-1)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], attn_bias, group=group)

        logits = self.output(self.norm(h))
        if self.model_parallel_size > 1:
            gather = [torch.empty_like(logits) for _ in range(self.model_parallel_size)]
            dist.all_gather(gather, logits, group=group)
            logits = torch.cat(gather, dim=-1)
        return logits.float()

    def forward(
        self,
        token_values: torch.Tensor,
        token_lengths: torch.Tensor,
        start_pos: torch.Tensor,
        cache: list[LayerCache],
        kv_padding: int,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        attn_bias = AttnBias.from_seqlens(
            q_seqlen=token_lengths.tolist(),
            kv_seqlen=(start_pos + token_lengths).tolist(),
            kv_padding=kv_padding,
        )
        return self.forward_with_attn_bias(token_values, attn_bias, cache, group=group)


def make_cache(
    args: ModelArgs,
    length: int,
    device: str | torch.device | None = None,
    n_layers: int | None = None,
    dtype: torch.dtype | None = None,
) -> list[LayerCache]:
    """
    Allocate a cache to be used with the Transformer module.

    Args:
        args (ModelArgs): the model configuration.
        length (int): per layer cache size.
            It is usually budgeted as ``max_batch * max_seq``
        device (torch.device, optional): the device on which
            the cache should be allocated.
        n_layers (int, optional): the number of layers to
            allocate a cache for (defaults to the model
            settings).
        dtype (torch.dtype, optional): the dtype to use for
            cache entries (defaults to the default dtype).

    Returns:
        The cache object to pass to ``Tranformer.forward``.
    """

    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads
    if n_kv_heads is None:
        n_kv_heads = args.n_heads
    n_local_kv_heads = n_kv_heads // args.model_parallel_size

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_local_kv_heads, head_dim)
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )
        for _ in range(n_layers)
    ]


def cache_prefix(cache: list[LayerCache], length: int) -> list[LayerCache]:
    """
    Take a prefix view of a larger cache.

    The original cache object remains of identical size and valid
    after the shrinked alias has been used. This function is useful
    when a cache was allocated for a larger batch size than what is
    necessary.

    Args:
        cache: the cache to take a view in.
        length (int): the desired length

    Returns:
        A view in the input cache object.
    """

    if len(cache) > 0:
        assert cache[0][0].shape[1] >= length

    return [(ck[:, :length], cv[:, :length]) for ck, cv in cache]
