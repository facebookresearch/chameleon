# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Callable, TypeVar

import torch

T = TypeVar("T")
FN = Callable[..., T]  # type: ignore


class CUDAGraphWrapper:
    def __init__(
        self,
        fn: FN[T],
        warmup_iter: int = 1,
        debug_dump_path: str | None = None,
    ):
        self.fn = fn
        self.warmup_iter = warmup_iter
        self.debug_dump_path = debug_dump_path
        self.graph: torch.cuda.CUDAGraph | None = None
        self.result: T | None = None

    def __call__(self, *args, **kwargs) -> Any:  # type: ignore
        if self.warmup_iter > 0:
            self.warmup_iter -= 1
            return self.fn(*args, **kwargs)

        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            if self.debug_dump_path is not None:
                self.graph.enable_debug_mode()
            recording_kwargs = {}
            if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
                # In PyTorch 2.1+ and nightlies from late Aug 2023,
                # we can do this to maybe avoid watchdog-related crashes
                recording_kwargs["capture_error_mode"] = "thread_local"
            with torch.cuda.graph(self.graph, **recording_kwargs):
                self.result = self.fn(*args, **kwargs)
            torch.cuda.synchronize()
            if self.debug_dump_path is not None:
                self.graph.debug_dump(self.debug_dump_path)

        assert self.graph is not None
        self.graph.replay()
        return self.result


def cudagraph_wrap(
    *args,
    warmup_iter: int = 1,
    debug_dump_path: str | None = None,
) -> Callable[[FN[T]], FN[T]]:
    def wrapper(fn: FN[T]) -> FN[T]:
        graph_wrapper = CUDAGraphWrapper(
            fn, warmup_iter=warmup_iter, debug_dump_path=debug_dump_path
        )

        @functools.wraps(fn)
        def call_wrapper(*inner_args, **inner_kwargs):
            return graph_wrapper(*inner_args, **inner_kwargs)

        return call_wrapper

    # @cudagraph_wrap
    # def fn(...):
    #   ...
    #
    # - or -
    #
    # fast_fn = cudagraph_wrap(slow_fn, warmup_iter=2)
    if len(args) == 1 and callable(args[0]):
        return wrapper(args[0])

    # @cudagraph_wrap(warmup_iter=3)
    # def fn(...):
    #   ...
    def decorator(fn: FN[T]) -> FN[T]:
        return wrapper(fn)

    return decorator
