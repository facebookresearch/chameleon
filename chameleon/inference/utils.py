# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import socket
from typing import Generator, Generic, Iterator, TypeVar

T = TypeVar("T")


class DynamicGenerator(Generic[T]):
    def __init__(self, gen: Generator[T, None, None]):
        self.gen = gen

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        return next(self.gen)


def advance(iterator: Iterator[T], steps: int):
    try:
        for _ in range(steps):
            next(iterator)
    except StopIteration:
        pass


def random_unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
