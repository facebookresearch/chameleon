# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import multiprocessing
import os
import random
import sys
import threading
import time
import traceback
from functools import partial
from typing import Any, Generator, TypeVar

import redis
import redis.asyncio as async_redis
import torch
from tokenizers import Tokenizer

from chameleon.inference.image_tokenizer import ImageTokenizer
from chameleon.inference.loader import load_model
from chameleon.inference.vocab import VocabInfo
from chameleon.viewer.backend.data_types import WSMessageType
from chameleon.viewer.backend.models.abstract_model import (
    DEFAULT_IMAGE_CFG_IMAGE,
    DEFAULT_IMAGE_CFG_TEXT,
    DEFAULT_MULTIMODAL_CFG_IMAGE,
    DEFAULT_MULTIMODAL_CFG_TEXT,
    AbstractMultimodalGenerator,
    MixedSequenceType,
    StreamingImage,
)
from chameleon.viewer.backend.models.chameleon_local import (
    ChameleonForwardMixin,
    ChameleonTokenizationMixin,
)
from chameleon.viewer.backend.utils import get_logger

logger = get_logger(__name__)

START = "START"

T = TypeVar("T")


def find_any(queue_by_id: dict[str, list]) -> str | None:
    for candidate_queue_id, candidate_queue in queue_by_id.items():
        if len(candidate_queue) > 0:
            return candidate_queue_id
    return None


class RedisQueue:
    def __init__(self, redis_client: redis.Redis, name: str, interval: float = 0.1):
        self.redis_client = redis_client
        self.name = name
        self.interval = interval
        self.lock = redis.lock.Lock(redis_client, f"lock_for_{name}")

    def reset(self):
        self.redis_client.set(self.name, json.dumps({}))
        try:
            self.lock.release()
        except redis.lock.LockError:
            pass

    def size(self) -> int:
        maybe_queue_by_id = self.redis_client.get(self.name)
        if maybe_queue_by_id is None:
            return 0
        else:
            return len(json.loads(maybe_queue_by_id))

    def clear(self, queue_id: str):
        with self.lock:
            maybe_queue_by_id = self.redis_client.get(self.name)
            if maybe_queue_by_id is None:
                queue_by_id: dict[str, list] = {}
            else:
                queue_by_id: dict[str, list] = json.loads(maybe_queue_by_id)
            queue_by_id[queue_id] = []
            self.redis_client.set(self.name, json.dumps(queue_by_id))

    def put(self, queue_id: str, value: T):
        logger.debug(
            "Thread %s: Starting PUT(%s) for %s",
            threading.get_ident(),
            self.name,
            queue_id,
        )
        with self.lock:
            maybe_queue_by_id = self.redis_client.get(self.name)
            if maybe_queue_by_id is None:
                queue_by_id: dict[str, list[T]] = {}
            else:
                queue_by_id: dict[str, list[T]] = json.loads(maybe_queue_by_id)

            if queue_id not in queue_by_id:
                queue_by_id[queue_id] = []
            queue_by_id[queue_id] = [value] + queue_by_id[queue_id]
            self.redis_client.set(self.name, json.dumps(queue_by_id))

        logger.debug(
            "Thread %s: Finished PUT(%s) for %s",
            threading.get_ident(),
            self.name,
            queue_id,
        )

    def get(self, queue_id: str | None) -> tuple[str, T]:
        """
        Get the next value in the queue.

        if queue_id is None, will get a value from any queue

        if queue_id is not none, will wait to get a value from a specific queue
        """
        logger.debug(
            "Thread %s: Starting GET(%s) for %s",
            threading.get_ident(),
            self.name,
            queue_id,
        )
        while True:
            with self.lock:
                # Initialization hasn't happened, so wait for it to happen
                maybe_queue_by_id = self.redis_client.get(self.name)
                if maybe_queue_by_id is None:
                    continue
                queue_by_id: dict[str, list[T]] = json.loads(maybe_queue_by_id)
                if queue_id is None:
                    queue_id = find_any(queue_by_id)

                # Ensure a queue_id was found or that it already existed
                if queue_id is not None and queue_id in queue_by_id:
                    queue = queue_by_id[queue_id]
                    if len(queue) == 0:
                        continue
                    value = queue.pop(-1)
                    # queue is mutated and queue_by_id references it, so this works
                    self.redis_client.set(self.name, json.dumps(queue_by_id))
                    logger.debug(
                        "Thread %s: Finished GET(%s) for %s",
                        threading.get_ident(),
                        self.name,
                        queue_id,
                    )
                    return queue_id, value
            time.sleep(self.interval)


class AsyncRedisQueue:
    def __init__(
        self, redis_client: async_redis.Redis, name: str, interval: float = 0.1
    ) -> None:
        self.redis_client = redis_client
        self.name = name
        self.interval = interval
        self.lock = async_redis.lock.Lock(redis_client, f"lock_for_{name}")

    async def reset(self):
        await self.redis_client.set(self.name, json.dumps({}))
        try:
            await self.lock.release()
        except async_redis.lock.LockError:
            pass

    async def size(self) -> int:
        maybe_queue_by_id = await self.redis_client.get(self.name)
        if maybe_queue_by_id is None:
            return 0
        else:
            return len(json.loads(maybe_queue_by_id))

    async def clear(self, queue_id: str):
        logger.debug(
            "ASYNC Thread %s: Starting CLEAR(%s) for %s",
            threading.get_ident(),
            self.name,
            queue_id,
        )
        async with self.lock:
            maybe_queue_by_id = await self.redis_client.get(self.name)
            if maybe_queue_by_id is None:
                queue_by_id: dict[str, list] = {}
            else:
                queue_by_id: dict[str, list] = json.loads(maybe_queue_by_id)
            queue_by_id[queue_id] = []
            await self.redis_client.set(self.name, json.dumps(queue_by_id))

        logger.debug(
            "ASYNC Thread %s: Finished CLEAR(%s) for %s",
            threading.get_ident(),
            self.name,
            queue_id,
        )

    async def put(self, queue_id: str, value: T):
        logger.debug(
            "ASYNC Thread %s: Starting PUT(%s) for %s",
            threading.get_ident(),
            self.name,
            queue_id,
        )

        async with self.lock:
            maybe_queue_by_id = await self.redis_client.get(self.name)
            if maybe_queue_by_id is None:
                queue_by_id: dict[str, list[T]] = {}
            else:
                queue_by_id: dict[str, list[T]] = json.loads(maybe_queue_by_id)

            if queue_id not in queue_by_id:
                queue_by_id[queue_id] = []
            queue_by_id[queue_id] = [value] + queue_by_id[queue_id]
            await self.redis_client.set(self.name, json.dumps(queue_by_id))

        logger.debug(
            "ASYNC Thread %s: Finished PUT(%s) for %s",
            threading.get_ident(),
            self.name,
            queue_id,
        )

    async def get(self, queue_id: str | None):
        """
        Get the next value in the queue.

        if queue_id is None, will get a value from any queue

        if queue_id is not none, will wait to get a value from a specific queue
        """
        logger.debug(
            "ASYNC Thread %s: Starting GET(%s) for %s",
            threading.get_ident(),
            self.name,
            queue_id,
        )
        while True:
            async with self.lock:
                maybe_queue_by_id = await self.redis_client.get(self.name)
                if maybe_queue_by_id is None:
                    continue
                queue_by_id: dict[str, list[T]] = json.loads(maybe_queue_by_id)
                if queue_id is None:
                    queue_id = find_any(queue_by_id)

                # Ensure a queue_id was found or that it already existed
                if queue_id is not None and queue_id in queue_by_id:
                    queue: list = queue_by_id[queue_id]
                    if len(queue) == 0:
                        continue
                    value = queue.pop(-1)
                    # queue is mutated and queue_by_id references it, so this works
                    await self.redis_client.set(self.name, json.dumps(queue_by_id))
                    logger.debug(
                        "ASYNC Thread %s: Finished GET(%s) for %s",
                        threading.get_ident(),
                        self.name,
                        queue_id,
                    )
                    return queue_id, value
            await asyncio.sleep(self.interval)


class AsyncRedisCounter:
    def __init__(self, redis_client: async_redis.Redis, name: str) -> None:
        self.redis_client = redis_client
        self.name = name
        self.lock = async_redis.lock.Lock(redis_client, f"lock_for_{name}")

    async def reset(self) -> int:
        try:
            await self.lock.release()
        except async_redis.lock.LockError:
            pass
        await self.redis_client.set(self.name, 0)

    async def add(self, n: int) -> int:
        async with self.lock:
            current_val = await self.redis_client.get(self.name)
            if current_val is None:
                current_val = 0
            else:
                current_val = int(current_val)
            new_val = current_val + n
            await self.redis_client.set(self.name, new_val)
            return new_val

    async def sub(self, n: int) -> int:
        async with self.lock:
            current_val = await self.redis_client.get(self.name)
            if current_val is None:
                raise ValueError("Invalid sub counter when counter does not exist")
            current_val = int(current_val)
            if current_val <= 0:
                raise ValueError("Invalid sub counter to counter that is already zero")
            new_val = current_val - n
            await self.redis_client.set(self.name, new_val)
            return new_val

    async def count(self) -> int:
        value = await self.redis_client.get(self.name)
        if value is None:
            return 0
        else:
            return int(value)


def distributed_workers(
    model_args: dict,
    master_address: str,
    master_port: str,
    world_size: int,
    rank: int,
    redis_port: int,
    worker_queues: dict[int, multiprocessing.Queue],
) -> None:
    redis_client = redis.Redis("redis", redis_port)
    request_queue = RedisQueue(redis_client, "request")
    response_queue = RedisQueue(redis_client, "response")

    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = str(master_port)

    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    assert rank == torch.distributed.get_rank()

    torch.cuda.set_device(rank)

    is_coord = rank == 0

    worker = ChameleonWorker(
        rank=rank,
        model_path=model_args["model_path"],
        tokenizer_path=model_args["tokenizer_path"],
        additional_eos_tokens=model_args["additional_eos_tokens"],
    )
    worker_id = id(worker)
    logger.info("Rank %s, master_port=%s worker=%s", rank, master_port, worker_id)

    step = 0
    while True:
        step += 1
        redis_client.set(f"status_rank_{rank}", "Pre-coordinator sync")
        if is_coord:
            distributed_objs = [request_queue.get(None)]
            logger.info("Objects from queue: %s", distributed_objs)
            for worker_rank in range(1, world_size):
                worker_message = {"message": START, "src": rank, "dst": worker_rank}
                logger.info("Rank %s Sending: %s", rank, worker_message)
                worker_queues[worker_rank].put(worker_message)
        else:
            distributed_objs = [None]
            logger.info("Rank %s worker %s waiting for rank 0", rank, worker_id)
            message_from_rank_0 = worker_queues[rank].get()
            logger.info(
                "Received message from rank 0 in rank %s: %s", rank, message_from_rank_0
            )
            if message_from_rank_0["message"] != START:
                raise ValueError(
                    f"Unexpected message from rank 0: {message_from_rank_0['message']}"
                )
        redis_client.set(f"status_rank_{rank}", "Post-coordinator sync")

        try:
            logger.info(
                "Broadcast Starting: Rank %s, worker %s, step %s",
                rank,
                worker_id,
                step,
            )
            redis_client.set(f"status_rank_{rank}", "Pre-torch sync")
            torch.distributed.broadcast_object_list(distributed_objs, src=0)
            redis_client.set(f"status_rank_{rank}", "Post-torch sync")
            logger.info(
                "Broadcast Complete: Rank %s, worker %s, step %s",
                rank,
                worker_id,
                step,
            )
        except RuntimeError as e:
            logger.error(
                "Rank %s, worker %s, step %s, Error detected in torch broadcast: %s",
                rank,
                worker_id,
                step,
                str(e),
            )
            raise

        logger.info("rank %s, objs %s", rank, distributed_objs)
        queue_id, data = distributed_objs[0]
        mode = data.pop("mode")
        request_id = data.pop("request_id")
        assert queue_id == request_id
        tokenized_prompt = data.pop("tokenized_prompt")
        try:
            match mode:
                case WSMessageType.GENERATE_TEXT:
                    generator_fn = partial(
                        worker._generate_text_streaming, tokenized_prompt, **data
                    )
                case WSMessageType.GENERATE_IMAGE:
                    generator_fn = partial(
                        worker._generate_image_streaming, tokenized_prompt, **data
                    )
                case WSMessageType.GENERATE_MULTIMODAL:
                    generator_fn = partial(
                        worker._generate_multimodal_streaming, tokenized_prompt, **data
                    )
                case _:
                    logger.error(
                        "Encountered unknown mode, crashing the program: %s", mode
                    )
                    response_queue.put(
                        queue_id, {"error": True, "final": True, "message": mode}
                    )
                    raise ValueError("Unknown mode")
            logger.info("Rank: %s, Processing request: %s", rank, request_id)
            i = 0
            redis_client.set(f"status_rank_{rank}", "Pre-generate")
            for output in generator_fn():
                i += 1
                if is_coord:
                    response = {"final": False, "output": output, "error": False}
                    logger.info(
                        "Rank: %s, Adding to response queue: %.100s",
                        rank,
                        response,
                    )
                    redis_client.set(f"status_rank_{rank}", f"Generate Pre Put {i}")
                    response_queue.put(queue_id, response)
                    redis_client.set(f"status_rank_{rank}", f"Generate Post Put {i}")
                else:
                    redis_client.set(f"status_rank_{rank}", f"Generate {i}")
                redis_client.set(f"step_on_rank_{rank}", i)
            redis_client.set(f"status_rank_{rank}", "Post-generate")
            if is_coord:
                logger.info("Rank: %s, Adding final result to output queue", rank)
                response_queue.put(queue_id, {"final": True, "error": False})
        except torch.cuda.OutOfMemoryError as e:
            logger.error("Encountered OOM, crashing the program: %s", e)
            response_queue.put(
                queue_id, {"error": True, "final": True, "message": str(e)}
            )
            crash_program()
        except RuntimeError as e:
            message = str(e)
            if "CUDA" in message:
                logger.error("Encountered CUDA error, crashing the program: %s", e)
                response_queue.put(
                    queue_id, {"error": True, "final": True, "message": str(e)}
                )
                crash_program()
            else:
                logger.error(
                    "Encountered unexpected runtime error, crashing the program: %s %s",
                    e,
                    traceback.format_exc(),
                )
                response_queue.put(
                    queue_id, {"error": True, "final": True, "message": str(e)}
                )
                crash_program()
        except Exception as e:
            logger.error(
                "Encountered unexpected exception: %s %s",
                str(e),
                traceback.format_exc(),
            )
            response_queue.put(
                queue_id, {"error": True, "final": True, "message": str(e)}
            )
            crash_program()


class ChameleonWorker(ChameleonForwardMixin):
    def __init__(
        self,
        *,
        rank: int,
        model_path: str,
        tokenizer_path: str,
        additional_eos_tokens: list[str] | None,
    ) -> None:
        self.rank = rank
        self.model_path = model_path
        self.additional_eos_tokens = additional_eos_tokens
        torch.set_default_device(f"cuda:{rank}")
        self.model = load_model(model_path, rank)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab = VocabInfo(json.load(open(tokenizer_path))["model"]["vocab"])
        logger.info(
            "Rank: %s, Model loaded in worker_obj: %s",
            rank,
            id(self),
        )


def crash_program() -> None:
    logger.error(
        "Crashing the program as instructed, likely due to distributed worker failures"
    )
    sys.exit(1)


class ChameleonDistributedGenerator(AbstractMultimodalGenerator, ChameleonTokenizationMixin):
    def __init__(
        self,
        *,
        world_size: int,
        model_path: str,
        master_port: int,
        tokenizer_path: str,
        vqgan_config_path: str,
        vqgan_ckpt_path: str | None = None,
        master_address: str = "0.0.0.0",
        additional_eos_tokens: list[str] | None = None,
        redis_port: int | None = None,
    ) -> None:
        self.master_port = master_port
        self.master_address = master_address
        self.additional_eos_tokens = additional_eos_tokens
        logger.info("Loading tokenizer...")
        tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab = VocabInfo(json.load(open(tokenizer_path))["model"]["vocab"])

        logger.info("Loading VQGAN...")
        self.image_tokenizer = ImageTokenizer(vqgan_config_path, vqgan_ckpt_path)
        self.redis_port = redis_port
        self.redis_pool = async_redis.ConnectionPool.from_url(
            f"redis://redis:{redis_port}"
        )
        self.redis_client = async_redis.Redis.from_pool(self.redis_pool)
        self.request_queue = AsyncRedisQueue(self.redis_client, "request")
        self.response_queue = AsyncRedisQueue(self.redis_client, "response")
        self.worker_queues: dict[int, multiprocessing.Queue] = {
            rank: multiprocessing.Queue() for rank in range(world_size)
        }
        self.procs: list[multiprocessing.Process] = []
        model_args = {
            "model_path": model_path,
            "master_address": master_address,
            "master_port": master_port,
            "tokenizer_path": tokenizer_path,
            "additional_eos_tokens": additional_eos_tokens,
        }
        logger.info("Launching paralle model with world_size=%s", world_size)
        for i in range(world_size):
            proc = multiprocessing.Process(
                target=distributed_workers,
                args=(
                    model_args,
                    master_address,
                    master_port,
                    world_size,
                    i,
                    self.redis_port,
                    self.worker_queues,
                ),
                daemon=True,
            )
            self.procs.append(proc)
            proc.start()

    def check_error(self, output: dict) -> None:
        if output["error"]:
            import sys
            print(f"check_error({output})", file=sys.stderr)
            self.kill_procs()
            logger.error(
                "COORDINATOR: Encountered error in managed processes, exiting: %s",
                output,
            )
            crash_program()

    def __del__(self) -> None:
        self.kill_procs(error=False)

    def kill_procs(self, error: bool = True) -> None:
        if error:
            log_fn = logger.error
        else:
            log_fn = logger.info
        log_fn("Error encountered, killing worker procs: %s", self.procs)
        for p in self.procs:
            try:
                log_fn("Killing: %s", p)
                p.kill()
            except:
                log_fn("Encountered issue killing process and ignoring: %s", p)

    # ALLOW_ANY(get_next_output.return)
    async def get_next_output(self, request_id: str) -> Any:
        logger.info("Waiting for response for request_id=%s", request_id)
        queue_id, output = await self.response_queue.get(request_id)
        assert queue_id == request_id
        return output

    async def generate_text_streaming(
        self,
        prompt: MixedSequenceType,
        max_gen_tokens: int = 256,
        temp: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
        debug: dict | None = None,
    ) -> Generator[str, None, None]:
        tokenized_prompt = self.tokens_from_inputs(prompt)
        request_id = f"request_{random.randint(100_000, 200_000)}"
        if seed is None:
            seed = random.randint(1, 2048)
            if debug is not None:
                debug["seed"] = seed
        if len(tokenized_prompt) > (4096 - 3):
            yield "ERROR: Your input exceeds the model's context length of 4096. Note that images consume 1024 tokens whether in input or output."
            return
        assert not isinstance(tokenized_prompt, torch.Tensor)
        request = {
            "mode": WSMessageType.GENERATE_TEXT.value,
            "request_id": request_id,
            "tokenized_prompt": tokenized_prompt,
            "max_gen_tokens": max_gen_tokens,
            "temp": temp,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
        }
        logger.info(
            "Sending request_id=%s: %s",
            request_id,
            request,
        )
        await asyncio.gather(
            self.request_queue.clear(request_id),
            self.response_queue.clear(request_id),
        )
        logger.info("Cleared request/response queue for %s", request_id)
        await self.request_queue.put(request_id, request)
        logger.info("Sent request to coordinator %s", request_id)
        try:
            while True:
                output = await self.get_next_output(request_id)
                logger.info("Received response for %s", request_id)
                self.check_error(output)
                if output["final"]:
                    break

                n_outs = len(output["output"])
                if n_outs != 1:
                    logger.error(
                        "Encountered unexpected number of %s arguments in: %s",
                        n_outs,
                        output["output"],
                    )
                tokens = output["output"]
                assert not isinstance(tokens, torch.Tensor)
                logger.info("output info: type=%s, value=%.20s", type(tokens), tokens)
                yield self.tokenizer.decode(tokens)
        finally:
            logger.info("Cleaning up queues in request_id=%s", request_id)
            await asyncio.gather(
                self.request_queue.clear(request_id),
                self.response_queue.clear(request_id),
            )
            logger.info("Completed cleaning for request_id=%s", request_id)

    async def generate_image_streaming(
        self,
        prompt: MixedSequenceType,
        temp: float = 1.0,
        top_p: float = 0.8,
        cfg_image_weight: float = DEFAULT_IMAGE_CFG_IMAGE,
        cfg_text_weight: float = DEFAULT_IMAGE_CFG_TEXT,
        yield_every_n: int = 32,
        debug: dict | None = None,
        seed: int | None = None,
    ) -> Generator[StreamingImage, None, None]:
        tokenized_prompt = self.tokens_from_inputs(prompt)
        tokenized_prompt.append(self.vocab.begin_image)
        assert not isinstance(tokenized_prompt, torch.Tensor)
        request_id = f"request_{random.randint(100_000, 200_000)}"
        if seed is None:
            seed = random.randint(1, 2048)
            if debug is not None:
                debug["seed"] = seed
        if len(tokenized_prompt) > (4096 - 3 - 1024):
            yield "ERROR: Your input exceeds the model's context length of 4096. Note that images consume 1024 tokens whether in input or output."
            return
        request = {
            "mode": WSMessageType.GENERATE_IMAGE.value,
            "request_id": request_id,
            "tokenized_prompt": tokenized_prompt,
            "cfg_image_weight": cfg_image_weight,
            "cfg_text_weight": cfg_text_weight,
            "yield_every_n": yield_every_n,
            "temp": temp,
            "top_p": top_p,
            "seed": seed,
        }
        logger.info(
            "Sending request_id=%s: %s",
            request_id,
            request,
        )
        await asyncio.gather(
            self.request_queue.clear(request_id),
            self.response_queue.clear(request_id),
        )
        logger.info("Cleared request/response queue for %s", request_id)
        await self.request_queue.put(request_id, request)
        logger.info("Sent request to coordinator %s", request_id)
        try:
            while True:
                output = await self.get_next_output(request_id)
                logger.info("Received response for %s", request_id)
                self.check_error(output)
                if output["final"]:
                    break
                n_outs = len(output["output"])
                if n_outs != 2:
                    logger.error(
                        "Encountered unexpected number of %s arguments in: %s",
                        n_outs,
                        output["output"],
                    )
                tokens, final = output["output"]
                assert not isinstance(tokens, torch.Tensor)
                yield StreamingImage(
                    image=self.pillow_from_bpe_tokens(torch.tensor(tokens)), final=final
                )
        finally:
            logger.info("Cleaning up queues in request_id=%s", request_id)
            await asyncio.gather(
                self.request_queue.clear(request_id),
                self.response_queue.clear(request_id),
            )
            logger.info("Completed cleaning for request_id=%s", request_id)

    async def generate_multimodal_streaming(
        self,
        prompt: MixedSequenceType,
        temp: float = 1.0,
        top_p: float = 0.8,
        cfg_image_weight: float = DEFAULT_MULTIMODAL_CFG_IMAGE,
        cfg_text_weight: float = DEFAULT_MULTIMODAL_CFG_TEXT,
        yield_every_n: int = 32,
        max_gen_tokens: int = 4096,
        repetition_penalty: float = 1.2,
        suffix_tokens: list[str] | None = None,
        seed: int | None = None,
        debug: dict | None = None,
    ) -> Generator[MixedSequenceType, None, None]:
        tokenized_prompt = self.tokens_from_inputs(prompt, suffix_tokens=suffix_tokens)
        assert not isinstance(tokenized_prompt, torch.Tensor)
        request_id = f"request_{random.randint(100_000, 200_000)}"
        if seed is None:
            seed = random.randint(1, 2048)
            if debug is not None:
                debug["seed"] = seed
        if len(tokenized_prompt) > (4096 - 3):
            yield "ERROR: Your input exceeds the model's context length of 4096. Note that images consume 1024 tokens."
            return

        request = {
            "mode": WSMessageType.GENERATE_MULTIMODAL.value,
            "request_id": request_id,
            "tokenized_prompt": tokenized_prompt,
            "cfg_image_weight": cfg_image_weight,
            "cfg_text_weight": cfg_text_weight,
            "repetition_penalty": repetition_penalty,
            "yield_every_n": yield_every_n,
            "max_gen_tokens": max_gen_tokens,
            "temp": temp,
            "top_p": top_p,
            "seed": seed,
        }
        logger.info(
            "Sending request_id=%s: %s",
            request_id,
            request,
        )
        await asyncio.gather(
            self.request_queue.clear(request_id),
            self.response_queue.clear(request_id),
        )
        logger.info("Cleared request/response queue for %s", request_id)
        await self.request_queue.put(request_id, request)
        logger.info("Sent request to coordinator %s", request_id)
        try:
            while True:
                output = await self.get_next_output(request_id)
                logger.info("Received response for %s", request_id)
                self.check_error(output)
                if output["final"]:
                    break
                n_outs = len(output["output"])
                if n_outs != 3:
                    logger.error(
                        "Encountered unexpected number of %s arguments in: %s",
                        n_outs,
                        output["output"],
                    )
                token_type, tokens, image_is_final = output["output"]
                assert not isinstance(tokens, torch.Tensor)
                match token_type:
                    case "TEXT":
                        yield self.tokenizer.decode(tokens)
                    case "IMAGE":
                        yield StreamingImage(
                            image=self.pillow_from_bpe_tokens(torch.tensor(tokens)),
                            final=image_is_final,
                        )
                    case _:
                        raise ValueError("Unknown token type")
        finally:
            logger.info("Cleaning up queues in request_id=%s", request_id)
            await self.request_queue.clear(request_id)
            await self.response_queue.clear(request_id)
