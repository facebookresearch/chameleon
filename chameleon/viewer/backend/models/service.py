# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import base64
import io
import socket
import subprocess
import time
from functools import partial

import fastapi
import PIL
import pydantic
import redis.asyncio as async_redis
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, WebSocketException
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from chameleon.viewer.backend.data_types import (
    Content,
    ContentType,
    NoOptionsForComplete,
    NoOptionsForFull,
    NoOptionsForPartial,
    NoOptionsForQueueStatus,
    WSMessageType,
    WSMultimodalMessage,
)
from chameleon.viewer.backend.models.abstract_model import (
    AbstractMultimodalGenerator,
    StreamingImage,
)
from chameleon.viewer.backend.models.chameleon_distributed import AsyncRedisCounter
from chameleon.viewer.backend.utils import get_logger

logger = get_logger(__name__)


def nvidia_smi() -> str:
    return subprocess.check_output(["nvidia-smi"], text=True)


async def await_generate_message(websocket: WebSocket) -> WSMultimodalMessage:
    while True:
        rec_message = await websocket.receive_json()
        try:
            maybe_message = WSMultimodalMessage.parse_obj(rec_message)
        except pydantic.ValidationError:
            maybe_message = None
            logger.info("Got invalid message", maybe_message)
        if maybe_message is not None:
            return maybe_message


async def async_acquire_lock(
    *,
    websocket: WebSocket,
    counter: AsyncRedisCounter,
    lock: async_redis.lock.Lock,
    interval=0.1,
    status_interval=1,
    hostname: str | None = None,
):
    start = time.time()
    await counter.add(1)
    while True:
        acquired = await lock.acquire(blocking_timeout=interval)
        if acquired:
            break
        elapsed = time.time() - start
        if elapsed > status_interval:
            n_requests = await counter.count()
            message = WSMultimodalMessage(
                message_type=WSMessageType.QUEUE_STATUS,
                content=[
                    Content(
                        content_type=ContentType.TEXT,
                        content=f"n_requests={n_requests}",
                    )
                ],
                options=NoOptionsForQueueStatus(),
                debug_info={"hostname": hostname},
            ).dict()
            await websocket.send_json(message)
            start = time.time()
    await counter.sub(1)


COORDINATOR = "coordinator"


def web_app(
    generator: AbstractMultimodalGenerator,
    debug: bool = True,
    redis_port: int | None = None,
) -> FastAPI:
    app = FastAPI(debug=debug)
    if redis_port is None:
        redis_client = None
        redis_lock = None
        queue_counter = None
    else:
        redis_client = async_redis.Redis.from_url(f"redis://redis:{redis_port}")
        redis_lock = async_redis.lock.Lock(redis_client, COORDINATOR)
        queue_counter = AsyncRedisCounter(redis_client, "count_pending")
    hostname = socket.gethostname()

    @app.get("/api/2.0/status")
    def alive() -> dict:
        return {
            "status": "alive",
            "hostname": hostname,
            "nvidia-smi": nvidia_smi(),
        }

    @app.websocket("/ws/chameleon/v2/{client_id}")
    async def websocket_chameleon_v2(*, websocket: WebSocket, client_id: str):
        logger.info("Requested client_id: %s", client_id)
        await websocket.accept()
        logger.info("Client opened %s with generator id %s", client_id, id(generator))

        try:
            while True:
                generate_message = await await_generate_message(websocket)
                logger.info("Got generate message: %s", str(generate_message)[:300])
                parsed_prompt = []
                for c in generate_message.content:
                    match c.content_type:
                        case ContentType.TEXT:
                            parsed_prompt.append(c.content)
                        case ContentType.IMAGE:
                            image_parts = c.content.split(",", 1)
                            if len(image_parts) < 2:
                                logger.error(
                                    "Encountered invalid image: %s", image_parts
                                )
                                raise WebSocketException(
                                    code=fastapi.status.WS_1008_POLICY_VIOLATION,
                                    reason=f"Invalid image: {image_parts}",
                                )
                            image_data = image_parts[1]
                            base64_image = base64.b64decode(image_data)
                            image_file = io.BytesIO(base64_image)
                            parsed_prompt.append(PIL.Image.open(image_file))
                        case _:
                            raise ValueError("Unknown content type")
                logger.info("Prompt: %s", parsed_prompt)
                partial_outputs = []
                final_contents: list[Content] = []

                match generate_message.message_type:
                    case WSMessageType.GENERATE_TEXT:
                        output_generator = generator.generate_text_streaming
                    case WSMessageType.GENERATE_IMAGE:
                        output_generator = generator.generate_image_streaming
                    case WSMessageType.GENERATE_MULTIMODAL:
                        output_generator = generator.generate_multimodal_streaming
                    case _:
                        raise WebSocketException(
                            code=fastapi.status.WS_1008_POLICY_VIOLATION,
                            reason="Unknown message type",
                        )

                logger.info(
                    "Acquiring lock for client %s generation with options: %s",
                    client_id,
                    generate_message.options,
                )
                option_args = generate_message.options.dict()
                debug_info = {"hostname": hostname}
                del option_args["message_type"]
                output_generator = partial(
                    output_generator,
                    **option_args,
                    debug=debug_info,
                )
                if redis_lock is not None:
                    await async_acquire_lock(
                        websocket=websocket,
                        lock=redis_lock,
                        hostname=hostname,
                        counter=queue_counter,
                    )
                    await redis_client.set("has_lock", client_id)

                logger.info(
                    "Starting locked generation for client %s with options: %s",
                    client_id,
                    generate_message.options,
                )
                try:
                    async for output_token in output_generator(parsed_prompt):
                        if isinstance(output_token, str):
                            content_type = ContentType.TEXT
                            content = output_token
                            message_type = WSMessageType.PARTIAL_OUTPUT
                            options = NoOptionsForPartial()
                            partial_outputs.extend(output_token)
                        elif isinstance(output_token, StreamingImage):
                            content_type = ContentType.IMAGE
                            image = output_token.image
                            img_io = io.BytesIO()
                            image.save(img_io, format="png")
                            content = (
                                "data:image/png;base64,"
                                + base64.b64encode(img_io.getvalue()).decode()
                            )
                            if output_token.final:
                                message_type = WSMessageType.FULL_OUTPUT
                                options = NoOptionsForFull()
                            else:
                                message_type = WSMessageType.PARTIAL_OUTPUT
                                options = NoOptionsForPartial()

                            if output_token.final:
                                partial_outputs.append(output_token.image)
                        else:
                            raise ValueError(f"Invalid output_token: {output_token}")

                        message_content = Content(
                            content_type=content_type, content=content
                        )
                        match content_type:
                            case ContentType.TEXT:
                                final_contents.append(message_content)
                            case ContentType.IMAGE:
                                if message_type == WSMessageType.FULL_OUTPUT:
                                    final_contents.append(message_content)
                            case _:
                                pass

                        message = WSMultimodalMessage(
                            message_type=message_type,
                            content=[message_content],
                            options=options,
                            debug_info=debug_info,
                        ).dict()
                        await websocket.send_json(message)
                finally:
                    if redis_lock is not None:
                        logger.info(
                            "Attempting release of lock for client %s generation with options: %s",
                            client_id,
                            generate_message.options,
                        )
                        owned = await redis_lock.owned()
                        if owned:
                            await redis_client.set("has_lock", "")
                            try:
                                await redis_lock.release()
                            except async_redis.lock.LockError:
                                pass

                        logger.info(
                            "Released lock for client %s generation with options: %s",
                            client_id,
                            generate_message.options,
                        )
                await websocket.send_json(
                    WSMultimodalMessage(
                        message_type=WSMessageType.COMPLETE,
                        content=final_contents,
                        options=NoOptionsForComplete(),
                        debug_info=debug_info,
                    ).dict()
                )
        except WebSocketDisconnect:
            logger.info("Client disconnected %s", client_id)
        except ConnectionClosedError:
            logger.info("Client forced a close %s", client_id)
        except ConnectionClosedOK:
            logger.info("Connection closed ok %s", client_id)
        finally:
            if redis_lock is not None:
                logger.info("Checking for client holding lock: %s", client_id)
                owned = await redis_lock.owned()
                if owned:
                    try:
                        logger.info("Attempted to release owned lock: %s", client_id)
                        await redis_lock.release()
                    except async_redis.lock.LockError:
                        pass
                    await redis_client.set("has_lock", "")

    return app


def serve(
    model: AbstractMultimodalGenerator,
    host: str,
    port: int,
    debug: bool = True,
    redis_port: int | None = None,
) -> None:
    app = web_app(model, debug=debug, redis_port=redis_port)
    # TODO: convert this to a subprocess call so enable more
    # uvicorn features like multiple workers
    uvicorn.run(app, host=host, port=port)
