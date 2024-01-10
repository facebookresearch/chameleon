# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import base64
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import click
import torch
from flask import Flask, request
from flask_socketio import SocketIO

from chameleon.inference.chameleon import ChameleonInferenceModel, Options, TokenManager


@dataclass
class Request:
    room: str
    key: str
    options: dict[str, int | float | bool]
    prompt_ui: list[dict]


def convert_options(ui_options: dict) -> Options:
    txt = None
    if ui_options["enable-text"]:
        txt = Options.Text(
            repetition_penalty=ui_options["text-rep-penalty"],
            temp=ui_options["text-temp"],
            top_p=ui_options["text-top-p"],
        )
    img = None
    if ui_options["enable-image"]:
        img = Options.Image(
            cfg=Options.Image.CFG(
                guidance_scale_image=ui_options["img-cfg-gsimage"],
                guidance_scale_text=ui_options["img-cfg-gstext"],
            ),
            temp=ui_options["img-temp"],
            top_p=ui_options["img-top-p"],
        )
    return Options(
        max_seq_len=ui_options["max-seq-len"],
        max_gen_len=ui_options["max-gen-len"],
        seed=ui_options["seed"],
        txt=txt,
        img=img,
    )


class UIDecoder:
    class State(Enum):
        TXT = 1
        IMG = 2
        IMG_END = 3

    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.state = UIDecoder.State.TXT
        self.image_builder = []
        self.image_yield_every_n = 32
        self.image_has_updated = False

    def _image_progress(self) -> dict:
        self.image_has_updated = False
        png = self.token_manager.png_from_bpe_tokens(torch.cat(self.image_builder))
        return {
            "type": "image",
            "value": "data:image/png;base64," + base64.b64encode(png).decode(),
        }

    def next(self, gpu_token: torch.LongTensor) -> dict | None:
        if self.state == UIDecoder.State.TXT:
            cpu_tok = gpu_token.item()

            if cpu_tok == self.token_manager.vocab.begin_image:
                self.state = UIDecoder.State.IMG
                return {"type": "image_start"}

            return {
                "type": "text",
                "value": self.token_manager.tokenizer.decode([cpu_tok]),
            }

        elif self.state == UIDecoder.State.IMG:
            self.image_builder.append(gpu_token)
            self.image_has_updated = True
            if len(self.image_builder) == 1024:
                self.state = UIDecoder.State.IMG_END
            if len(self.image_builder) % self.image_yield_every_n == 0:
                return self._image_progress()

        elif self.state == UIDecoder.State.IMG_END:
            # assert gpu_token == end_image
            self.state = UIDecoder.State.TXT
            progress = self._image_progress() if self.image_has_updated else None
            self.image_builder = []
            return progress


@dataclass
class State:
    room_keys: dict[str, set[str]]
    pending_requests: list[Request]
    cond: threading.Condition

    def __enter__(self, *args, **kwargs):
        self.cond.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self.cond.__exit__(*args, **kwargs)
        return self


GlobalState = State(room_keys={}, pending_requests=[], cond=threading.Condition())

app = Flask(__name__)
socketio = SocketIO(app, max_http_buffer_size=16 * 1024 * 1024)


@app.route("/")
def index():
    with open(Path(__file__).parent / "miniviewer.html") as f:
        return f.read()


@socketio.on("disconnect")
def handle_disconnect():
    with GlobalState as state:
        try:
            del state.room_keys[request.sid]
        except KeyError:
            pass


@socketio.on("cancel")
def handle_cancel(key):
    with GlobalState as state:
        try:
            state.room_keys[request.sid].remove(key)
        except KeyError:
            pass


@socketio.on("generate")
def handle_generate(key, options, prompt_ui):
    with GlobalState as state:
        if request.sid not in state.room_keys:
            state.room_keys[request.sid] = set()
        state.room_keys[request.sid].add(key)
        state.pending_requests.append(Request(request.sid, key, options, prompt_ui))
        state.cond.notify_all()


def generation_thread(model: ChameleonInferenceModel):
    while True:
        with GlobalState as state:
            state.cond.wait_for(lambda: state.pending_requests)
            req = state.pending_requests.pop(0)

        start = time.time()
        ui_decoder = UIDecoder(model.token_manager)
        options = convert_options(req.options)

        if not options.txt:
            progress = ui_decoder.next(
                torch.tensor([model.token_manager.vocab.begin_image])
            )
            socketio.emit(
                "progress",
                {"key": req.key, **progress},
                room=req.room,
            )

        for token in model.stream(
            prompt_ui=req.prompt_ui,
            options=options,
        ):
            with GlobalState as state:
                if req.key not in state.room_keys.get(req.room, {}):
                    break

            if progress := ui_decoder.next(token.id):
                socketio.emit(
                    "progress",
                    {"key": req.key, **progress},
                    room=req.room,
                )

        timing = time.time() - start
        socketio.emit(
            "progress",
            {"key": req.key, "type": "done", "value": timing},
            room=req.room,
        )


def queue_position_thread():
    local_pending_requests = []
    while True:
        with GlobalState as state:
            state.cond.wait_for(
                lambda: local_pending_requests != state.pending_requests
            )
            local_pending_requests = state.pending_requests[:]

        for i, req in enumerate(local_pending_requests):
            progress = {
                "type": "queue",
                "key": req.key,
                "value": i + 1,
            }
            socketio.emit("progress", progress, room=req.room)


@click.command()
@click.option("--data-path", type=click.Path(), default="./data")
@click.option(
    "--model-size", type=click.Choice(["7b", "30b"], case_sensitive=False), default="7b"
)
def main(data_path, model_size):
    data_path = Path(data_path)

    model_path = str(data_path / "models" / model_size)
    tokenizer_path = str(data_path / "tokenizer/text_tokenizer.json")
    vqgan_cfg_path = str(data_path / "tokenizer/vqgan.yaml")
    vqgan_ckpt_path = str(data_path / "tokenizer/vqgan.ckpt")

    if not os.path.exists(model_path):
        raise ValueError(
            "Model not found. Did you run python -m chameleon.download_data {PRESIGNED_URL}"
        )

    cm3v2_inference_model = ChameleonInferenceModel(
        model_path, tokenizer_path, vqgan_cfg_path, vqgan_ckpt_path
    )
    threading.Thread(
        target=generation_thread,
        args=(cm3v2_inference_model,),
        daemon=True,
    ).start()
    threading.Thread(target=queue_position_thread, daemon=True).start()
    socketio.run(app, debug=False)


if __name__ == "__main__":
    main()
