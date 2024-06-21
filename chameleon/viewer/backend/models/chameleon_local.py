# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import io
import json
from typing import Generator

import PIL.Image
import torch
import transformers
from tokenizers import Tokenizer
from transformers import (
    MaxLengthCriteria,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)

from chameleon.inference.alignment import AlignPromptRight
from chameleon.inference.generation import ChameleonGenerator
from chameleon.inference.image_tokenizer import ImageTokenizer
from chameleon.inference.loader import load_model
from chameleon.inference.logits_processor import (
    AllowOnlyTokensAfterIndexLogitsProcessor,
    AllowOnlyTokensLogitsProcessor,
    InBatchInstructCFGLogitsProcessor,
)
from chameleon.inference.model_adapter import ChameleonModelAdapter
from chameleon.inference.stopping_criteria import StopOnEOS, StopOnEOSAfterBatchIndex
from chameleon.inference.token_selector import (
    MultinomialTokenSelector,
    ReplicatedInputTokenSelector,
)
from chameleon.inference.vocab import VocabInfo, VocabTranslation
from chameleon.viewer.backend.models.abstract_model import (
    DEFAULT_IMAGE_CFG_IMAGE,
    DEFAULT_IMAGE_CFG_TEXT,
    DEFAULT_MULTIMODAL_CFG_IMAGE,
    DEFAULT_MULTIMODAL_CFG_TEXT,
    AbstractMultimodalGenerator,
    MixedSequenceType,
    StreamingImage,
)
from chameleon.viewer.backend.utils import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    transformers.enable_full_determinism(seed, warn_only=True)


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


class ChameleonTokenizationMixin:
    def png_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> bytes:
        img = self.pillow_from_bpe_tokens(bpe_tokens)

        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        return img_io.getvalue()

    def pillow_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> PIL.Image.Image:
        image_tensor = VocabTranslation(self.vocab).convert_bpe2img(bpe_tokens)
        if image_tensor.shape[0] < 1024:
            padding = (
                torch.ones([1024 - image_tensor.shape[0]], dtype=int) * image_tensor[0]
            )
            image_tensor = torch.cat((image_tensor, padding)).unsqueeze(0)

        return self.image_tokenizer.pil_from_img_toks(image_tensor)

    def tokens_from_inputs(
        self,
        inputs: MixedSequenceType,
        suffix_tokens: list[str] | None = None,
    ) -> list[int]:
        tokens = [self.vocab.bos_id]
        for input_ in inputs:
            if isinstance(input_, str):
                tokens.extend(self.tokenizer.encode(input_.strip()).ids)
            elif isinstance(input_, PIL.Image.Image):
                tokens.append(self.vocab.begin_image)
                imgtoks = self.image_tokenizer.img_tokens_from_pil(input_)
                tokens.extend(VocabTranslation(self.vocab).convert_img2bp2(imgtoks))
                tokens.append(self.vocab.end_image)
            else:
                raise ValueError(f"Unknown input type: {type(input_)}")

        if suffix_tokens is not None:
            for t in suffix_tokens:
                tokens.extend(self.tokenizer.encode(t).ids)
        sanitized_tokens = []
        for t in tokens:
            if isinstance(t, torch.Tensor):
                sanitized_tokens.append(t.item())
            else:
                sanitized_tokens.append(t)
        return sanitized_tokens


class GeneratorWrapper:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)


class Decoder:
    def __init__(
        self,
        chameleon_generator: "ChameleonLocalGenerator",
        input_ids: list[int],
    ):
        ...

    def __next__(self) -> tuple[list[int], dict | None, type["Decoder"] | None]:
        ...


class TextDecoder(Decoder):
    def __init__(
        self,
        chameleon_generator: "ChameleonLocalGenerator",
        input_ids: list[int],
        *,
        temp: float,
        top_p: float,
        max_seq_len: int,
        # TODO: Propagage setting upwards
        repetition_penalty: float,
        **kwargs,
    ):
        self.chameleon_generator = chameleon_generator
        assert chameleon_generator.vocab.eos_id is not None

        stopping_criteria = [
            StopOnEOS(chameleon_generator.vocab.eos_id),
            MaxLengthCriteria(max_seq_len),
        ]
        if chameleon_generator.additional_eos_tokens is not None:
            for token in chameleon_generator.additional_eos_tokens:
                stopping_criteria.append(
                    StopOnEOSAfterBatchIndex(
                        chameleon_generator.tokenizer.token_to_id(token), [len(input_ids)]
                    )
                )

        logits_processors = [
            AllowOnlyTokensLogitsProcessor(
                chameleon_generator.vocab.text_tokens
                + [chameleon_generator.vocab.eos_id, chameleon_generator.vocab.begin_image]
            ),
            # Don't allow any more images near the end since there isn't enough room
            AllowOnlyTokensAfterIndexLogitsProcessor(
                chameleon_generator.vocab.text_tokens + [chameleon_generator.vocab.eos_id],
                # TODO: Calculate exact
                1024 * 3 - 3,
            ),
            RepetitionPenaltyLogitsProcessor(repetition_penalty),
            TemperatureLogitsWarper(temp),
            TopPLogitsWarper(top_p),
        ]

        self.gen = ChameleonGenerator(
            model=ChameleonModelAdapter(chameleon_generator.model, max_seq_len=max_seq_len),
            input_ids=[input_ids],
            stopping_criteria=stopping_criteria,
            logits_processors=logits_processors,
        )
        for _ in range(len(input_ids)):
            next(self.gen)

    def __next__(self) -> tuple[list[int], dict | None, type[Decoder] | None]:
        gpu_tok = next(self.gen).id.item()
        cpu_tok = gpu_tok
        if cpu_tok == self.chameleon_generator.vocab.begin_image:
            # return "TEXT", [cpu_tok], [], False, ImageDecoder
            raise StopIteration()

        return (
            "TEXT",
            [cpu_tok],
            [cpu_tok],
            False,
            None,
        )


class ImageDecoder(Decoder):
    def __init__(
        self,
        chameleon_generator: "ChameleonLocalGenerator",
        input_ids: list[int],
        *,
        cfg_image_weight: float,
        cfg_text_weight: float,
        temp: float,
        top_p: float,
        yield_every_n: int,
        **kwargs,
    ):
        self.yield_every_n = yield_every_n
        self.chameleon_generator = chameleon_generator
        logits_processors = [
            InBatchInstructCFGLogitsProcessor(cfg_text_weight, cfg_image_weight),
            AllowOnlyTokensLogitsProcessor(chameleon_generator.vocab.image_tokens),
            TemperatureLogitsWarper(temp),
            TopPLogitsWarper(top_p),
        ]

        image_conditioned_allowed = set(chameleon_generator.vocab.image_tokens) | {
            chameleon_generator.vocab.bos_id,
            chameleon_generator.vocab.begin_image,
            chameleon_generator.vocab.end_image,
        }

        full_conditioned = input_ids
        image_conditioned = [
            in_id for in_id in input_ids if in_id in image_conditioned_allowed
        ]
        unconditioned = [
            chameleon_generator.vocab.bos_id,
            chameleon_generator.vocab.begin_image,
        ]

        self.gen = ChameleonGenerator(
            model=ChameleonModelAdapter(
                chameleon_generator.model, max_seq_len=len(input_ids) + 1024
            ),
            input_ids=[full_conditioned, image_conditioned, unconditioned],
            logits_processors=logits_processors,
            alignment=AlignPromptRight(chameleon_generator.vocab.pad_id),
            token_selector=ReplicatedInputTokenSelector(
                MultinomialTokenSelector(), n=3
            ),
        )
        for _ in range(len(input_ids)):
            next(self.gen)
        self.image_builder: list[torch.LongTensor] = []
        self.gpu_tok_batch: list[torch.LongTensor] = []

    def __next__(self) -> tuple[list[int], dict | None, type[Decoder] | None]:
        while True:
            gpu_tok = next(self.gen)
            gpu_tok = torch.chunk(gpu_tok, chunks=3, dim=0)[0]

            self.image_builder.append(gpu_tok)
            self.gpu_tok_batch.append(gpu_tok)

            if len(self.image_builder) == 1024:
                return (
                    "IMAGE",
                    torch.tensor(self.gpu_tok_batch).tolist()
                    + [self.chameleon_generator.vocab.end_image],
                    torch.tensor(self.image_builder).tolist(),
                    True,
                    TextDecoder,
                )
            elif len(self.image_builder) % self.yield_every_n == 0:
                cpu_toks = torch.tensor(self.gpu_tok_batch).tolist()
                self.gpu_tok_batch = []

                return (
                    "IMAGE",
                    cpu_toks,
                    torch.tensor(self.image_builder).tolist(),
                    False,
                    None,
                )


class ChameleonForwardMixin:
    @torch.inference_mode()
    def _generate_text_streaming(
        self,
        input_ids: list[int],
        max_gen_tokens: int = 256,
        temp: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
    ) -> Generator[str, None, None]:
        if seed is not None:
            set_seed(seed)
            logger.info(
                "Rank: %s, set seed: %s",
                get_rank(),
                seed,
            )

        logits_processors = [
            # Only allow text tokens and end-of-sequence.
            AllowOnlyTokensLogitsProcessor(
                self.vocab.text_tokens + [self.vocab.eos_id]
            ),
            # Don't allow the first token to be end-of-sequence.
            # DisallowTokensAtIndexLogitProcessor([self.vocab.eos_id], len()),
            RepetitionPenaltyLogitsProcessor(repetition_penalty),
            TemperatureLogitsWarper(temp),
            TopPLogitsWarper(top_p),
        ]

        stopping_criteria = [
            StopOnEOS(self.vocab.eos_id),
            MaxLengthCriteria(len(input_ids) + max_gen_tokens),
        ]
        if self.additional_eos_tokens is not None:
            for token in self.additional_eos_tokens:
                stopping_criteria.append(
                    StopOnEOSAfterBatchIndex(
                        self.tokenizer.token_to_id(token), [len(input_ids)]
                    )
                )
        for tok in ChameleonGenerator(
            model=ChameleonModelAdapter(
                self.model,
                max_seq_len=len(input_ids) + max_gen_tokens,
            ),
            input_ids=[input_ids],
            stopping_criteria=stopping_criteria,
            logits_processors=logits_processors,
        ):
            yield tok.tolist()

    @torch.inference_mode()
    def _generate_batched_text_streaming(
        self,
        batch: list[list[int]],
        max_gen_tokens: int = 256,
        temp: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
    ) -> Generator[list[str], None, None]:
        if seed is not None:
            set_seed(seed)
        logits_processors = [
            # Only allow text tokens and end-of-sequence.
            AllowOnlyTokensLogitsProcessor(
                self.vocab.text_tokens + [self.vocab.eos_id]
            ),
            # Don't allow the first token to be end-of-sequence.
            # DisallowTokensAtIndexLogitProcessor([self.vocab.eos_id], len()),
            RepetitionPenaltyLogitsProcessor(repetition_penalty),
            TemperatureLogitsWarper(temp),
            TopPLogitsWarper(top_p),
        ]

        max_batch_size = max(len(p) for p in batch)
        stopping_criteria = [
            StopOnEOS(self.vocab.eos_id),
            MaxLengthCriteria(max_batch_size + max_gen_tokens),
        ]
        if self.additional_eos_tokens is not None:
            for token in self.additional_eos_tokens:
                stopping_criteria.append(
                    StopOnEOSAfterBatchIndex(
                        self.tokenizer.token_to_id(token), [len(x) for x in batch]
                    )
                )
        for tok in ChameleonGenerator(
            model=ChameleonModelAdapter(
                self.model,
                max_seq_len=max_batch_size + max_gen_tokens,
            ),
            input_ids=batch,
            stopping_criteria=stopping_criteria,
            logits_processors=logits_processors,
        ):
            yield tok.unsqueeze(1).tolist()

    @torch.inference_mode()
    def _generate_image_streaming(
        self,
        tokenized_prompt: list[int],
        temp: float = 1.0,
        top_p: float = 0.8,
        cfg_image_weight: float = DEFAULT_IMAGE_CFG_IMAGE,
        cfg_text_weight: float = DEFAULT_IMAGE_CFG_TEXT,
        yield_every_n: int = 32,
        seed: int | None = None,
    ) -> Generator[tuple[list[int], bool], None, None]:
        if seed is not None:
            set_seed(seed)
            logger.info(
                "Rank: %s, set seed: %s",
                get_rank(),
                seed,
            )

        decoder = ImageDecoder(
            self,
            tokenized_prompt,
            cfg_image_weight=cfg_image_weight,
            cfg_text_weight=cfg_text_weight,
            temp=temp,
            top_p=top_p,
            yield_every_n=yield_every_n,
        )

        for _, _, frontend_tokens, is_final, next_decoder in GeneratorWrapper(decoder):
            if next_decoder is not None:
                break

            yield torch.tensor(frontend_tokens).tolist(), is_final

    @torch.inference_mode()
    def _generate_multimodal_streaming(
        self,
        input_ids: list[int],
        temp: float = 1.0,
        top_p: float = 0.8,
        cfg_image_weight: float = DEFAULT_MULTIMODAL_CFG_IMAGE,
        cfg_text_weight: float = DEFAULT_MULTIMODAL_CFG_TEXT,
        yield_every_n: int = 32,
        max_gen_tokens: int = 4096,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
    ) -> Generator[tuple[str, list[int], bool], None, None]:
        if seed is not None:
            set_seed(seed)
            logger.info(
                "Rank: %s, set seed: %s",
                get_rank(),
                seed,
            )
        max_seq_len = min(len(input_ids) + max_gen_tokens, 4096)
        gen_wrapper = GeneratorWrapper(
            TextDecoder(
                self,
                input_ids,
                temp=temp,
                top_p=top_p,
                max_seq_len=max_seq_len,
                repetition_penalty=repetition_penalty,
            )
        )

        for (
            message_type,
            cpu_toks,
            frontend_tokens,
            is_final,
            next_decoder,
        ) in gen_wrapper:
            input_ids.extend(cpu_toks)
            if len(frontend_tokens) > 0:
                yield message_type, frontend_tokens, is_final
            if next_decoder is not None:
                gen_wrapper.gen = next_decoder(
                    self,
                    input_ids,
                    temp=temp,
                    top_p=top_p,
                    max_seq_len=max_seq_len,
                    cfg_image_weight=cfg_image_weight,
                    cfg_text_weight=cfg_text_weight,
                    yield_every_n=yield_every_n,
                    repetition_penalty=repetition_penalty,
                )


class ChameleonLocalGenerator(
    AbstractMultimodalGenerator, ChameleonForwardMixin, ChameleonTokenizationMixin
):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        vqgan_config_path: str,
        vqgan_ckpt_path: str | None = None,
        additional_eos_tokens: list[str] | None = None,
    ) -> None:
        super().__init__()
        logger.info("Loading model...")
        self.model = load_model(model_path)
        self.additional_eos_tokens = additional_eos_tokens

        logger.info("Loading tokenizer...")
        tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab = VocabInfo(json.load(open(tokenizer_path))["model"]["vocab"])

        logger.info("Loading VQGAN...")
        self.image_tokenizer = ImageTokenizer(vqgan_config_path, vqgan_ckpt_path)

    @torch.inference_mode()
    def generate_batched_text(
        self,
        prompts: list[MixedSequenceType],
        max_gen_tokens: int = 256,
        temp: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
    ) -> list[str]:
        outputs = [""] * len(prompts)
        for vals in self.generate_batched_text_streaming(
            prompts,
            max_gen_tokens=max_gen_tokens,
            temp=temp,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ):
            for idx, val in enumerate(vals):
                outputs[idx] += val
        return outputs

    @torch.inference_mode()
    def generate_batched_text_streaming(
        self,
        prompts: list[MixedSequenceType],
        max_gen_tokens: int = 256,
        temp: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
    ) -> Generator[list[str], None, None]:
        batch = []
        for prompt in prompts:
            batch.append(self.tokens_from_inputs(prompt))

        for tok in self._generate_batched_text_streaming(
            batch,
            max_gen_tokens=max_gen_tokens,
            temp=temp,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ):
            yield self.tokenizer.decode_batch(tok)

    @torch.inference_mode()
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
        if len(tokenized_prompt) > (4096 - 3):
            yield "ERROR: Your input exceeds the model's context length of 4096. Note that images consume 1024 tokens whether in input or output."
            return
        for out in self.generate_batched_text_streaming(
            [prompt],
            max_gen_tokens=max_gen_tokens,
            temp=temp,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ):
            yield out[0]

    @torch.inference_mode()
    async def generate_image_streaming(
        self,
        prompt: MixedSequenceType,
        temp: float = 1.0,
        top_p: float = 0.8,
        cfg_image_weight: float = DEFAULT_IMAGE_CFG_IMAGE,
        cfg_text_weight: float = DEFAULT_IMAGE_CFG_TEXT,
        yield_every_n: int = 32,
        seed: int | None = None,
        debug: dict | None = None,
    ) -> Generator[StreamingImage, None, None]:
        assert isinstance(prompt, list)
        tokenized_prompt = self.tokens_from_inputs(prompt)
        tokenized_prompt.append(self.vocab.begin_image)
        if len(tokenized_prompt) > (4096 - 3 - 1024):
            yield "ERROR: Your input exceeds the model's context length of 4096. Note that images consume 1024 tokens whether in input or output."
            return
        for tokens, final in self._generate_image_streaming(
            tokenized_prompt,
            temp=temp,
            top_p=top_p,
            cfg_image_weight=cfg_image_weight,
            cfg_text_weight=cfg_text_weight,
            yield_every_n=yield_every_n,
            seed=seed,
        ):
            yield StreamingImage(
                image=self.pillow_from_bpe_tokens(torch.tensor(tokens)), final=final
            )

    @torch.inference_mode()
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
        input_ids = self.tokens_from_inputs(prompt, suffix_tokens=suffix_tokens)
        if len(input_ids) > (4096 - 3):
            yield "ERROR: Your input exceeds the model's context length of 4096. Note that images consume 1024 tokens."
            return

        for token_type, tokens, is_final in self._generate_multimodal_streaming(
            input_ids,
            temp=temp,
            top_p=top_p,
            cfg_image_weight=cfg_image_weight,
            cfg_text_weight=cfg_text_weight,
            yield_every_n=yield_every_n,
            max_gen_tokens=max_gen_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ):
            match token_type:
                case "TEXT":
                    yield self.tokenizer.decode(tokens)
                case "IMAGE":
                    yield StreamingImage(
                        image=self.pillow_from_bpe_tokens(torch.tensor(tokens)),
                        final=is_final,
                    )
                case _:
                    raise ValueError("Unknown token type")
