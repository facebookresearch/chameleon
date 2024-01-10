# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from chameleon.inference.chameleon import ChameleonInferenceModel


def main():
    model = ChameleonInferenceModel(
        "./data/models/7b/",
        "./data/tokenizer/text_tokenizer.json",
        "./data/tokenizer/vqgan.yaml",
        "./data/tokenizer/vqgan.ckpt",
    )

    batch_tokens = model.generate(batch_prompt_text=["All your base", "import asyncio"])
    for text in model.decode_text(batch_tokens):
        print(text)


if __name__ == "__main__":
    main()
