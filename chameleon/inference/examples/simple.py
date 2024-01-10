# Copyright (c) Meta Platforms, Inc. and affiliates.
#
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

    tokens = model.generate(prompt_text="All your base")
    print(model.decode_text(tokens)[0])


if __name__ == "__main__":
    main()
