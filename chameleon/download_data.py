# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Chameleon License Agreement.

import hashlib
from pathlib import Path
import subprocess
import sys


def download_file(url: str, output_path: Path):
    print(f"Downloading {output_path}")
    subprocess.check_call(["wget", "--continue", url, "-O", str(output_path)])


def validate_checksum(folder: Path):
    chks_parts = (folder / "checklist.chk").read_text().split()
    for expected_checksum, file in zip(chks_parts[::2], chks_parts[1::2]):
        file_path = folder / file
        checksum = hashlib.md5(file_path.read_bytes()).hexdigest()
        if checksum != expected_checksum:
            print(f"Checksum mismatch for {file_path}")
            sys.exit(1)


def download_tokenizer(presigned_url: str, target_folder: Path):
    tokenizer_folder = target_folder / "tokenizer"
    tokenizer_folder.mkdir(parents=True, exist_ok=True)

    for filename in [
        "text_tokenizer.json",
        "vqgan.ckpt",
        "vqgan.yaml",
        "checklist.chk",
    ]:
        download_file(
            presigned_url.replace("*", f"tokenizer/{filename}"),
            tokenizer_folder / filename,
        )

    validate_checksum(tokenizer_folder)


def download_model(presigned_url: str, target_folder: Path, model: str):
    model_folder = target_folder / "models" / model
    model_folder.mkdir(parents=True, exist_ok=True)

    download_filenames = ["params.json", "consolidate_params.json", "checklist.chk"]

    if model == "7b":
        download_filenames += ["consolidated.pth"]
    elif model == "30b":
        download_filenames += [f"consolidated.{i:02}.pth" for i in range(4)]
    else:
        print(f"Unknown model: {model}")
        sys.exit(1)

    for filename in download_filenames:
        download_file(
            presigned_url.replace("*", f"{model}/{filename}"),
            model_folder / filename,
        )

    validate_checksum(model_folder)


def main():
    presigned_url = (
        sys.argv[1] if len(sys.argv) > 1 else input("Enter the URL from email: ")
    )

    target_folder = Path("./data")
    target_folder.mkdir(parents=True, exist_ok=True)

    download_tokenizer(presigned_url, target_folder)

    model_size = input(
        "Enter the list of models to download without spaces (7B,30B), or press Enter for all: "
    )
    if not model_size:
        model_size = "7B,30B"

    for model in model_size.split(","):
        model = model.strip().lower()
        download_model(presigned_url, target_folder, model)


if __name__ == "__main__":
    main()
