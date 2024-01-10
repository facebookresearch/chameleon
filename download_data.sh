#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Chameleon License Agreement.

set -e
PRESIGNED_URL=$1

if [ -z "$PRESIGNED_URL" ]; then
  read -p "Enter the URL from email:" PRESIGNED_URL
fi

read -p "Enter the list of models to download without spaces (7B,30B), or press Enter for all: " MODEL_SIZE
TARGET_FOLDER="./data"             # where all files should end up
mkdir -p ${TARGET_FOLDER}

if [[ $MODEL_SIZE == "" ]]; then
    MODEL_SIZE="7B,30B"
fi


echo "Downloading tokenizer"
mkdir -p ${TARGET_FOLDER}"/tokenizer"
wget --continue ${PRESIGNED_URL/'*'/"tokenizer/text_tokenizer.json"} -O ${TARGET_FOLDER}"/tokenizer/text_tokenizer.json"
wget --continue ${PRESIGNED_URL/'*'/"tokenizer/vqgan.ckpt"} -O ${TARGET_FOLDER}"/tokenizer/vqgan.ckpt"
wget --continue ${PRESIGNED_URL/'*'/"tokenizer/vqgan.yaml"} -O ${TARGET_FOLDER}"/tokenizer/vqgan.yaml"
wget --continue ${PRESIGNED_URL/'*'/"tokenizer/checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer/checklist.chk"
CPU_ARCH=$(uname -m)
if [ "$CPU_ARCH" == "arm64" ]; then
(cd ${TARGET_FOLDER}/tokenizer && md5 checklist.chk)
else
(cd ${TARGET_FOLDER}/tokenizer && md5sum -c checklist.chk)
fi

TARGET_FOLDER_MODELS=${TARGET_FOLDER}"/models"
mkdir -p ${TARGET_FOLDER_MODELS}
for m in ${MODEL_SIZE//,/ }
do
    if [[ $m == "7B" ]] || [[ $m == "7b" ]]; then
        SHARD=0
        MODEL_PATH="7b"
    elif [[ $m == "30B" ]] || [[ $m == "30b" ]]; then
        SHARD=4
        MODEL_PATH="30b"
    fi

    echo "Downloading ${MODEL_PATH}"
    mkdir -p ${TARGET_FOLDER_MODELS}"/${MODEL_PATH}"

    if [[ $SHARD == 0 ]]; then
        wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/consolidated.pth"} -O ${TARGET_FOLDER_MODELS}"/${MODEL_PATH}/consolidated.pth"
    else
        for s in $(seq -f "0%g" 0 $(( ${SHARD} - 1 )))
        do
            wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/consolidated.${s}.pth"} -O ${TARGET_FOLDER_MODELS}"/${MODEL_PATH}/consolidated.${s}.pth"
        done
    fi

    wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/params.json"} -O ${TARGET_FOLDER_MODELS}"/${MODEL_PATH}/params.json"
    wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/consolidate_params.json"} -O ${TARGET_FOLDER_MODELS}"/${MODEL_PATH}/consolidate_params.json"
    wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/checklist.chk"} -O ${TARGET_FOLDER_MODELS}"/${MODEL_PATH}/checklist.chk"
 
    echo "Checking checksums"
    if [ "$CPU_ARCH" == "arm64" ]; then
      (cd ${TARGET_FOLDER_MODELS}"/${MODEL_PATH}" && md5 checklist.chk)
    else
      (cd ${TARGET_FOLDER_MODELS}"/${MODEL_PATH}" && md5sum -c checklist.chk)
    fi
done
