/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/

export const Config = {
  ws_address: import.meta.env.VITE_WS_ADDRESS || "ws://0.0.0.0:7102",
  default_seed: 97,
};
