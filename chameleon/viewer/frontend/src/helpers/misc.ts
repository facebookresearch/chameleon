/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/

export function assetPath(url: string): string {
  const baseUrl = import.meta.env.BASE_URL;
  return `${baseUrl}${url}`;
}

export function timeout(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
