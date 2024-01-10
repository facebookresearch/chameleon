/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { expect, test } from "vitest";
import { WSContent, mergeTextContent, TEXT, IMAGE } from "./DataTypes";

test("Flatten contents works correctly", () => {
  const oneText: Array<WSContent> = [
    { content_type: TEXT, content: "hello world" },
  ];
  expect(mergeTextContent(oneText)).toStrictEqual(oneText);

  const twoText: Array<WSContent> = [
    { content_type: TEXT, content: "hello world" },
    { content_type: TEXT, content: " hello back" },
  ];
  expect(mergeTextContent(twoText)).toStrictEqual([
    { content_type: TEXT, content: "hello world hello back" },
  ]);

  const twoTextOneImage: Array<WSContent> = [
    { content_type: TEXT, content: "hello world" },
    { content_type: TEXT, content: " hello back" },
    { content_type: IMAGE, content: "IMAGE_ONE" },
  ];
  expect(mergeTextContent(twoTextOneImage)).toStrictEqual([
    { content_type: TEXT, content: "hello world hello back" },
    { content_type: IMAGE, content: "IMAGE_ONE" },
  ]);

  const oneImage: Array<WSContent> = [
    { content_type: IMAGE, content: "IMAGE_ONE" },
  ];
  expect(mergeTextContent(oneImage)).toStrictEqual([
    { content_type: IMAGE, content: "IMAGE_ONE" },
  ]);

  const oneImageTwoText: Array<WSContent> = [
    { content_type: IMAGE, content: "IMAGE_ONE" },
    { content_type: TEXT, content: "hello world" },
    { content_type: TEXT, content: " hello back" },
  ];
  expect(mergeTextContent(oneImageTwoText)).toStrictEqual([
    { content_type: IMAGE, content: "IMAGE_ONE" },
    { content_type: TEXT, content: "hello world hello back" },
  ]);

  const oneImageTwoTextOneImage: Array<WSContent> = [
    { content_type: IMAGE, content: "IMAGE_ONE" },
    { content_type: TEXT, content: "hello world" },
    { content_type: TEXT, content: " hello back" },
    { content_type: IMAGE, content: "IMAGE_TWO" },
  ];
  expect(mergeTextContent(oneImageTwoTextOneImage)).toStrictEqual([
    { content_type: IMAGE, content: "IMAGE_ONE" },
    { content_type: TEXT, content: "hello world hello back" },
    { content_type: IMAGE, content: "IMAGE_TWO" },
  ]);
});
