/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/

import { ReadyState } from "react-use-websocket";
import { z } from "zod";

export const ZUidList = z.array(z.string()).nonempty();
export type UidList = z.infer<typeof ZUidList>;

export const GENERATE_TEXT = "GENERATE_TEXT";
export const GENERATE_IMAGE = "GENERATE_IMAGE";
export const GENERATE_MULTIMODAL = "GENERATE_MULTIMODAL";
export const PARTIAL_OUTPUT = "PARTIAL_OUTPUT";
export const FULL_OUTPUT = "FULL_OUTPUT";
export const COMPLETE = "COMPLETE";
export const QUEUE_STATUS = "QUEUE_STATUS";
export const TEXT = "TEXT";
export const IMAGE = "IMAGE";
export function readableWsState(state: number) {
  if (state == ReadyState.CONNECTING) {
    return "Connecting";
  } else if (state == ReadyState.OPEN) {
    return "Open";
  } else if (state == ReadyState.CLOSING) {
    return "Closing";
  } else if (state == ReadyState.CLOSED) {
    return "Closed";
  } else if (state == ReadyState.UNINSTANTIATED) {
    return "Uninstatiated";
  } else {
    return "Unknown";
  }
}
export const EOT_TOKEN = "<reserved08706>";

// These should be in sync with: chameleon/viewer/backend/data_types.py
export const ZWSContent = z.object({
  content_type: z.enum([TEXT, IMAGE]),
  content: z.string(),
});

export const ZWSMessageType = z.enum([
  GENERATE_TEXT,
  GENERATE_IMAGE,
  GENERATE_MULTIMODAL,
  PARTIAL_OUTPUT,
  FULL_OUTPUT,
  COMPLETE,
  QUEUE_STATUS,
]);

export const ZWSTextOptions = z.object({
  message_type: ZWSMessageType,
  max_gen_tokens: z.number().optional(),
  temp: z.number().optional(),
  top_p: z.number().optional(),
  repetition_penalty: z.number(),
  seed: z.number().optional().nullable(),
});

export const ZWSImageOptions = z.object({
  message_type: ZWSMessageType,
  temp: z.number().optional(),
  top_p: z.number().optional(),
  cfg_image_weight: z.number().optional(),
  cfg_text_weight: z.number().optional(),
  yield_every_n: z.number().optional(),
  seed: z.number().optional().nullable(),
});

export const ZWSMultimodalOptions = z.object({
  message_type: ZWSMessageType,
  temp: z.number().optional(),
  top_p: z.number().optional(),
  cfg_image_weight: z.number().optional(),
  cfg_text_weight: z.number().optional(),
  yield_every_n: z.number().optional(),
  max_gen_tokens: z.number().optional(),
  repetition_penalty: z.number().optional(),
  suffix_tokens: z.array(z.string()).optional().nullable(),
  seed: z.number().optional().nullable(),
});

export const ZWSMultimodalMessage = z.object({
  message_type: ZWSMessageType,
  // Array<text | image>, where image are encoded <img src="data..."/>
  content: z.array(ZWSContent),
  options: z.union([ZWSTextOptions, ZWSImageOptions, ZWSMultimodalOptions]),
  debug_info: z.record(z.string()),
});

export const ZFrontendMultimodalSequencePair = z.object({
  uid: z.string().optional().nullable(),
  user: z.string(),
  inputs: ZWSMultimodalMessage,
  outputs: z.array(ZWSContent),
});

export type WSTextOptions = z.infer<typeof ZWSTextOptions>;
export type WSImageOptions = z.infer<typeof ZWSImageOptions>;
export type WSMultimodalOptions = z.infer<typeof ZWSMultimodalOptions>;
export type WSOptions = WSTextOptions | WSImageOptions | WSMultimodalOptions;

export type WSContent = z.infer<typeof ZWSContent>;
export type WSMultimodalMessage = z.infer<typeof ZWSMultimodalMessage>;
export type FrontendMultimodalSequencePair = z.infer<
  typeof ZFrontendMultimodalSequencePair
>;

export function mergeTextContent(contents: Array<WSContent>) {
  let output: Array<WSContent> = [];
  let buffer: Array<WSContent> = [];
  let prevType: string | null = null;

  const processBuffer = (type: string | null) => {
    switch (type) {
      case IMAGE:
        output = output.concat(buffer);
        break;

      case TEXT:
        const text = buffer.map((x) => x.content).join("");
        output.push({ content_type: TEXT, content: text });
        break;

      case null:
        // Do nothing for null
        break;

      default:
        throw new Error("Invalid content type");
    }
    buffer = [];
  };

  for (const content of contents) {
    if (prevType !== null && prevType !== content.content_type) {
      processBuffer(prevType);
    }

    buffer.push(content);
    prevType = content.content_type;
  }

  processBuffer(prevType);

  return output;
}
