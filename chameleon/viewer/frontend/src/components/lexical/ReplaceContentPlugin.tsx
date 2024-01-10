/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { useLexicalComposerContext } from "@lexical/react/LexicalComposerContext";
import {
  $createParagraphNode,
  $createTextNode,
  $getRoot,
} from "lexical";
import { useEffect, useState } from "react";

import type { InsertImagePayload } from "./ImagesPlugin";
import { INSERT_IMAGE_COMMAND } from "./ImagesPlugin";

/**
 * This is a hacky plugin to replace contents in the Lexical Composer. It needs to be improved.
 */

export type ReplaceContentData = {
  content_type: string;
  content: string;
};

export function ReplaceContentPlugin({
  payload,
}: {
  payload: ReplaceContentData[];
}) {
  const [editor] = useLexicalComposerContext();
  const [last, setLast] = useState<ReplaceContentData[] | null>(null);

  useEffect(() => {
    if (last == null) {
      return;
    }

    editor.update(() => {
      const root = $getRoot();
      root.clear();

      for (let i = 0; i < last.length; i++) {
        const item = last[i];
        if (item.content_type === "TEXT") {
          const paragraphNode = $createParagraphNode();
          const text = $createTextNode(item.content);
          paragraphNode.append(text);
          root.append(paragraphNode);
        } else {
          editor.dispatchCommand(INSERT_IMAGE_COMMAND, {
            altText: "an image",
            src: item.content,
          } as InsertImagePayload);
        }
      }

      setLast(null);
    });
  }, [last]);

  useEffect(() => {
    if (payload !== null) {
      setLast(payload);
    }
  }, [payload]);

  return null;
}
