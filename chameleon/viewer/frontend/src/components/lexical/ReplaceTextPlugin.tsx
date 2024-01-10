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

/**
 * This is a hacky plugin to replace contents in the Lexical Composer. It needs to be improved.
 */

export function ReplaceTextPlugin({ payload }: { payload: string | null }) {
  const [editor] = useLexicalComposerContext();
  const [last, setLast] = useState<string | null>(null);

  useEffect(() => {
    if (last !== null) {
      editor.update(() => {
        const root = $getRoot();
        root.clear();
        const paragraphNode = $createParagraphNode();
        const text = $createTextNode(payload || "");
        paragraphNode.append(text);
        root.append(paragraphNode);

        setLast(null);
      });
    }
  }, [last]);

  useEffect(() => {
    // To prevent a weird infinite loop
    if (last !== payload && payload !== null) {
      console.log(`replace content with ${payload.substring(0, 10)}...`);
      setLast(payload);
    }
  }, [payload]);

  return null;
}
