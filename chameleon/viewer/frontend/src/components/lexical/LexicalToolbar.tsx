/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { useLexicalComposerContext } from "@lexical/react/LexicalComposerContext";
import * as React from "react";

import type { InsertImagePayload } from "./ImagesPlugin";
import { INSERT_IMAGE_COMMAND } from "./ImagesPlugin";

export function FillURL() {
  const srcfile = prompt("Enter the URL of the image:", "");

  return srcfile;
}

export function ToolbarPlugin() {
  const [editor] = useLexicalComposerContext();
  const onClick = (payload: InsertImagePayload) => {
    editor.dispatchCommand(INSERT_IMAGE_COMMAND, payload);
  };

  return (
    <div className="p-4">
      <button
        className="btn mr-2"
        onClick={() =>
          onClick({
            altText: "Pink flowers",
            src: "https://images.pexels.com/photos/5656637/pexels-photo-5656637.jpeg?auto=compress&cs=tinysrgb&w=200",
          })
        }
      >
        <span className="text">Insert Sample</span>
      </button>
      <button
        className="btn mr-2"
        onClick={() =>
          onClick({
            altText: "URL image",
            src: FillURL(),
          })
        }
      >
        <span className="text">Insert from URL</span>
      </button>
    </div>
  );
}
