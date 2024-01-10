/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { useEffect, useState } from "react";
import { ZoomIn, ZoomOut } from "@carbon/icons-react";

interface ImageResultProps {
  src: string;
  large?: boolean;
  completed?: boolean;
}

export function ImageResult({
  src: base64,
  large = false,
  completed = true,
}: ImageResultProps) {
  const [expand, setExpand] = useState(false);

  useEffect(() => {
    setExpand(large);
  }, [large]);

  return base64 ? (
    <div className="relative flex flex-col w-fit h-fit my-4 group">
      <img src={base64} width={expand ? 512 : 256} className="m-0" />
      {completed && (
        <div
          className="absolute top-0 right-[-36px] w-9 h-9 flex py-2 rounded-r-lg bg-white text-black flex-row items-center justify-center text-xs font-medium gap-1 hover:bg-gray-100 cursor-pointer"
          onClick={() => setExpand(!expand)}
        >
          {expand ? (
            <>
              <ZoomOut size={20} />
            </>
          ) : (
            <>
              <ZoomIn size={20} />
            </>
          )}
        </div>
      )}
    </div>
  ) : (
    <></>
  );
}
