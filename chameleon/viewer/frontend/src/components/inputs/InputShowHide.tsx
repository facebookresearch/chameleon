/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import React, { useState } from "react";

export type InputShowHideProps = {
  children: React.ReactNode;
  labelShow: string;
  labelHide: string;
};

export function InputShowHide({
  children,
  labelShow = "Show",
  labelHide = "Hide",
}: InputShowHideProps) {
  const [advanced, setAdvanced] = useState(false);
  return (
    <>
      <div
        className="mt-6 flex flex-row gap-2 items-center cursor-pointer"
        onClick={() => setAdvanced(!advanced)}
      >
        <div className="flex-1 p-1 text-blue-500 whitespace-nowrap">
          {advanced ? labelHide : labelShow}
        </div>
        <div className="flex-1 flex-grow-3">
          <div className="w-full h-[1px] bg-blue-200"></div>
        </div>
      </div>
      <div className={`${advanced ? "block" : "hidden"}`}>{children}</div>
    </>
  );
}
