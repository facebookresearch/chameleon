/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { useEffect, useState } from "react";

export function useAdvancedMode(): [boolean, (on: boolean) => void] {
  const [advancedMode, setAdvancedMode] = useState(
    (localStorage.getItem("advancedMode") || "") === "true",
  );

  useEffect(() => {
    localStorage.setItem("advancedMode", advancedMode ? "true" : "false");
  }, [advancedMode]);

  return [advancedMode, setAdvancedMode];
}
