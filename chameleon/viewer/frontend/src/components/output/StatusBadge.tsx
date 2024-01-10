/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
interface Props {
  label: string;
  status: string;
  category?: StatusCategory;
}

export type StatusCategory =
  | "success"
  | "warning"
  | "error"
  | "info"
  | "neutral"
  | "green";

import {
  CheckmarkFilled,
  WarningAltFilled,
  ErrorFilled,
  InformationFilled,
  HelpFilled,
} from "@carbon/icons-react";

export function StatusBadge({ label, status, category = "neutral" }: Props) {
  const extra = "";
  const colorMap = (cat: string) => {
    const map = {
      success: (
        <CheckmarkFilled size={16} className={`fill-green-500 ${extra}`} />
      ),
      green: (
        <CheckmarkFilled size={16} className={`fill-green-500 ${extra}`} />
      ),
      warning: (
        <WarningAltFilled size={16} className={`fill-orange-400 ${extra}`} />
      ),
      error: <ErrorFilled size={16} className={`fill-red-400 ${extra}`} />,
      info: (
        <InformationFilled size={16} className={`fill-blue-400 ${extra}`} />
      ),
    };
    return (
      map[cat] || <HelpFilled size={16} className={`fill-gray-400 ${extra}`} />
    );
  };

  return (
    <div className="relative cursor-default select-none">
      <div
        className="tooltip flex flex-row items-center gap-1"
        data-tip={status}
      >
        {colorMap(category)} <div>{label}</div>
      </div>
    </div>
  );
}
