/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { ChangeEvent, useState } from "react";

export type InputToggleProps = {
  value: boolean;
  optional?: boolean;
  label?: string;
  className?: string;
  onValueChange: (n: any) => void;
};

export function InputToggle({
  value,
  onValueChange,
  optional = false,
  label,
  className,
}: InputToggleProps) {
  const [skipped, setSkipped] = useState<boolean>(false);

  function handleOptional(evt: ChangeEvent<HTMLInputElement>) {
    // if checked, skip should be false
    if (evt.currentTarget.checked === skipped) {
      setSkipped(!skipped);
    }
  }

  const input = (
    <div
      className={`flex flex-row flex-1 items-center flex-grow-3 gap-2 ${className} ${
        skipped && "hidden"
      }`}
    >
      <input
        type="checkbox"
        className="toggle toggle-accent toggle-lg"
        checked={value}
        onChange={(evt) => onValueChange(evt.currentTarget.checked)}
      />
      <span>{value ? "Yes" : "No"}</span>
    </div>
  );

  return label ? (
    <div className="form-control mt-6 flex flex-row gap-2 items-center">
      <label className="label font-semibold flex flex-1 leading-5 gap-2">
        {optional && (
          <input
            type="checkbox"
            checked={!skipped}
            className="checkbox checkbox-primary"
            onChange={handleOptional}
          />
        )}
        <span className="flex-1">
          {label}{" "}
          {skipped && <div className="text-xs text-gray-400">(skipped)</div>}
        </span>
      </label>
      {input}
    </div>
  ) : (
    input
  );
}
