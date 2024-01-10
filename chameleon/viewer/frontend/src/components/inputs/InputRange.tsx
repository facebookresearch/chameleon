/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { ChangeEvent, useEffect, useState } from "react";

export type InputRangeProps = {
  value: number;
  step?: number;
  min?: number;
  max?: number;
  integerOnly?: boolean;
  slider?: boolean;
  optional?: boolean;
  placeholder?: string;
  label?: string;
  className?: string;
  onValueChange: (n: any) => void;
};

export function InputRange({
  value,
  onValueChange,
  step = 0.1,
  min = 0,
  max = 1,
  integerOnly = false,
  slider = true,
  optional = false,
  placeholder,
  label,
  className,
}: InputRangeProps) {
  const [tempValue, setTempValue] = useState<string>("");
  const [tempOptionalValue, setTempOptionalValue] = useState<number>(0);
  const [valid, setValid] = useState<boolean>(true);
  const [skipped, setSkipped] = useState<boolean>(optional);

  /*
   * Here we first validate the value and then run the update callback if the value is valid.
   */
  function validate(valueString: string, updateFn: (value: any) => void) {
    setTempValue(valueString);
    const parseFn = integerOnly ? parseInt : parseFloat;
    const n = parseFn(valueString) || NaN;
    const integerCheck = integerOnly ? Math.floor(n) === n : true;

    if (skipped) {
      updateFn(null);
      setValid(true);
    } else if (n && n >= min && n <= max && integerCheck) {
      updateFn(n);
      setValid(true);
    } else {
      setValid(false);
    }
  }

  function handleOptional(evt: ChangeEvent<HTMLInputElement>) {
    // if checked, skip should be false
    if (evt.currentTarget.checked === skipped) {
      setSkipped(!skipped);
    }
  }

  useEffect(() => {
    setTempValue(`${value}`);
  }, [value, setTempValue]);

  useEffect(() => {
    if (skipped) {
      setTempOptionalValue(value);
      validate("", onValueChange);
    } else if (optional && !value) {
      validate(`${tempOptionalValue || min}`, onValueChange);
    }
  }, [skipped, tempOptionalValue, setTempOptionalValue]);

  const input = (
    <div className={`flex flex-row items-center gap-4 ${className}`}>
      {slider && (
        <input
          type="range"
          min={min}
          max={max}
          value={tempValue}
          step={step}
          disabled={skipped}
          className={`range flex-1`}
          onChange={(evt) => validate(evt.currentTarget.value, onValueChange)}
        />
      )}
      <input
        type="text"
        placeholder={placeholder || ""}
        value={tempValue}
        disabled={skipped}
        onChange={(evt) => validate(evt.currentTarget.value, onValueChange)}
        className={`input ${valid ? "border-gray-200" : "border-red-300"} w-24`}
      />
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
        <span className={`flex-1 `}>
          {label}{" "}
          {skipped && <div className="text-xs text-gray-400">(skipped)</div>}
        </span>
      </label>
      <div className={`flex-1 flex-grow-3 ${skipped && "hidden"}`}>{input}</div>
    </div>
  ) : (
    input
  );
}
