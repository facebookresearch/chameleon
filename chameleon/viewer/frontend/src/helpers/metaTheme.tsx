/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
/**
 * Returns tailwind color classes for text contents, based on the darkMode boolean
 */
export function getTextColors(darkMode: boolean = false): {
  primary: string;
  secondary: string;
} {
  const primary = darkMode ? "text-white" : "text-gray-800";
  const secondary = darkMode ? "text-gray-300" : "text-gray-600";
  return { primary, secondary };
}

export type PrimaryColors = "white" | "gray" | "darkGray" | "blue";

export function getBackgroundColors(id: string): string {
  const bgColorToClass = {
    white: "bg-white",
    gray: "bg-gray-50",
    darkGray: "bg-gray-800",
    blue: "bg-blue-50",
  };
  return bgColorToClass[id] || undefined;
}

export function getBorderColors(id: string): string {
  const bgColorToClass = {
    white: "border-white",
    gray: "border-gray-100",
    darkGray: "border-gray-800",
    blue: "border-blue-100",
  };
  return bgColorToClass[id] || undefined;
}
