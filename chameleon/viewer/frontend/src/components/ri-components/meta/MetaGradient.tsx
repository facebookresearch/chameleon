/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
export type GradientFill = "dark" | "light";
export type GradientAspectRatio = "1x1" | "16x9" | "9x16";

export type MetaGradientProps = {
  gradient?: GradientFill;
  aspectRatio?: GradientAspectRatio;
  className?: string;
};

import g_16x9 from "../../ri-assets/images/gradient_16x9.jpg";
import g_16x9_dark from "../../ri-assets/images/gradient_16x9_dark.jpg";
import g_1x1 from "../../ri-assets/images/gradient_1x1.jpg";
import g_1x1_dark from "../../ri-assets/images/gradient_1x1_dark.jpg";
import g_9x16 from "../../ri-assets/images/gradient_9x16.jpg";
import g_9x16_dark from "../../ri-assets/images/gradient_9x16_dark.jpg";

export function MetaGradient({
  gradient,
  aspectRatio = "16x9",
  className = "",
}: MetaGradientProps) {
  const getGradient = (aspectRatio, gradient) => {
    const files = {
      g_1x1,
      g_1x1_dark,
      g_9x16,
      g_9x16_dark,
      g_16x9,
      g_16x9_dark,
    };
    return files[`g_${aspectRatio}_${gradient}`] || g_16x9;
  };

  return (
    <div
      className={`absolute inset-0 z-0 ${className}`}
      style={{
        backgroundImage: `url(${getGradient(aspectRatio, gradient)})`,
        backgroundSize: "100% 100%",
      }}
    />
  );
}
