/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { logo_light, logo_dark } from "./_logos";

export type MetaAILogoVariant = "light" | "dark" | undefined;
export type MetaAILogoProps = {
  className?: string;
  variant: MetaAILogoVariant;
  link?: string;
  style?: object;
};

export function MetaAILogo({
  className = "",
  variant,
  link,
  style = {},
}: MetaAILogoProps) {
  const logo = (
    <div
      className={`block cursor-pointer text-base ${className}`}
      style={style}
    >
      {variant === "dark" ? logo_dark : logo_light}
    </div>
  );

  return (
    <a className="no-style h-fit" href={link || "https://ai.meta.com/"}>
      {logo}
    </a>
  );
}
