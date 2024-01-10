/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { Button, ButtonProps } from "react-daisyui";

import { ReplaceContentData } from "../lexical/ReplaceContentPlugin";

export interface Props extends ButtonProps {
  label: string;
  uuid: string;
  onLoadExample: (example: ReplaceContentData[]) => void;
}

export function InputExampleButton({
  label,
  example,
  onLoadExample,
  ...props
}: Props) {
  return (
    <Button
      {...props}
      onClick={(evt) => {
        onLoadExample(example.inputs.content);
        evt.preventDefault();
      }}
    >
      {label}
    </Button>
  );
}
