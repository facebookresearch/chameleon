/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/

import type { Meta, StoryObj } from "@storybook/react";

import { MetaAILogo } from "./MetaAILogo";

// More on how to set up stories at: https://storybook.js.org/docs/react/writing-stories/introduction
const meta = {
  title: "Meta/MetaAILogo",
  component: MetaAILogo,
  tags: ["autodocs"],
} satisfies Meta<typeof MetaAILogo>;

export default meta;
type Story = StoryObj<typeof meta>;

// More on writing stories with args: https://storybook.js.org/docs/react/writing-stories/args
export const dark: Story = {
  args: {
    variant: "dark",
  },
};
