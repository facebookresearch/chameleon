/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { WSContent, IMAGE } from "../../DataTypes";
import { User, Bot } from "@carbon/icons-react";
import { ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ImageResult } from "./ImageResult";

export interface ChatRowProps {
  isUser: boolean;
  data?: WSContent;
  children?: ReactNode;
  index?: number;
  streaming?: boolean;
}

export function ChatRow({
  isUser,
  data,
  children,
  index = 0,
  streaming = false,
}: ChatRowProps) {
  const userIconStyle = `w-6 h-6 p-1 ${
    isUser ? "bg-gray-100" : "bg-purple-200"
  } flex items-center justify-start rounded-full`;

  const badgeStyle = "flex flex-row items-center gap-2 text-sm font-bold";

  return (
    <div
      className={`chat-row flex flex-col items-stretch my-6 ${
        streaming ? "text-purple-700" : "text-gray-900"
      }`}
    >
      {data && (
        <>
          <div className={`${badgeStyle} ${!isUser && "text-purple-700"}`}>
            {isUser ? (
              <>
                <div className={userIconStyle}>
                  <User size={16} />
                </div>
                <div>You</div>
              </>
            ) : (
              <>
                <div className={userIconStyle}>
                  <Bot size={16} />
                </div>
                <div>Chameleon</div>
              </>
            )}
          </div>

          <div
            className={`flex-1 mt-2 ml-8 ${
              !isUser && " border-l-2 border-l-purple-200 pl-4"
            }`}
          >
            {data.content_type === IMAGE ? (
              <ImageResult src={data.content} key={`img${index}`} />
            ) : (
              <Markdown remarkPlugins={[remarkGfm]} key={`text${index}`}>
                {data.content}
              </Markdown>
            )}
          </div>
        </>
      )}

      {/* Streaming temporary content in children */}

      {children && (
        <>
          <div className={`${badgeStyle} text-purple-700`}>
            <div className={userIconStyle}>
              <Bot size={16} />
            </div>
            <div>Chameleon</div>
          </div>
          <div className={`flex-1 mt-2 ml-8`}>{children}</div>
        </>
      )}
    </div>
  );
}
