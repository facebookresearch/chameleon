/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { useEffect, useState, useRef } from "react";

import { LexicalComposer } from "@lexical/react/LexicalComposer";
import { ContentEditable } from "@lexical/react/LexicalContentEditable";
import { HistoryPlugin } from "@lexical/react/LexicalHistoryPlugin";
import { RichTextPlugin } from "@lexical/react/LexicalRichTextPlugin";
import { OnChangePlugin } from "@lexical/react/LexicalOnChangePlugin";
import DragDropPaste from "../lexical/DragDropPastePlugin";
import { ImagesPlugin } from "../lexical/ImagesPlugin";
import { ImageNode } from "../lexical/ImageNode";
import { ReplaceContentPlugin } from "../lexical/ReplaceContentPlugin";
import LexicalErrorBoundary from "@lexical/react/LexicalErrorBoundary";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { z } from "zod";
import JsonView from "react18-json-view";
import { InputRange } from "../inputs/InputRange";
import { Config } from "../../Config";
import axios from "axios";
import { useHotkeys } from "react-hotkeys-hook";
import {
  COMPLETE,
  FULL_OUTPUT,
  FrontendMultimodalSequencePair,
  GENERATE_MULTIMODAL,
  IMAGE,
  PARTIAL_OUTPUT,
  QUEUE_STATUS,
  TEXT,
  WSContent,
  WSMultimodalMessage,
  WSOptions,
  ZWSMultimodalMessage,
  mergeTextContent,
  readableWsState,
} from "../../DataTypes";
import { StatusBadge, StatusCategory } from "../output/StatusBadge";
import {
  SettingsAdjust,
  Close,
  Idea,
} from "@carbon/icons-react";
import { useAdvancedMode } from "../hooks/useAdvancedMode";
import { InputShowHide } from "../inputs/InputShowHide";
import { InputToggle } from "../inputs/InputToggle";

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { EOT_TOKEN } from "../../DataTypes";
import { ImageResult } from "../output/ImageResult";

enum GenerationSocketState {
  Generating = "GENERATING",
  UserWriting = "USER_WRITING",
  NotReady = "NOT_READY",
}
function makeid(length) {
  let result = "";
  const characters =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const charactersLength = characters.length;
  let counter = 0;
  while (counter < length) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
    counter += 1;
  }
  return result;
}

// Prepend an arbitrary texdt prompt to an existing list of contents
export function prependTextPrompt(
  toPrepend: string,
  contents: WSContent[],
): WSContent[] {
  if (toPrepend.length == 0) {
    return contents;
  }
  const promptContent: WSContent = {
    content: toPrepend,
    content_type: TEXT,
  };
  return [promptContent].concat(contents);
}

// Extract a flat list of text and image contents from the editor state
export function flattenContents(obj): WSContent[] {
  let result: WSContent[] = [];

  if (!obj || !obj.children || obj.children.length === 0) return result;

  for (const child of obj.children) {
    // Only take text and image contents
    if (child.type === "text") {
      result.push({ content: child.text, content_type: TEXT });
    } else if (child.type === "image") {
      result.push({
        // TODO: Convert the src from URL to base64 image
        content: child.src,
        content_type: IMAGE,
      });
    }
    const grandChildren = flattenContents(child);
    result = result.concat(grandChildren);
  }

  return result;
}

export function contentToHtml(content: WSContent, index?: number) {
  if (content.content_type == TEXT) {
    return (
      <Markdown remarkPlugins={[remarkGfm]} key={`t${index}`}>
        {content.content}
      </Markdown>
      // <code style={{ whiteSpace: "pre-wrap" }} key={`code${index}`}>
      //   {content.content}
      // </code>
    );
  } else if (content.content_type == IMAGE) {
    return <ImageResult src={content.content} key={`img${index}`} />;
  } else {
    return <p key={`p${index}`}>Unknown content type</p>;
  }
}

export function GenerateMixedModal() {
  function Editor() {
    const [clientId, setClientId] = useState<string>(makeid(8));
    const [generationState, setGenerationState] =
      useState<GenerationSocketState>(GenerationSocketState.NotReady);
    const [contents, setContents] = useState<WSContent[]>([]);

    const [partialImage, setPartialImage] = useState<string>("");

    // Model hyperparams
    const [temp, setTemp] = useState<number>(0.7);
    const [topP, setTopP] = useState<number>(0.9);
    const [cfgImageWeight, setCfgImageWeight] = useState<number>(1.2);
    const [cfgTextWeight, setCfgTextWeight] = useState<number>(3.0);
    const [yieldEveryN, setYieldEveryN] = useState<number>(32);
    const [seed, setSeed] = useState<number | null>(Config.default_seed);
    const [maxGenTokens, setMaxGenTokens] = useState<number>(4096);
    const [repetitionPenalty, setRepetitionPenalty] = useState<number>(1.2);

    const [showSeed, setShowSeed] = useState<boolean>(true);
    const [numberInQueue, setNumberInQueue] = useState<number>();

    const socketUrl = `${Config.ws_address}/ws/chameleon/v2/${clientId}`;

    // Array of text string or html string (i.e., an image)
    const [modelOutput, setModelOutput] = useState<Array<WSContent>>([]);

    const { readyState, sendJsonMessage, lastJsonMessage, getWebSocket } =
      useWebSocket(socketUrl, {
        onOpen: () => {
          console.log("WS Opened");
          setGenerationState(GenerationSocketState.UserWriting);
        },
        onClose: (e) => {
          console.log("WS Closed", e);
          setGenerationState(GenerationSocketState.NotReady);
        },
        onError: (e) => {
          console.log("WS Error", e);
          setGenerationState(GenerationSocketState.NotReady);
        },
        // TODO: Inspect error a bit
        shouldReconnect: (closeEvent) => true,
        heartbeat: false,
      });

    function abortGeneration() {
      getWebSocket()?.close();
      setModelOutput([]);
      setGenerationState(GenerationSocketState.UserWriting);
      setClientId(makeid(8));
    }

    useEffect(() => {
      if (lastJsonMessage != null) {
        const maybeMessage = ZWSMultimodalMessage.safeParse(lastJsonMessage);
        console.log("Message", lastJsonMessage, "Parsed", maybeMessage.success);
        if (maybeMessage.success) {
          if (
            maybeMessage.data.content.length != 1 &&
            maybeMessage.data.message_type != COMPLETE
          ) {
            console.error("Too few or too many content");
          }
          console.log("parsed message", maybeMessage);
          if (maybeMessage.data.message_type == PARTIAL_OUTPUT) {
            // Currently, the backend only sends one content piece at a time
            const content = maybeMessage.data.content[0];
            if (content.content_type == IMAGE) {
              setPartialImage(content.content);
            } else if (content.content_type == TEXT) {
              setModelOutput((prev) => {
                return prev.concat(maybeMessage.data.content);
              });
            }
            setNumberInQueue(undefined);
          } else if (maybeMessage.data.message_type == FULL_OUTPUT) {
            // Only image gives full output, text is rendered as it
            // comes.
            const content = maybeMessage.data.content[0];
            if (content.content_type == IMAGE) {
              setPartialImage("");
              setModelOutput((prev) => {
                console.log("Set model image output");
                return prev.concat(maybeMessage.data.content);
              });
            }
          } else if (maybeMessage.data.message_type == COMPLETE) {
            setGenerationState(GenerationSocketState.UserWriting);
          } else if (maybeMessage.data.message_type == QUEUE_STATUS) {
            console.log("Queue Status Message", maybeMessage);
            // expects payload to be n_requests=<number>
            setNumberInQueue(
              Number(maybeMessage.data.content[0].content.match(/\d+/g)),
            );
          }
        }
      } else {
        console.log("Null message");
      }
    }, [lastJsonMessage, setModelOutput]);

    const initialConfig = {
      namespace: "MyEditor",
      theme: {
        heading: {
          h1: "text-24 text-red-500",
        },
      },
      onError,
      nodes: [ImageNode],
    };

    function onError(error) {
      console.error(error);
    }

    function Placeholder() {
      return (
        <>
          <div className="absolute top-4 left-4 z-0 select-none pointer-events-none opacity-50 prose">
            You can edit text and drag/paste images in the input above.<br />
            It's just like writing a mini document.
          </div>
        </>
      );
    }

    function onChange(editorState) {
      // Call toJSON on the EditorState object, which produces a serialization safe string
      const editorStateJSON = editorState.toJSON();
      setContents(flattenContents(editorStateJSON?.root));
      setExamplePrompt(null);
    }

    function onRunModelClick() {
      if (runButtonDisabled) return;

      async function prepareContent(content: WSContent): Promise<WSContent> {
        if (content.content_type == TEXT) {
          return content;
        } else if (content.content_type == IMAGE) {
          if (content.content.startsWith("http")) {
            const response = await fetch(content.content);
            const blob = await response.blob();
            const reader = new FileReader();
            return new Promise((resolve) => {
              reader.onload = (event) => {
                const result = event.target?.result;
                if (typeof result === "string") {
                  resolve({ ...content, content: result });
                } else {
                  resolve(content);
                }
              };
              reader.readAsDataURL(blob);
            });
          } else {
            return content;
          }
        } else {
          console.error("Unknown content type");
          return content;
        }
      }

      async function prepareAndRun() {
        if (contents.length != 0) {
          setModelOutput([]);
          setGenerationState(GenerationSocketState.Generating);
          const currentContent = await Promise.all(
            contents.map(prepareContent),
          );

          let processedContents = currentContent;

          const suffix_tokens: Array<string> = [EOT_TOKEN];
          const options: WSOptions = {
            message_type: GENERATE_MULTIMODAL,
            temp: temp,
            top_p: topP,
            cfg_image_weight: cfgImageWeight,
            cfg_text_weight: cfgTextWeight,
            repetition_penalty: repetitionPenalty,
            yield_every_n: yieldEveryN,
            max_gen_tokens: maxGenTokens,
            suffix_tokens: suffix_tokens,
            seed: seed,
          };

          const message: WSMultimodalMessage = {
            message_type: GENERATE_MULTIMODAL,
            content: processedContents,
            options: options,
            debug_info: {},
          };
          setContents(processedContents);
          sendJsonMessage(message);
        }
      }
      prepareAndRun().catch(console.error);
    }

    useHotkeys("ctrl+enter, cmd+enter", () => {
      console.log("Run Model by hotkey");
      onRunModelClick();
    });

    const readableSocketState = readableWsState(readyState);
    let socketStatus: StatusCategory = "neutral";
    if (readableSocketState == "Open") {
      socketStatus = "success";
    } else if (readableSocketState == "Closed") {
      socketStatus = "error";
    } else if (readableSocketState == "Connecting") {
      socketStatus = "warning";
    } else {
      socketStatus = "error";
    }
    const runButtonDisabled =
      readyState !== ReadyState.OPEN ||
      generationState != GenerationSocketState.UserWriting;
    const runButtonText = runButtonDisabled ? (
      <div className="loading loading-infinity loading-lg text-neutral"></div>
    ) : (
      <div className="flex flex-row items-center">
        Run Model
        {/* Use the following label when hot-key is implemented
        <span className="flex flex-row items-center ml-2 text-[10px] text-gray-600">
          <MacCommand size={12} className="inline" />
          +ENTER
        </span> */}
      </div>
    );
    const runButtonColor = runButtonDisabled
      ? "btn-neutral opacity-60"
      : "btn-success";
    let uiStatus: StatusCategory = "neutral";
    if (generationState == "USER_WRITING") {
      uiStatus = "success";
    } else if (generationState == "GENERATING") {
      uiStatus = "info";
    } else if (generationState == "NOT_READY") {
      uiStatus = "error";
    }

    const [advancedMode, setAdvancedMode] = useAdvancedMode();

    const [tutorialBanner, setTutorialBanner] = useState(true);
    const [examplePrompt, setExamplePrompt] = useState<string | null>(null);

    const chatRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      chatRef?.current?.scrollIntoView({
        behavior: "smooth",
        block: "end",
        inline: "end",
      });
    }, [modelOutput]);
    return (
      <>
        <div className="flex-1 flex flex-col min-h-[calc(100vh-150px)] max-h-[calc(100vh-150px)]">
          <div
            className={`flex-1 flex flex-col relative overflow-x-hidden mb-10`}
          >
            <div
              className={`flex-1 flex flex-row items-stretch gap-4 max-h-[calc(100vh-200px)] ${
                advancedMode ? "ml-[500px]" : "ml-0"
              } transition-all`}
            >
              <div className="flex-1 flex flex-col relative rounded-md px-6 py-4 bg-purple-50 gap-8">
                <div className="flex flex-row items-center justify-between">
                  <div className="prose">
                    <h4>Input</h4>
                  </div>
                  <SettingsAdjust
                    onClick={() => setAdvancedMode(!advancedMode)}
                    size={24}
                    className="hover:fill-primary cursor-pointer"
                  />
                </div>
                <div className="flex flex-col flex-1 items-stretch overflow-y-scroll h-full">
                  <LexicalComposer initialConfig={initialConfig}>
                    {/* Toolbar on top, if needed */}
                    {/* <ToolbarPlugin /> */}

                    <div className="relative flex-1">
                      <RichTextPlugin
                        contentEditable={
                          <ContentEditable
                            className={`relative bg-white ${
                              tutorialBanner ? "rounded-t-md" : "rounded-md"
                            } block p-4 leading-5 text-md h-full`}
                          />
                        }
                        placeholder={<Placeholder />}
                        ErrorBoundary={LexicalErrorBoundary}
                      />
                    </div>
                    <DragDropPaste />
                    <HistoryPlugin />
                    <ImagesPlugin />
                    <OnChangePlugin onChange={onChange} />
                    <ReplaceContentPlugin payload={examplePrompt} />
                  </LexicalComposer>
                </div>
                <div className="flex flex-row items-center justify-between my-4 gap-2">
                  <div className="flex flex-row items-center gap-2">
                    <button
                      onClick={onRunModelClick}
                      disabled={runButtonDisabled}
                      className={"btn" + " " + runButtonColor}
                    >
                      {runButtonText}
                    </button>
                    <button onClick={abortGeneration} className="btn btn-ghost">
                      Abort
                    </button>
                  </div>
                  {!tutorialBanner && (
                    <button
                      className="btn btn-circle bg-white border-none"
                      onClick={() => setTutorialBanner(true)}
                    >
                      <Idea size={24} />
                    </button>
                  )}
                </div>
              </div>

              {/* Results */}

              <div className="flex-1 flex flex-col bg-gray-50 rounded-md overflow-x-hidden px-6 py-4 max-h-[calc(100vh-200px)] ">
                <div className="prose">
                  <h4>Output</h4>
                </div>
                <div className="mt-6 overflow-scroll flex-1 leading-relaxed markdown">
                  {numberInQueue && numberInQueue > 0 && (
                    <div
                      role="alert"
                      className="p-4 mb-4 text-med rounded-lg bg-purple-50"
                    >
                      There are {numberInQueue} other users in the queue for
                      generation.
                    </div>
                  )}
                  <div className="prose leading-snug">
                    {mergeTextContent(modelOutput).map(contentToHtml)}
                  </div>

                  <ImageResult src={partialImage} completed={false} />
                </div>
              </div>
            </div>

            {/* Side panel */}

            <div
              className={`absolute top-0 bottom-11 w-[490px] max-h-[calc(100vh-200px)] rounded-md px-6 py-4 overflow-y-scroll ${
                advancedMode ? "left-0" : "left-[-500px]"
              } bg-gray-100 transition-all`}
            >
              <div className="prose flex flex-row items-center justify-between">
                <h3>Advanced settings</h3>
                <Close
                  size={32}
                  className="cursor-pointer hover:fill-primary"
                  onClick={() => setAdvancedMode(false)}
                />
              </div>
              <InputRange
                value={temp}
                onValueChange={setTemp}
                label="Temperature"
                min={0.01}
                step={0.01}
                max={1}
              />
              <InputRange
                value={topP}
                onValueChange={setTopP}
                label="Top P"
                min={0.01}
                step={0.01}
                max={1}
              />
              <InputRange
                value={maxGenTokens}
                onValueChange={setMaxGenTokens}
                label="Max Gen Tokens"
                integerOnly
                step={1}
                min={1}
                max={4096}
              />
              <InputRange
                value={repetitionPenalty}
                onValueChange={setRepetitionPenalty}
                label="Text Repetition Penalty"
                min={0}
                max={10}
              />
              <InputRange
                value={cfgImageWeight}
                onValueChange={setCfgImageWeight}
                label="CFG Image Weight"
                min={0.01}
                max={10}
              />
              <InputRange
                value={cfgTextWeight}
                onValueChange={setCfgTextWeight}
                label="CFG Text Weight"
                min={0.01}
                max={10}
              />
              <InputToggle
                label="Set seed"
                value={showSeed}
                onValueChange={(checked) => {
                  setShowSeed(checked);
                }}
              />
              {showSeed && seed != null && (
                <InputRange
                  value={seed}
                  step={1}
                  integerOnly={true}
                  onValueChange={setSeed}
                  label="Seed"
                  min={1}
                  max={1000}
                />
              )}

              {/* Input preview */}

              <InputShowHide
                labelShow="Show input data"
                labelHide="Hide input data"
              >
                <div className="overflow-auto bg-white text-xs font-mono p-4 rounded-md mt-4">
                  <JsonView
                    src={contents}
                    collapsed={({ node, indexOrName, depth, size }) =>
                      indexOrName !== "data" && depth > 3
                    }
                  />
                </div>
              </InputShowHide>
            </div>
          </div>
          <div className="absolute bottom-0 left-20 right-20 bg-white flex flex-row  items-center gap-4 text-xs h-10">
            <StatusBadge
              label="Connection"
              category={socketStatus}
              status={readableSocketState}
            />

            <StatusBadge
              label="UI"
              category={uiStatus}
              status={generationState}
            />
          </div>
        </div>
      </>
    );
  }

  return <Editor />;
}
