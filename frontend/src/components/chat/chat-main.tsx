"use client";

import { CopyIcon, RefreshCcwIcon, Volume2Icon } from "lucide-react";
import dynamic from "next/dynamic";
import type { FormEvent, MouseEvent } from "react";
import { useEffect, useState } from "react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageAction,
  MessageActions,
  MessageBranch,
  MessageBranchContent,
  MessageBranchNext,
  MessageBranchPage,
  MessageBranchPrevious,
  MessageBranchSelector,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { useChatWorkspace } from "@/components/chat/chat-context";
import { MessageSources } from "@/components/chat/message-sources";
import {
  messageAnchorId,
  type ReadAloud,
  ReadAloudControls,
  useReadAloud,
} from "@/components/chat/read-aloud";
import { UsageDetailsButton } from "@/components/chat/usage-details";
import type {
  ChatArtifact,
  ChatMessage,
  ChatTimelineItem,
  ChatToolCall,
} from "@/lib/api/chat";
import { cn } from "@/lib/utils";

const ChatPrompt = dynamic(
  () =>
    import("@/components/chat/chat-prompt").then((m) => ({
      default: m.ChatPrompt,
    })),
  {
    ssr: false,
    loading: () => (
      <div className="h-24 animate-pulse rounded-xl bg-muted/40" />
    ),
  },
);

const ANCHOR_PREFIX = "message-";

export type PendingReference = { id: string; text: string };

/** Floating "Add as reference" button over any selection inside a message. */
function SelectionTooltip({
  onAdd,
}: {
  onAdd: (reference: PendingReference) => void;
}) {
  const [hit, setHit] = useState<
    (PendingReference & { top: number; left: number }) | null
  >(null);

  useEffect(() => {
    const sync = () => {
      const selection = window.getSelection();
      const text = selection?.toString().trim() ?? "";
      if (!selection || selection.isCollapsed || !text) {
        setHit(null);
        return;
      }
      const range = selection.getRangeAt(0);
      const node = range.commonAncestorContainer;
      const el = (
        node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement
      ) as HTMLElement | null;
      const anchor = el?.closest<HTMLElement>(`[id^="${ANCHOR_PREFIX}"]`);
      if (!anchor) {
        setHit(null);
        return;
      }
      const rect = range.getBoundingClientRect();
      setHit({
        id: anchor.id.slice(ANCHOR_PREFIX.length),
        text,
        top: rect.top,
        left: rect.left + rect.width / 2,
      });
    };

    document.addEventListener("selectionchange", sync);
    window.addEventListener("scroll", sync, true);
    return () => {
      document.removeEventListener("selectionchange", sync);
      window.removeEventListener("scroll", sync, true);
    };
  }, []);

  if (!hit) {
    return null;
  }

  return (
    <button
      type="button"
      className="-translate-x-1/2 -translate-y-full fixed z-50 rounded-md border border-border bg-popover px-2 py-1 text-popover-foreground text-xs shadow-md"
      style={{ top: hit.top - 8, left: hit.left }}
      onMouseDown={(e) => e.preventDefault()}
      onClick={() => {
        onAdd({ id: hit.id, text: hit.text });
        window.getSelection()?.removeAllRanges();
        setHit(null);
      }}
    >
      Add as reference
    </button>
  );
}

function ToolCallView({ tool }: { tool: ChatToolCall }) {
  return (
    <Tool defaultOpen={tool.state !== "output-available"}>
      <ToolHeader
        title={tool.name}
        type="dynamic-tool"
        state={tool.state}
        toolName={tool.name}
      />
      <ToolContent>
        <ToolInput input={tool.args} />
        {tool.state === "output-available" ? (
          <ToolOutput
            output={
              <MessageResponse>
                {typeof tool.result === "string"
                  ? tool.result
                  : JSON.stringify(tool.result, null, 2)}
              </MessageResponse>
            }
            errorText={undefined}
          />
        ) : null}
      </ToolContent>
    </Tool>
  );
}

function AssistantBody({
  message,
  branchContent,
  timeline,
  thinking,
  tools,
  streaming,
  activeThinkingStep,
  artifacts,
}: {
  message: ChatMessage;
  branchContent: string;
  timeline?: ChatTimelineItem[];
  thinking?: string;
  tools?: ChatToolCall[];
  streaming?: boolean;
  activeThinkingStep?: number | null;
  artifacts?: ChatArtifact[];
}) {
  const { focusChunk } = useChatWorkspace();
  // Document citations render as `[Source n](reference_id=<doc>:<chunk>)`; open
  // them in the preview panel instead of letting the browser follow the href.
  const onCitationClick = (event: MouseEvent<HTMLDivElement>) => {
    const anchor = (event.target as HTMLElement).closest("a");
    const href = anchor?.getAttribute("href") ?? "";
    if (!href.startsWith("reference_id=")) {
      return;
    }
    event.preventDefault();
    const artifact = (artifacts ?? message.artifacts ?? []).find(
      (a) => a.url === href,
    );
    if (artifact?.documentId && artifact.chunkUuid) {
      focusChunk(artifact.documentId, artifact.chunkUuid);
    }
  };

  const items: ChatTimelineItem[] =
    timeline && timeline.length > 0
      ? timeline
      : [
          ...(thinking
            ? [
                {
                  kind: "thinking" as const,
                  id: "think-0",
                  step: 0,
                  text: thinking,
                },
              ]
            : []),
          ...(tools ?? []).map((t) => ({ kind: "tool" as const, ...t })),
        ];

  return (
    <>
      {items.map((item) => {
        if (item.kind === "thinking") {
          const isLive = Boolean(streaming) && activeThinkingStep === item.step;
          return (
            <Reasoning key={item.id} className="w-full" isStreaming={isLive}>
              <ReasoningTrigger />
              <ReasoningContent>{item.text}</ReasoningContent>
            </Reasoning>
          );
        }
        return <ToolCallView key={`${message.id}-${item.id}`} tool={item} />;
      })}
      {branchContent || streaming ? (
        <MessageContent>
          {/* biome-ignore lint/a11y/noStaticElementInteractions: click delegation for markdown-rendered anchors, which stay keyboard-activatable themselves */}
          {/* biome-ignore lint/a11y/useKeyWithClickEvents: the anchors handle keyboard activation */}
          <div onClick={onCitationClick}>
            <MessageResponse isAnimating={Boolean(streaming)}>
              {branchContent || (streaming ? "…" : "")}
            </MessageResponse>
          </div>
        </MessageContent>
      ) : null}
    </>
  );
}

function AssistantMessage({
  message,
  onRegenerate,
  onBranchChange,
  reader,
}: {
  message: ChatMessage;
  onRegenerate: (messageId: string) => void;
  onBranchChange: (index: number) => void;
  reader: ReadAloud;
}) {
  const branches = message.branches ?? [];
  const hasBranches = branches.length > 1;
  const active = message.activeBranch ?? Math.max(0, branches.length - 1);
  const current = branches[active];
  const content = current?.content ?? message.content;
  const timeline = current?.timeline ?? message.timeline;
  const thinking = current?.thinking ?? message.thinking;
  const tools = current?.tools ?? message.tools;
  const streaming = current?.streaming ?? message.streaming;
  const activeThinkingStep =
    current?.activeThinkingStep ?? message.activeThinkingStep;
  const usage = current?.usage ?? message.usage;
  const artifacts = current?.artifacts ?? message.artifacts ?? [];
  const regenerateId = current?.id ?? message.id;

  const reading = reader.activeId === regenerateId;

  if (hasBranches) {
    return (
      <Message from="assistant" id={messageAnchorId(regenerateId)}>
        <MessageBranch
          key={`${message.id}-${branches.length}`}
          defaultBranch={active}
          onBranchChange={onBranchChange}
        >
          <MessageBranchContent>
            {branches.map((branch) => (
              <div className="flex w-full flex-col gap-2" key={branch.id}>
                <AssistantBody
                  message={message}
                  branchContent={branch.content}
                  timeline={branch.timeline}
                  thinking={branch.thinking}
                  tools={branch.tools}
                  streaming={branch.streaming}
                  activeThinkingStep={branch.activeThinkingStep}
                  artifacts={branch.artifacts}
                />
              </div>
            ))}
          </MessageBranchContent>
          <MessageActions>
            <MessageBranchSelector>
              <MessageBranchPrevious />
              <MessageBranchPage />
              <MessageBranchNext />
            </MessageBranchSelector>
            <div className="flex items-center gap-1">
              <MessageAction
                label="Retry"
                tooltip="Regenerate"
                onClick={() => onRegenerate(regenerateId)}
                disabled={Boolean(streaming)}
              >
                <RefreshCcwIcon className="size-3" />
              </MessageAction>
              <MessageAction
                label="Copy"
                tooltip="Copy"
                onClick={() => navigator.clipboard.writeText(content)}
              >
                <CopyIcon className="size-3" />
              </MessageAction>
              <MessageAction
                label="Read aloud"
                tooltip="Read aloud"
                onClick={() => reader.start(regenerateId, message.chatId ?? "")}
                disabled={Boolean(streaming) || !message.chatId}
              >
                <Volume2Icon className="size-3" />
              </MessageAction>
              {usage ? <UsageDetailsButton usage={usage} /> : null}
            </div>
          </MessageActions>
          <MessageSources artifacts={artifacts} content={content} />
          {reading ? <ReadAloudControls reader={reader} /> : null}
        </MessageBranch>
      </Message>
    );
  }

  return (
    <Message from="assistant" id={messageAnchorId(regenerateId)}>
      <AssistantBody
        message={message}
        branchContent={content}
        timeline={timeline}
        thinking={thinking}
        tools={tools}
        streaming={streaming}
        activeThinkingStep={activeThinkingStep}
        artifacts={artifacts}
      />
      <MessageActions>
        <MessageAction
          label="Retry"
          tooltip="Regenerate"
          onClick={() => onRegenerate(regenerateId)}
          disabled={Boolean(streaming)}
        >
          <RefreshCcwIcon className="size-3" />
        </MessageAction>
        <MessageAction
          label="Copy"
          tooltip="Copy"
          onClick={() => navigator.clipboard.writeText(content)}
        >
          <CopyIcon className="size-3" />
        </MessageAction>
        <MessageAction
          label="Read aloud"
          tooltip="Read aloud"
          onClick={() => reader.start(regenerateId, message.chatId ?? "")}
          disabled={Boolean(streaming) || !message.chatId}
        >
          <Volume2Icon className="size-3" />
        </MessageAction>
        {usage ? <UsageDetailsButton usage={usage} /> : null}
      </MessageActions>
      <MessageSources artifacts={artifacts} content={content} />
      {reading ? <ReadAloudControls reader={reader} /> : null}
    </Message>
  );
}

export function ChatMain({
  className,
  promptGlobalDrop = true,
}: {
  className?: string;
  promptGlobalDrop?: boolean;
}) {
  const {
    projectId,
    messages,
    appendUserMessage,
    regenerateMessage,
    setActiveBranch,
    isStreaming,
  } = useChatWorkspace();
  const reader = useReadAloud(projectId);
  const [reference, setReference] = useState<PendingReference | null>(null);

  return (
    <div
      className={cn(
        "flex h-full min-h-0 flex-col bg-background dark:bg-black",
        className,
      )}
    >
      <SelectionTooltip onAdd={setReference} />
      <div className="flex shrink-0 items-center border-border/30 border-b px-4 py-3">
        <h1 className="font-medium text-sm">Chat</h1>
      </div>
      <Conversation className="min-h-0 flex-1">
        <ConversationContent className="mx-auto w-full max-w-3xl gap-6 px-4 py-6">
          {messages.length === 0 ? (
            <ConversationEmptyState
              title="No messages yet"
              description="Ask anything to start this thread."
            />
          ) : (
            messages.map((m) =>
              m.role === "user" ? (
                <Message key={m.id} from="user" id={messageAnchorId(m.id)}>
                  <MessageContent>
                    {m.selection ? (
                      <button
                        type="button"
                        title="Jump to referenced message"
                        className="mb-2 block w-full border-border border-l-2 pl-3 text-left text-muted-foreground text-xs hover:text-foreground"
                        onClick={() =>
                          m.referenceId &&
                          document
                            .getElementById(messageAnchorId(m.referenceId))
                            ?.scrollIntoView({
                              behavior: "smooth",
                              block: "center",
                            })
                        }
                      >
                        {m.selection}
                      </button>
                    ) : null}
                    {m.imageUrls?.length ? (
                      <div className="mb-2 flex flex-wrap gap-2">
                        {m.imageUrls.map((url) => (
                          // biome-ignore lint/performance/noImgElement: dynamic S3 URLs; next/image needs configured remote patterns
                          <img
                            key={url}
                            src={url}
                            alt="Attached"
                            className="max-h-40 max-w-full rounded-md object-contain"
                          />
                        ))}
                      </div>
                    ) : null}
                    {m.content ? (
                      <p className="whitespace-pre-wrap">{m.content}</p>
                    ) : null}
                  </MessageContent>
                </Message>
              ) : (
                <AssistantMessage
                  key={m.id}
                  message={m}
                  onRegenerate={(messageId) =>
                    regenerateMessage(m.id, messageId)
                  }
                  onBranchChange={(index) => setActiveBranch(m.id, index)}
                  reader={reader}
                />
              ),
            )
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="shrink-0 border-border/30 border-t px-4 py-4">
        <div className="mx-auto w-full max-w-3xl">
          <ChatPrompt
            globalDrop={promptGlobalDrop}
            disabled={isStreaming}
            reference={reference}
            onClearReference={() => setReference(null)}
            onSubmit={(text, e: FormEvent<HTMLFormElement>, images) => {
              e.preventDefault();
              if (isStreaming) return;
              appendUserMessage(text, {
                images: images?.length ? images : undefined,
                selection: reference?.text,
                referenceId: reference?.id,
              });
              setReference(null);
            }}
          />
        </div>
      </div>
    </div>
  );
}
