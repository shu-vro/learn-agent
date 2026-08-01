"use client";

import { CopyIcon, RefreshCcwIcon } from "lucide-react";
import dynamic from "next/dynamic";
import type { FormEvent } from "react";
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
import { UsageDetailsButton } from "@/components/chat/usage-details";
import type {
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
}: {
  message: ChatMessage;
  branchContent: string;
  timeline?: ChatTimelineItem[];
  thinking?: string;
  tools?: ChatToolCall[];
  streaming?: boolean;
  activeThinkingStep?: number | null;
}) {
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
          <MessageResponse isAnimating={Boolean(streaming)}>
            {branchContent || (streaming ? "…" : "")}
          </MessageResponse>
        </MessageContent>
      ) : null}
    </>
  );
}

function AssistantMessage({
  message,
  onRegenerate,
  onBranchChange,
}: {
  message: ChatMessage;
  onRegenerate: (messageId: string) => void;
  onBranchChange: (index: number) => void;
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
  const regenerateId = current?.id ?? message.id;

  if (hasBranches) {
    return (
      <Message from="assistant">
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
              {usage ? <UsageDetailsButton usage={usage} /> : null}
            </div>
          </MessageActions>
        </MessageBranch>
      </Message>
    );
  }

  return (
    <Message from="assistant">
      <AssistantBody
        message={message}
        branchContent={content}
        timeline={timeline}
        thinking={thinking}
        tools={tools}
        streaming={streaming}
        activeThinkingStep={activeThinkingStep}
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
        {usage ? <UsageDetailsButton usage={usage} /> : null}
      </MessageActions>
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
    messages,
    appendUserMessage,
    regenerateMessage,
    setActiveBranch,
    isStreaming,
  } = useChatWorkspace();

  return (
    <div
      className={cn(
        "flex h-full min-h-0 flex-col bg-background dark:bg-black",
        className,
      )}
    >
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
                <Message key={m.id} from="user">
                  <MessageContent>
                    {m.selection ? (
                      <blockquote className="mb-2 border-border border-l-2 pl-3 text-muted-foreground text-xs">
                        {m.selection}
                      </blockquote>
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
            onSubmit={(text, e: FormEvent<HTMLFormElement>, images) => {
              e.preventDefault();
              if (isStreaming) return;
              appendUserMessage(text, {
                images: images?.length ? images : undefined,
              });
            }}
          />
        </div>
      </div>
    </div>
  );
}
