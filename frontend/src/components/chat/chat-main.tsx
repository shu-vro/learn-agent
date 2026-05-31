"use client";

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
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import { useChatWorkspace } from "@/components/chat/chat-context";
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

export function ChatMain({ className }: { className?: string }) {
  const { messages, appendUserMessage } = useChatWorkspace();

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
                    <p className="whitespace-pre-wrap">{m.content}</p>
                  </MessageContent>
                </Message>
              ) : (
                <Message key={m.id} from="assistant">
                  <MessageContent>
                    <MessageResponse>{m.content}</MessageResponse>
                  </MessageContent>
                </Message>
              ),
            )
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="shrink-0 border-border/30 border-t px-4 py-4">
        <div className="mx-auto w-full max-w-3xl">
          <ChatPrompt
            onSubmit={(text, e: FormEvent<HTMLFormElement>) => {
              e.preventDefault();
              appendUserMessage(text);
            }}
          />
        </div>
      </div>
    </div>
  );
}
