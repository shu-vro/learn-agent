"use client";

import dynamic from "next/dynamic";

import type { MarkdownProps } from "@/components/markdown";

/**
 * AI-elements message markdown surface. Lazily loads the shared Markdown
 * renderer (Streamdown + code / mermaid / math / cjk).
 */
const MessageResponseLazy = dynamic(
  () => import("./message-response-inner").then((m) => m.MessageResponseInner),
  {
    ssr: false,
    loading: () => (
      <p className="text-muted-foreground text-sm">Loading preview…</p>
    ),
  },
);

export type MessageResponseProps = MarkdownProps;

export function MessageResponse({ className, ...props }: MessageResponseProps) {
  return <MessageResponseLazy className={className} {...props} />;
}
