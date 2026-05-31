"use client";

import dynamic from "next/dynamic";
import type { ComponentProps } from "react";
import type { Streamdown } from "streamdown";

const MessageResponseLazy = dynamic(
  () => import("./message-response-inner").then((m) => m.MessageResponseInner),
  {
    ssr: false,
    loading: () => (
      <p className="text-muted-foreground text-sm">Loading preview…</p>
    ),
  },
);

export type MessageResponseProps = ComponentProps<typeof Streamdown>;

export function MessageResponse({ className, ...props }: MessageResponseProps) {
  return <MessageResponseLazy className={className} {...props} />;
}
