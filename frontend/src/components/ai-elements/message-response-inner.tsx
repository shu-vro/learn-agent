"use client";

import { cjk } from "@streamdown/cjk";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { type ComponentProps, memo } from "react";
import { Streamdown } from "streamdown";

import { normalizeMathDelimiters } from "@/lib/math-markdown";
import { cn } from "@/lib/utils";

const streamdownPlugins = { cjk, code, math, mermaid };

export type MessageResponseInnerProps = ComponentProps<typeof Streamdown>;

export const MessageResponseInner = memo(
  ({ className, children, ...props }: MessageResponseInnerProps) => (
    <Streamdown
      className={cn(
        "size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        className,
      )}
      plugins={streamdownPlugins}
      {...props}
    >
      {typeof children === "string"
        ? normalizeMathDelimiters(children)
        : children}
    </Streamdown>
  ),
  (prevProps, nextProps) =>
    prevProps.children === nextProps.children &&
    nextProps.isAnimating === prevProps.isAnimating,
);

MessageResponseInner.displayName = "MessageResponseInner";
