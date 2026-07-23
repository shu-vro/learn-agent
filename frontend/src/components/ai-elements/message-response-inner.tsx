"use client";

import { type ComponentProps, memo } from "react";

import { Markdown } from "@/components/markdown";
import { cn } from "@/lib/utils";

export type MessageResponseInnerProps = ComponentProps<typeof Markdown>;

export const MessageResponseInner = memo(
  ({ className, ...props }: MessageResponseInnerProps) => (
    <Markdown
      className={cn(
        "size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        className,
      )}
      {...props}
    />
  ),
  (prevProps, nextProps) =>
    prevProps.children === nextProps.children &&
    nextProps.isAnimating === prevProps.isAnimating,
);

MessageResponseInner.displayName = "MessageResponseInner";
