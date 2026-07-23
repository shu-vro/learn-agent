"use client";

import { cjk } from "@streamdown/cjk";
import { createCodePlugin } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { type ComponentProps, memo } from "react";
import { Streamdown } from "streamdown";

import { normalizeMathDelimiters } from "@/lib/math-markdown";
import { cn } from "@/lib/utils";

const code = createCodePlugin({
  themes: ["github-light", "github-dark"],
});

const plugins = { cjk, code, math, mermaid };

export type MarkdownProps = ComponentProps<typeof Streamdown>;

/**
 * Project-wide markdown renderer (Streamdown + code / mermaid / math / cjk).
 * Prefer this over importing Streamdown or plugins directly.
 */
export const Markdown = memo(
  ({ className, children, ...props }: MarkdownProps) => (
    <Streamdown className={cn(className)} plugins={plugins} {...props}>
      {typeof children === "string"
        ? normalizeMathDelimiters(children)
        : children}
    </Streamdown>
  ),
  (prevProps, nextProps) =>
    prevProps.children === nextProps.children &&
    prevProps.isAnimating === nextProps.isAnimating &&
    prevProps.className === nextProps.className,
);

Markdown.displayName = "Markdown";
