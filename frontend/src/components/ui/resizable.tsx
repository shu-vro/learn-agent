"use client";

import type { ComponentProps } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";
import { cn } from "@/lib/utils";

function ResizablePanelGroup({
  className,
  ...props
}: ComponentProps<typeof Group>) {
  return (
    <Group
      className={cn(
        "flex min-h-0 w-full min-w-0 flex-1 flex-row overflow-hidden",
        className,
      )}
      {...props}
    />
  );
}

const ResizablePanel = Panel;

function ResizableHandle({
  className,
  ...props
}: ComponentProps<typeof Separator>) {
  return (
    <Separator
      className={cn(
        "relative z-10 w-2 shrink-0 bg-sidebar/80 transition-colors hover:bg-border",
        "outline-none focus-visible:bg-border focus-visible:ring-2 focus-visible:ring-ring/60",
        "data-[separator]:cursor-col-resize",
        className,
      )}
      {...props}
    />
  );
}

export { ResizablePanelGroup, ResizablePanel, ResizableHandle };
