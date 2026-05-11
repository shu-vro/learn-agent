"use client";

import { MessageSquarePlusIcon } from "lucide-react";

import { useChatWorkspace } from "@/components/chat/chat-context";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export function ThreadsPanel({ className }: { className?: string }) {
  const { threads, activeThreadId, setActiveThreadId, newThread } =
    useChatWorkspace();

  return (
    <div
      className={cn("flex h-full min-h-0 flex-col bg-sidebar/80", className)}>
      <div className="flex shrink-0 items-center justify-between gap-2 border-border/40 border-b px-3 py-3">
        <span className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
          Threads
        </span>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="rounded-xl text-muted-foreground hover:text-foreground"
          onClick={newThread}>
          <MessageSquarePlusIcon className="size-4" />
          <span className="sr-only">New thread</span>
        </Button>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <ul className="p-2">
          {threads.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                onClick={() => setActiveThreadId(t.id)}
                className={cn(
                  "w-full rounded-xl px-3 py-2.5 text-left text-sm transition-colors",
                  t.id === activeThreadId
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                )}>
                {t.title}
              </button>
            </li>
          ))}
        </ul>
      </ScrollArea>
    </div>
  );
}
