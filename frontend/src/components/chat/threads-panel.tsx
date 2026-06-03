"use client";

import { EllipsisVertical, MessageSquarePlusIcon } from "lucide-react";
import { useState } from "react";

import { useChatWorkspace } from "@/components/chat/chat-context";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { threadDisplayName } from "@/lib/api/chat";
import { cn } from "@/lib/utils";

export function ThreadsPanel({ className }: { className?: string }) {
  const {
    threads,
    activeThreadId,
    setActiveThreadId,
    newThread,
    renameThread,
    deleteThread,
  } = useChatWorkspace();

  const [renameOpen, setRenameOpen] = useState(false);
  const [renameThreadId, setRenameThreadId] = useState<string | null>(null);
  const [renameName, setRenameName] = useState("");
  const [renameSaving, setRenameSaving] = useState(false);

  const openRename = (threadId: string, currentName: string) => {
    setRenameThreadId(threadId);
    setRenameName(currentName);
    setRenameOpen(true);
  };

  const handleRenameSubmit = async () => {
    if (!renameThreadId || !renameName.trim()) {
      return;
    }
    setRenameSaving(true);
    try {
      await renameThread(renameThreadId, renameName.trim());
      setRenameOpen(false);
    } finally {
      setRenameSaving(false);
    }
  };

  return (
    <div
      className={cn("flex h-full min-h-0 flex-col bg-sidebar/80", className)}
    >
      <div className="flex shrink-0 items-center justify-between gap-2 border-border/40 border-b px-3 py-3">
        <span className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
          Threads
        </span>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="rounded-xl text-muted-foreground hover:text-foreground"
          onClick={() => void newThread()}
        >
          <MessageSquarePlusIcon className="size-4" />
          <span className="sr-only">New thread</span>
        </Button>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <ul className="p-2">
          {threads.map((t) => (
            <li key={t.id}>
              <div
                className={cn(
                  "flex items-center gap-1 rounded-xl transition-colors",
                  t.id === activeThreadId
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                )}
              >
                <button
                  type="button"
                  onClick={() => setActiveThreadId(t.id)}
                  className="min-w-0 flex-1 truncate px-3 py-2.5 text-left text-sm"
                >
                  {threadDisplayName(t)}
                </button>
                <DropdownMenu>
                  <DropdownMenuTrigger>
                    <button
                      type="button"
                      onClick={(e) => e.stopPropagation()}
                      className="mr-1 rounded-xl p-1 text-muted-foreground hover:text-foreground"
                      aria-label="Thread menu"
                    >
                      <EllipsisVertical className="size-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        openRename(t.id, t.thread_name);
                      }}
                    >
                      Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      data-variant="destructive"
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        void deleteThread(t.id);
                      }}
                    >
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </li>
          ))}
        </ul>
      </ScrollArea>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent showCloseButton>
          <DialogHeader>
            <DialogTitle>Rename thread</DialogTitle>
          </DialogHeader>
          <div className="grid gap-2">
            <Input
              aria-label="Name"
              placeholder="Name"
              value={renameName}
              onChange={(e) => setRenameName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void handleRenameSubmit();
                }
              }}
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setRenameOpen(false)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              disabled={!renameName.trim() || renameSaving}
              onClick={() => void handleRenameSubmit()}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
