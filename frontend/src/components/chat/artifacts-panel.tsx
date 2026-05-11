"use client";

import { UploadIcon } from "lucide-react";
import { useRef } from "react";

import {
  Artifact,
  ArtifactHeader,
  ArtifactTitle,
} from "@/components/ai-elements/artifact";
import { MessageResponse } from "@/components/ai-elements/message";
import { useChatWorkspace } from "@/components/chat/chat-context";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export function ArtifactsPanel({ className }: { className?: string }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const {
    artifacts,
    selectedArtifactId,
    setSelectedArtifactId,
    addArtifactFromFile,
  } = useChatWorkspace();

  const selected = artifacts.find((a) => a.id === selectedArtifactId) ?? null;

  return (
    <div
      className={cn("flex h-full min-h-0 flex-col bg-sidebar/80", className)}>
      <div className="flex shrink-0 items-center justify-between gap-2 border-border/40 border-b px-3 py-3">
        <span className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
          Files
        </span>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          multiple
          onChange={async (e) => {
            const files = e.target.files;
            if (!files?.length) {
              return;
            }
            for (const file of [...files]) {
              await addArtifactFromFile(file);
            }
            e.target.value = "";
          }}
        />
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="rounded-xl text-muted-foreground hover:text-foreground"
          onClick={() => inputRef.current?.click()}>
          <UploadIcon className="size-4" />
          <span className="sr-only">Upload</span>
        </Button>
      </div>

      <ScrollArea className="min-h-0 flex-1 p-3">
        {selected ? (
          <Artifact className="border-border/40 bg-background/60">
            <ArtifactHeader className="border-border/40 bg-transparent py-2">
              <ArtifactTitle>{selected.name}</ArtifactTitle>
            </ArtifactHeader>
            <div className="max-h-[calc(100vh-12rem)] overflow-auto px-4 pb-4">
              <MessageResponse className="text-sm">
                {selected.content}
              </MessageResponse>
            </div>
          </Artifact>
        ) : (
          <p className="text-muted-foreground text-sm">
            Upload a file or select a chip to preview markdown.
          </p>
        )}
      </ScrollArea>
    </div>
  );
}
