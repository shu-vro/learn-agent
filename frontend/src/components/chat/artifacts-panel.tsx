"use client";

import { ChevronDownIcon, UploadIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  Artifact,
  ArtifactHeader,
  ArtifactTitle,
} from "@/components/ai-elements/artifact";
import { MessageResponse } from "@/components/ai-elements/message";
import { useChatWorkspace } from "@/components/chat/chat-context";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { defaultRehypePlugins } from "streamdown";
import { harden } from "rehype-harden";

function artifactPreviewMarkdown(artifact: {
  chunks: Record<string, string>;
  ingestion_status?: string;
}): string {
  const text = Object.values(artifact.chunks)
    .map((chunk) => {
      return chunk.replaceAll("!\n\n[", "![").trim();
    })
    .filter(Boolean)
    .join("\n\n");
  if (text) {
    return text;
  }
  if (artifact.ingestion_status === "processing") {
    return "_This document is being processed. It will appear here when ready._";
  }
  if (artifact.ingestion_status === "failed") {
    return "_This document failed to process. Try uploading again._";
  }
  return "_No text content available for this file._";
}

function isDesktopViewport() {
  if (typeof window === "undefined") {
    return false;
  }
  return window.matchMedia("(min-width: 768px)").matches;
}

export function ArtifactsPanel({ className }: { className?: string }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const { artifacts, selectedArtifactId, addArtifactFromFile } =
    useChatWorkspace();

  const selected = artifacts.find((a) => a.id === selectedArtifactId) ?? null;
  const [previewOpen, setPreviewOpen] = useState(true);

  useEffect(() => {
    setPreviewOpen(true);
  }, [selectedArtifactId]);

  const processFiles = useCallback(
    async (files: FileList | File[] | null | undefined) => {
      if (!files?.length) {
        return;
      }
      const list = files instanceof FileList ? [...files] : files;
      for (const file of list) {
        await addArtifactFromFile(file);
      }
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    },
    [addArtifactFromFile],
  );

  const onUploadButtonClick = () => {
    if (isDesktopViewport()) {
      setUploadOpen(true);
      return;
    }
    inputRef.current?.click();
  };

  return (
    <div
      className={cn("flex h-full min-h-0 flex-col bg-sidebar/80", className)}>
      <Dialog onOpenChange={setUploadOpen} open={uploadOpen}>
        <DialogContent className="gap-4 sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Add files</DialogTitle>
            <DialogDescription>
              Drag PDF or Markdown files here, or choose them from your device.
            </DialogDescription>
          </DialogHeader>
          <div
            className="flex cursor-default flex-col items-center justify-center gap-4 rounded-2xl border border-dashed border-border/60 bg-muted/30 px-6 py-10 text-center transition-colors hover:border-border hover:bg-muted/40"
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onDrop={async (e) => {
              e.preventDefault();
              e.stopPropagation();
              await processFiles(e.dataTransfer.files);
              setUploadOpen(false);
            }}>
            <UploadIcon className="size-8 text-muted-foreground" />
            <p className="text-muted-foreground text-sm">
              Drop PDF or Markdown files to upload
            </p>
            <Button
              type="button"
              variant="secondary"
              onClick={() => inputRef.current?.click()}>
              Select files
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <div className="flex shrink-0 items-center justify-between gap-2 border-border/40 border-b px-3 py-3">
        <span className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
          Files
        </span>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept=".pdf,.md,.markdown,application/pdf,text/markdown"
          multiple
          onChange={async (e) => {
            await processFiles(e.target.files);
            setUploadOpen(false);
          }}
        />
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="rounded-xl text-muted-foreground hover:text-foreground"
          onClick={onUploadButtonClick}>
          <UploadIcon className="size-4" />
          <span className="sr-only">Upload</span>
        </Button>
      </div>

      <ScrollArea className="min-h-0 flex-1 p-3">
        {selected ? (
          <Artifact className="border-border/40 bg-background/60">
            <Collapsible
              className="flex min-h-0 flex-col"
              onOpenChange={setPreviewOpen}
              open={previewOpen}>
              <ArtifactHeader className="border-border/40 bg-transparent p-0">
                <CollapsibleTrigger
                  aria-expanded={previewOpen}
                  aria-label={`${previewOpen ? "Collapse" : "Expand"} preview: ${selected.name}`}
                  className="flex w-full items-center gap-2 px-4 py-2 text-left transition-colors hover:bg-muted/40">
                  <ChevronDownIcon
                    aria-hidden
                    className={cn(
                      "size-4 shrink-0 text-muted-foreground transition-transform",
                      previewOpen && "rotate-180",
                    )}
                  />
                  <ArtifactTitle className="min-w-0 flex-1 truncate border-0 py-0">
                    {selected.name}
                  </ArtifactTitle>
                </CollapsibleTrigger>
              </ArtifactHeader>
              <CollapsibleContent className="min-h-0 overflow-hidden data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0">
                <div className="max-h-[calc(100vh-12rem)] overflow-auto px-4 pb-4">
                  <MessageResponse
                    className="text-sm"
                    components={{
                      img: ({ src, alt, ...props }) => {
                        return <img src={src} alt={alt} {...props} />;
                      },
                    }}
                    rehypePlugins={[
                      defaultRehypePlugins.raw,
                      defaultRehypePlugins.sanitize,
                      [
                        harden,
                        {
                          // allowedImagePrefixes: ["file://"],
                          // allowDataImages: false,
                          // allow any image
                          allowAnyImage: true,
                          // defaultOrigin: "file://",
                        },
                      ],
                    ]}>
                    {artifactPreviewMarkdown(selected)}
                  </MessageResponse>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </Artifact>
        ) : (
          <p className="text-muted-foreground text-sm">
            Upload a file, then open the preview from the file header when you
            need it.
          </p>
        )}
      </ScrollArea>
    </div>
  );
}
