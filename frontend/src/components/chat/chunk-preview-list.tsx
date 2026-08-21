"use client";

import { StickyNoteIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { MessageResponse } from "@/components/ai-elements/message";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { generateChunkNote, getArtifactNotes } from "@/lib/api/chat";
import {
  type ArtifactChunks,
  sortedDisplayChunks,
} from "@/lib/artifact-chunks";
import { cn } from "@/lib/utils";

type ChunkPreviewListProps = {
  chunks: ArtifactChunks;
  projectId?: string | null;
  artifactId?: string | null;
  /** Chunk to scroll to and highlight (a citation was clicked). */
  focusedChunkId?: string | null;
  className?: string;
};

export function ChunkPreviewList({
  chunks,
  projectId,
  artifactId,
  focusedChunkId,
  className,
}: ChunkPreviewListProps) {
  const entries = sortedDisplayChunks(chunks);
  const canUseNotes = Boolean(projectId && artifactId);

  const focusedRef = useRef<HTMLDivElement | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [generating, setGenerating] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!projectId || !artifactId) {
      setNotes({});
      return;
    }
    let cancelled = false;
    (async () => {
      const existing = await getArtifactNotes(projectId, artifactId);
      if (!cancelled) {
        setNotes(existing);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, artifactId]);

  useEffect(() => {
    if (!focusedChunkId) {
      return;
    }
    // Chunks render after the artifact loads; wait a frame before scrolling.
    const frame = requestAnimationFrame(() => {
      focusedRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [focusedChunkId]);

  const handleGenerate = useCallback(
    async (chunkId: string, regenerate = false) => {
      if (!projectId || !artifactId) {
        return;
      }
      setErrors((prev) => {
        const next = { ...prev };
        delete next[chunkId];
        return next;
      });
      setGenerating((prev) => ({ ...prev, [chunkId]: true }));
      try {
        const content = await generateChunkNote(
          projectId,
          artifactId,
          chunkId,
          regenerate,
        );
        if (content != null) {
          setNotes((prev) => ({ ...prev, [chunkId]: content }));
        } else {
          setErrors((prev) => ({
            ...prev,
            [chunkId]: "Could not generate a note. Try again.",
          }));
        }
      } catch {
        setErrors((prev) => ({
          ...prev,
          [chunkId]: "Could not generate a note. Try again.",
        }));
      } finally {
        setGenerating((prev) => {
          const next = { ...prev };
          delete next[chunkId];
          return next;
        });
      }
    },
    [projectId, artifactId],
  );

  if (entries.length === 0) {
    return (
      <p className="text-muted-foreground text-sm">
        No text content available for this file.
      </p>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {entries.map((chunk) => {
        const note = notes[chunk.id];
        const isGenerating = generating[chunk.id];
        const error = errors[chunk.id];
        const hasNote = typeof note === "string" && note.length > 0;

        const isFocused = chunk.id === focusedChunkId;

        return (
          <div
            key={chunk.id}
            ref={isFocused ? focusedRef : undefined}
            className={cn(
              "overflow-hidden rounded-xl border border-border/40 bg-background/40 transition-colors",
              isFocused &&
                "border-primary/60 bg-primary/5 ring-1 ring-primary/40",
            )}
          >
            {chunk.type === "image" ? (
              <div className="border-border/40 border-b px-4 py-2">
                <span className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
                  Image
                </span>
              </div>
            ) : null}
            <MessageResponse className="px-4 pt-4 text-sm">
              {chunk.content}
            </MessageResponse>

            {canUseNotes ? (
              <div className="border-border/40 border-t">
                {hasNote ? (
                  <Accordion className="rounded-none border-0">
                    <AccordionItem value="note" className="border-0">
                      <AccordionTrigger className="px-4 py-2 text-sm hover:no-underline">
                        <span className="flex items-center gap-2">
                          <StickyNoteIcon className="size-4" />
                          Notes
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <MessageResponse className="text-sm">
                          {note}
                        </MessageResponse>
                        <div className="mt-3">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            disabled={isGenerating}
                            onClick={() => void handleGenerate(chunk.id, true)}
                          >
                            {isGenerating ? (
                              <>
                                <Spinner className="size-3.5" />
                                Regenerating…
                              </>
                            ) : (
                              "Regenerate note"
                            )}
                          </Button>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                ) : (
                  <div className="flex items-center justify-between gap-3 px-4 py-2">
                    {error ? (
                      <span className="text-destructive text-xs">{error}</span>
                    ) : (
                      <span className="text-muted-foreground text-xs">
                        Generate a study note for this chunk.
                      </span>
                    )}
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      disabled={isGenerating}
                      onClick={() => void handleGenerate(chunk.id)}
                    >
                      {isGenerating ? (
                        <>
                          <Spinner className="size-3.5" />
                          Generating…
                        </>
                      ) : (
                        <>
                          <StickyNoteIcon className="size-3.5" />
                          Generate note
                        </>
                      )}
                    </Button>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
