"use client";

import {
  ChevronDownIcon,
  MoreHorizontal,
  SettingsIcon,
  UploadIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  Artifact,
  ArtifactHeader,
  ArtifactTitle,
} from "@/components/ai-elements/artifact";
import {
  defaultIngestionPreferences,
  useAuth,
} from "@/components/auth/auth-provider";
import {
  ArtifactProgress,
  isArtifactInProgress,
} from "@/components/chat/artifact-progress";
import { useChatWorkspace } from "@/components/chat/chat-context";
import { ChunkPreviewList } from "@/components/chat/chunk-preview-list";
import {
  IngestionSettingsFields,
  type IngestionSettingsValue,
} from "@/components/settings/ingestion-settings-fields";
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
import { ScrollArea } from "@/components/ui/scroll-area";
import { Spinner } from "@/components/ui/spinner";
import type { IngestionUploadOptions } from "@/lib/api/preferences";
import { cn } from "@/lib/utils";

// Extensions Docling parses out of the box; the backend rejects anything else
// (see SUPPORTED_UPLOAD_SUFFIXES in src/config/constants.py).
const SUPPORTED_UPLOAD_ACCEPT = [
  ".pdf",
  ".docx,.dotx,.docm,.dotm",
  ".pptx,.potx,.ppsx,.pptm,.potm,.ppsm",
  ".xlsx,.xlsm",
  ".html,.htm,.xhtml",
  ".md,.txt,.text,.qmd,.rmd",
  ".adoc,.asciidoc,.asc",
  ".tex,.latex",
  ".csv,.json,.xml,.nxml,.xbrl,.dclg",
  ".jpg,.jpeg,.png,.tif,.tiff,.bmp,.webp",
  ".wav,.mp3,.m4a,.aac,.ogg,.flac,.mp4,.avi,.mov",
  ".vtt,.eml,.epub,.gz",
].join(",");

function isDesktopViewport() {
  if (typeof window === "undefined") {
    return false;
  }
  return window.matchMedia("(min-width: 768px)").matches;
}

function toUploadOptions(
  value: IngestionSettingsValue,
): IngestionUploadOptions {
  const { rebuild, ...ingestion } = value;
  return { ...ingestion, rebuild };
}

export function ArtifactsPanel({
  className,
  projectId = null,
}: {
  className?: string;
  projectId?: string | null;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { preferences } = useAuth();
  const {
    artifacts,
    selectedArtifactId,
    setSelectedArtifactId,
    focusedChunkId,
    chunkFocusSeq,
    addArtifactsFromFiles,
    deleteArtifact,
  } = useChatWorkspace();

  const [ingestionSettings, setIngestionSettings] =
    useState<IngestionSettingsValue>(() => ({
      ...defaultIngestionPreferences(),
      rebuild: false,
    }));

  useEffect(() => {
    if (preferences?.ingestion) {
      setIngestionSettings((prev) => ({
        ...preferences.ingestion,
        rebuild: prev.rebuild,
      }));
    }
  }, [preferences]);

  const selected = artifacts.find((a) => a.id === selectedArtifactId) ?? null;
  const uploadingArtifact =
    artifacts.find((artifact) => artifact.ingestion_status === "uploading") ??
    null;
  const isUploading = uploadingArtifact !== null;
  const isIngesting = selected?.ingestion_status === "processing";
  const [previewOpen, setPreviewOpen] = useState(false);
  const showIngestionSettings = Boolean(projectId);

  // biome-ignore lint/correctness/useExhaustiveDependencies: chunkFocusSeq re-opens the preview when the same chunk is clicked again
  useEffect(() => {
    if (isIngesting || isUploading || focusedChunkId) {
      setPreviewOpen(true);
    }
  }, [isIngesting, isUploading, focusedChunkId, chunkFocusSeq]);

  const processFiles = useCallback(
    async (files: FileList | File[] | null | undefined) => {
      if (!files?.length) {
        return;
      }
      const list = files instanceof FileList ? [...files] : files;
      const options = showIngestionSettings
        ? toUploadOptions(ingestionSettings)
        : undefined;
      try {
        await addArtifactsFromFiles(list, options);
      } finally {
        if (inputRef.current) {
          inputRef.current.value = "";
        }
      }
    },
    [addArtifactsFromFiles, ingestionSettings, showIngestionSettings],
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
      className={cn("flex h-full min-h-0 flex-col bg-sidebar/80", className)}
    >
      <Dialog onOpenChange={setUploadOpen} open={uploadOpen}>
        <DialogContent className="gap-4 sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Add files</DialogTitle>
            <DialogDescription>
              Drag documents here, or choose them from your device. PDF, Office
              files, images, HTML, Markdown and more are supported.
            </DialogDescription>
          </DialogHeader>
          <section
            aria-label="File drop zone"
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
            }}
          >
            {isUploading && uploadingArtifact ? (
              <div className="flex w-full max-w-xs flex-col gap-3">
                <p className="truncate text-muted-foreground text-sm">
                  {uploadingArtifact.name}
                </p>
                <ArtifactProgress artifact={uploadingArtifact} />
              </div>
            ) : (
              <>
                <UploadIcon className="size-8 text-muted-foreground" />
                <p className="text-muted-foreground text-sm">
                  Drop files to upload
                </p>
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => inputRef.current?.click()}
                >
                  Select files
                </Button>
              </>
            )}
          </section>
        </DialogContent>
      </Dialog>

      <Dialog onOpenChange={setSettingsOpen} open={settingsOpen}>
        <DialogContent className="gap-4 sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Upload settings</DialogTitle>
            <DialogDescription>
              Options used for document ingestion on this upload.
            </DialogDescription>
          </DialogHeader>
          <IngestionSettingsFields
            showRebuild
            value={ingestionSettings}
            onChange={setIngestionSettings}
          />
          <DialogFooter>
            <Button type="button" onClick={() => setSettingsOpen(false)}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className="flex shrink-0 items-center justify-between gap-2 border-border/40 border-b px-3 py-3">
        <span className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
          Files
        </span>
        <div className="flex items-center gap-0.5">
          {showIngestionSettings ? (
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              className="rounded-xl text-muted-foreground hover:text-foreground"
              onClick={() => setSettingsOpen(true)}
            >
              <SettingsIcon className="size-4" />
              <span className="sr-only">Upload settings</span>
            </Button>
          ) : null}
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept={SUPPORTED_UPLOAD_ACCEPT}
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
            onClick={onUploadButtonClick}
          >
            <UploadIcon className="size-4" />
            <span className="sr-only">Upload</span>
          </Button>
        </div>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex min-h-0 flex-col gap-2 p-2">
          {artifacts.length === 0 ? (
            <p className="px-2 py-3 text-muted-foreground text-sm">
              Upload a file to get started.
            </p>
          ) : (
            <ul className="space-y-1">
              {artifacts.map((artifact) => {
                const isSelected = artifact.id === selectedArtifactId;
                const inProgress = isArtifactInProgress(artifact);
                const isFailed = artifact.ingestion_status === "failed";
                return (
                  <li key={artifact.id} className="space-y-1">
                    <div
                      className={cn(
                        "flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-sm transition-colors",
                        isSelected
                          ? "bg-accent text-accent-foreground"
                          : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => setSelectedArtifactId(artifact.id)}
                        className="min-w-0 flex-1 text-left truncate"
                      >
                        {artifact.name}
                      </button>
                      {inProgress ? (
                        <Spinner className="size-3.5 shrink-0" />
                      ) : null}
                      {isFailed ? (
                        <span className="shrink-0 text-destructive text-xs">
                          Failed
                        </span>
                      ) : null}

                      <DropdownMenu>
                        <DropdownMenuTrigger>
                          <button
                            type="button"
                            className="rounded-xl p-1 text-muted-foreground hover:text-foreground"
                          >
                            <MoreHorizontal className="size-4" />
                            <span className="sr-only">More</span>
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent>
                          <DropdownMenuItem
                            onClick={async () => {
                              await deleteArtifact(artifact.id);
                            }}
                            data-variant="destructive"
                          >
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    {inProgress ? (
                      <div className="px-3 pb-1">
                        <ArtifactProgress artifact={artifact} />
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          )}

          {selected ? (
            <Artifact className="border-border/40 bg-background/60">
              <Collapsible
                className="flex min-h-0 flex-col"
                onOpenChange={setPreviewOpen}
                open={previewOpen}
              >
                <ArtifactHeader className="border-border/40 bg-transparent p-0">
                  <CollapsibleTrigger
                    aria-expanded={previewOpen}
                    aria-label={`${
                      previewOpen ? "Collapse" : "Expand"
                    } preview: ${selected.name}`}
                    className="flex w-full items-center gap-2 px-4 py-2 text-left transition-colors hover:bg-muted/40"
                  >
                    <ChevronDownIcon
                      aria-hidden
                      className={cn(
                        "size-4 shrink-0 text-muted-foreground transition-transform",
                        previewOpen && "rotate-180",
                      )}
                    />
                    <ArtifactTitle className="min-w-0 flex-1 truncate border-0 py-0">
                      Preview
                    </ArtifactTitle>
                    {isIngesting ? (
                      <Spinner className="size-4 shrink-0 text-muted-foreground" />
                    ) : null}
                  </CollapsibleTrigger>
                </ArtifactHeader>
                <CollapsibleContent className="min-h-0 overflow-hidden data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0">
                  <div className="max-h-[calc(100vh-16rem)] overflow-auto px-4 pb-4">
                    {isIngesting ? (
                      <div className="space-y-3 py-2">
                        <ArtifactProgress artifact={selected} />
                        <p className="text-muted-foreground text-sm">
                          You can close this panel or refresh the page —
                          processing continues in the background.
                        </p>
                      </div>
                    ) : selected.ingestion_status === "uploading" ? (
                      <div className="py-2">
                        <ArtifactProgress artifact={selected} />
                      </div>
                    ) : (
                      <ChunkPreviewList
                        chunks={selected.chunks}
                        projectId={projectId}
                        artifactId={selected.id}
                        focusedChunkId={focusedChunkId}
                        focusSeq={chunkFocusSeq}
                      />
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </Artifact>
          ) : null}
        </div>
      </ScrollArea>
    </div>
  );
}
