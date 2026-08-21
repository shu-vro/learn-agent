"use client";

import {
  ChevronDownIcon,
  FileTextIcon,
  ImageIcon,
  PlayIcon,
} from "lucide-react";
import { useMemo, useState } from "react";

import { useChatWorkspace } from "@/components/chat/chat-context";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { ChatArtifact } from "@/lib/api/chat";
import { cn } from "@/lib/utils";

const TYPE_LABEL: Record<ChatArtifact["type"], string> = {
  document: "doc",
  website: "web",
  video: "video",
  image: "image",
};

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function faviconOf(url: string): string {
  return `https://www.google.com/s2/favicons?domain=${hostOf(url)}&sz=64`;
}

/** YouTube thumbnail, or null for non-YouTube video links. */
function youtubeThumbOf(url: string): string | null {
  try {
    const parsed = new URL(url);
    const id = parsed.hostname.endsWith("youtu.be")
      ? parsed.pathname.slice(1)
      : parsed.searchParams.get("v");
    return id ? `https://i.ytimg.com/vi/${id}/mqdefault.jpg` : null;
  } catch {
    return null;
  }
}

/** Cited artifacts first, ordered by where the answer cites them. */
function orderArtifacts(
  artifacts: ChatArtifact[],
  content: string,
): { cited: ChatArtifact[]; rest: ChatArtifact[] } {
  const cited = artifacts
    .filter((artifact) => artifact.cited)
    .map((artifact) => ({ artifact, at: content.indexOf(artifact.url) }))
    .sort((left, right) => left.at - right.at)
    .map((entry) => entry.artifact);
  return {
    cited,
    rest: artifacts.filter((artifact) => !artifact.cited),
  };
}

function summarize(artifacts: ChatArtifact[]): string {
  const counts = new Map<ChatArtifact["type"], number>();
  for (const artifact of artifacts) {
    counts.set(artifact.type, (counts.get(artifact.type) ?? 0) + 1);
  }
  return [...counts]
    .map(
      ([type, count]) => `${count} ${TYPE_LABEL[type]}${count > 1 ? "s" : ""}`,
    )
    .join(" · ");
}

function SourceCard({
  artifact,
  index,
}: {
  artifact: ChatArtifact;
  index: number | null;
}) {
  const { focusChunk } = useChatWorkspace();
  const badge =
    index === null ? null : (
      <span className="shrink-0 rounded-full bg-primary/10 px-1.5 font-medium text-[10px] text-primary leading-4">
        {index}
      </span>
    );

  const className = cn(
    "flex h-[104px] w-52 shrink-0 flex-col gap-1.5 rounded-xl border border-border/50",
    "bg-background/60 p-3 text-left transition-colors hover:border-border hover:bg-accent/40",
  );

  if (artifact.type === "document") {
    const canOpen = Boolean(artifact.documentId && artifact.chunkUuid);
    return (
      <button
        type="button"
        className={cn(className, !canOpen && "cursor-default opacity-70")}
        disabled={!canOpen}
        onClick={() => {
          if (artifact.documentId && artifact.chunkUuid) {
            focusChunk(artifact.documentId, artifact.chunkUuid);
          }
        }}
      >
        <div className="flex items-center gap-1.5">
          {badge}
          <FileTextIcon className="size-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate text-muted-foreground text-xs">
            {TYPE_LABEL.document}
          </span>
        </div>
        <span className="line-clamp-2 font-medium text-sm leading-snug">
          {artifact.documentName ?? "Document"}
        </span>
        <span className="mt-auto text-muted-foreground text-xs">
          {[
            artifact.page ? `p. ${artifact.page}` : null,
            artifact.chunkUuid
              ? `chunk ${artifact.url.split(":").pop()}`
              : null,
            artifact.score ? `score ${artifact.score}` : null,
          ]
            .filter(Boolean)
            .join(" · ")}
        </span>
      </button>
    );
  }

  const thumb = artifact.type === "video" ? youtubeThumbOf(artifact.url) : null;
  const preview =
    artifact.type === "image"
      ? artifact.url
      : (thumb ?? faviconOf(artifact.url));

  return (
    <a
      className={className}
      href={artifact.url}
      rel="noreferrer"
      target="_blank"
    >
      <div className="flex items-center gap-1.5">
        {badge}
        {artifact.type === "video" ? (
          <PlayIcon className="size-3.5 shrink-0 text-muted-foreground" />
        ) : artifact.type === "image" ? (
          <ImageIcon className="size-3.5 shrink-0 text-muted-foreground" />
        ) : null}
        {/* biome-ignore lint/performance/noImgElement: third-party favicons/thumbnails are not in next.config images */}
        <img
          alt=""
          aria-hidden
          className={cn(
            "shrink-0 rounded",
            artifact.type === "website" ? "size-4" : "h-4 w-7 object-cover",
          )}
          src={preview}
        />
        <span className="truncate text-muted-foreground text-xs">
          {hostOf(artifact.url)}
        </span>
      </div>
      <span className="line-clamp-3 font-medium text-sm leading-snug">
        {artifact.title || hostOf(artifact.url)}
      </span>
    </a>
  );
}

export function MessageSources({
  artifacts,
  content,
  className,
}: {
  artifacts: ChatArtifact[];
  content: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const { cited, rest } = useMemo(
    () => orderArtifacts(artifacts, content),
    [artifacts, content],
  );

  if (artifacts.length === 0) {
    return null;
  }

  const shown = cited.length > 0 ? cited : rest;
  const hidden = cited.length > 0 ? rest : [];
  const stack = shown.slice(0, 3);

  return (
    <Collapsible className={className} onOpenChange={setOpen} open={open}>
      <CollapsibleTrigger className="flex items-center gap-2 rounded-full px-2 py-1 text-muted-foreground text-xs transition-colors hover:bg-accent/50 hover:text-foreground">
        <span className="-space-x-1.5 flex items-center">
          {stack.map((artifact) => (
            <span
              key={artifact.id}
              className="flex size-4 items-center justify-center overflow-hidden rounded-full border border-background bg-muted"
            >
              {artifact.type === "document" ? (
                <FileTextIcon className="size-2.5 text-muted-foreground" />
              ) : (
                // biome-ignore lint/performance/noImgElement: third-party favicons are not in next.config images
                <img alt="" aria-hidden src={faviconOf(artifact.url)} />
              )}
            </span>
          ))}
        </span>
        <span>{summarize(shown)}</span>
        <ChevronDownIcon
          className={cn("size-3.5 transition-transform", open && "rotate-180")}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:animate-out data-[state=open]:animate-in">
        <div className="flex gap-2 overflow-x-auto py-2">
          {shown.map((artifact, index) => (
            <SourceCard
              artifact={artifact}
              index={cited.length > 0 ? index + 1 : null}
              key={artifact.id}
            />
          ))}
          {showAll
            ? hidden.map((artifact) => (
                <SourceCard
                  artifact={artifact}
                  index={null}
                  key={artifact.id}
                />
              ))
            : null}
        </div>
        {hidden.length > 0 && !showAll ? (
          <button
            className="px-2 pb-1 text-muted-foreground text-xs hover:text-foreground"
            onClick={() => setShowAll(true)}
            type="button"
          >
            {hidden.length} more found, not cited
          </button>
        ) : null}
      </CollapsibleContent>
    </Collapsible>
  );
}
