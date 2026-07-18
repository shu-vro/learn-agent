export type ArtifactChunkEntry = {
  content: string;
  order: number;
  type?: "text_chunk" | "image" | "note";
};

/** API chunks map; values may be legacy plain strings in seed/demo data. */
export type ArtifactChunks = Record<string, ArtifactChunkEntry | string>;

function parseChunkEntry(
  value: string | ArtifactChunkEntry,
  index: number,
): ArtifactChunkEntry {
  if (typeof value === "string") {
    return { content: value, order: index, type: "text_chunk" };
  }
  return {
    content: value.content,
    order: value.order,
    type: value.type ?? "text_chunk",
  };
}

export type SortedArtifactChunk = ArtifactChunkEntry & { id: string };

export function sortedArtifactChunks(
  chunks: ArtifactChunks,
): SortedArtifactChunk[] {
  return Object.entries(chunks)
    .map(([id, value], index) => ({
      id,
      ...parseChunkEntry(value, index),
    }))
    .sort((left, right) => left.order - right.order);
}

export function sortedDisplayChunks(
  chunks: ArtifactChunks,
): SortedArtifactChunk[] {
  return sortedArtifactChunks(chunks).filter(
    (chunk) =>
      chunk.type === "text_chunk" || chunk.type === "image" || !chunk.type,
  );
}

export function artifactChunksToMarkdown(chunks: ArtifactChunks): string {
  const parts = sortedDisplayChunks(chunks).map((chunk) =>
    chunk.content.trim(),
  );
  return parts.filter(Boolean).join("\n\n");
}
