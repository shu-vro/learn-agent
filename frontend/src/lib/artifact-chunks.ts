export type ArtifactChunkEntry = {
  content: string;
  order: number;
};

/** API chunks map; values may be legacy plain strings in seed/demo data. */
export type ArtifactChunks = Record<string, ArtifactChunkEntry | string>;

function parseChunkEntry(
  value: string | ArtifactChunkEntry,
  index: number,
): ArtifactChunkEntry {
  if (typeof value === "string") {
    return { content: value, order: index };
  }
  return {
    content: value.content,
    order: value.order,
  };
}

export function sortedArtifactChunks(
  chunks: ArtifactChunks,
): ArtifactChunkEntry[] {
  return Object.values(chunks)
    .map((value, index) => parseChunkEntry(value, index))
    .sort((left, right) => left.order - right.order);
}

export function artifactChunksToMarkdown(chunks: ArtifactChunks): string {
  const parts = sortedArtifactChunks(chunks).map((chunk) =>
    chunk.content.trim(),
  );
  return parts.filter(Boolean).join("\n\n");
}
