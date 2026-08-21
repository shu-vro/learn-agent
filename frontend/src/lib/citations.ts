import type { Artifact } from "@/lib/api/chat";
import { sortedArtifactChunks } from "@/lib/artifact-chunks";

/**
 * Document citations render as `[Source n](reference_id=<doc_id>:<chunk_id>)`.
 * Streamdown's rehype-harden blocks hrefs it cannot parse as URLs, so the bare
 * form renders as dead "[blocked]" text; fragment URLs pass through untouched.
 */
const CITATION_HREF = /\]\(\s*#?reference_id=/g;

/** Rewrite document citation links into the fragment form harden allows. */
export function linkifyCitations(markdown: string): string {
  return markdown.replace(CITATION_HREF, "](#reference_id=");
}

/** Split `reference_id=<sha256>:<chunk_id>` into its parts. */
export function parseCitation(
  url: string,
): { sha256: string; chunkId: number } | null {
  const match = /^#?reference_id=([^\s:]+):(\d+)$/.exec(url);
  return match
    ? { sha256: match[1], chunkId: Number.parseInt(match[2], 10) }
    : null;
}

/**
 * Locate the previewable chunk a citation points at.
 *
 * `reference_id` carries the document's sha256 (not its id) and a 1-based chunk
 * index, while the preview panel addresses documents by id and chunks by uuid —
 * so resolve against the loaded artifacts rather than trusting stored metadata.
 */
export function resolveCitation(
  url: string,
  artifacts: Artifact[],
): { documentId: string; chunkId: string; documentName: string } | null {
  const parsed = parseCitation(url);
  if (!parsed) {
    return null;
  }
  const artifact = artifacts.find((item) => item.sha256 === parsed.sha256);
  if (!artifact) {
    return null;
  }
  const chunk = sortedArtifactChunks(artifact.chunks).find(
    (item) => item.order === parsed.chunkId - 1,
  );
  return chunk
    ? {
        documentId: artifact.id,
        chunkId: chunk.id,
        documentName: artifact.name,
      }
    : null;
}

/** The artifact url a clicked anchor points at, or null if it is not a citation. */
export function citationTarget(href: string | null | undefined): string | null {
  const url = (href ?? "").replace(/^#/, "");
  return url.startsWith("reference_id=") ? url : null;
}
