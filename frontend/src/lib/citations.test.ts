import { expect, test } from "bun:test";

import {
  citationTarget,
  linkifyCitations,
  resolveCitation,
} from "@/lib/citations";

test("citation links are rewritten to the fragment form harden allows", () => {
  expect(
    linkifyCitations(
      "saturates [Source 1](reference_id=abc123:8) and [Source 2](reference_id=abc123:9)",
    ),
  ).toBe(
    "saturates [Source 1](#reference_id=abc123:8) and [Source 2](#reference_id=abc123:9)",
  );
  // Already-fragment links and real URLs are left alone.
  expect(linkifyCitations("[Source 1](#reference_id=abc:1)")).toBe(
    "[Source 1](#reference_id=abc:1)",
  );
  expect(linkifyCitations("[Source 2](https://arxiv.org/abs/1706.03762)")).toBe(
    "[Source 2](https://arxiv.org/abs/1706.03762)",
  );
});

test("citationTarget maps an anchor href back to the artifact url", () => {
  expect(citationTarget("#reference_id=abc123:8")).toBe(
    "reference_id=abc123:8",
  );
  expect(citationTarget("reference_id=abc123:8")).toBe("reference_id=abc123:8");
  expect(citationTarget("https://example.com")).toBeNull();
  expect(citationTarget(null)).toBeNull();
});

test("resolveCitation maps sha256 + 1-based chunk index to document id + chunk uuid", () => {
  const artifacts = [
    {
      id: "doc-uuid-1",
      name: "1706.03762v7.pdf",
      sha256: "bdfaa68d",
      chunks: {
        "chunk-uuid-a": { content: "first", order: 0 },
        "chunk-uuid-b": { content: "second", order: 1 },
      },
    },
  ];

  expect(resolveCitation("reference_id=bdfaa68d:2", artifacts)).toEqual({
    documentId: "doc-uuid-1",
    chunkId: "chunk-uuid-b",
    documentName: "1706.03762v7.pdf",
  });
  // The fragment form the renderer produces resolves identically.
  expect(resolveCitation("#reference_id=bdfaa68d:1", artifacts)?.chunkId).toBe(
    "chunk-uuid-a",
  );
  // Unknown document, out-of-range chunk, and non-citations resolve to null.
  expect(resolveCitation("reference_id=deadbeef:1", artifacts)).toBeNull();
  expect(resolveCitation("reference_id=bdfaa68d:99", artifacts)).toBeNull();
  expect(resolveCitation("https://example.com", artifacts)).toBeNull();
});
