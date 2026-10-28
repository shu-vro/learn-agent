"use client";

import { CodeIcon, DownloadIcon, EyeIcon, MaximizeIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { CustomRendererProps } from "streamdown";

import {
  Artifact,
  ArtifactAction,
  ArtifactActions,
  ArtifactHeader,
  ArtifactTitle,
} from "@/components/ai-elements/artifact";
import { CodeBlock } from "@/components/ai-elements/code-block";

// Reports document height to the parent so the iframe grows with its content.
const RESIZE_SCRIPT = `<script>(()=>{const post=()=>parent.postMessage({__artifactHeight:document.documentElement.scrollHeight},"*");new ResizeObserver(post).observe(document.documentElement);addEventListener("load",post);})();</script>`;

// Artifacts always render on a light canvas: models reliably style for light
// backgrounds, and inheriting the app's dark scheme washes out their text.
const BASE_STYLE = `<style>:root{color-scheme:light}body{margin:0;padding:16px;background:#fff;color:#111;font-family:ui-sans-serif,system-ui,sans-serif;line-height:1.5}</style>`;

function buildDocument(code: string): string {
  const head = `<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">${BASE_STYLE}`;
  if (/<html[\s>]/i.test(code)) {
    return code.includes("<head>")
      ? code
          .replace("<head>", `<head>${head}`)
          .replace(/<\/body>/i, `${RESIZE_SCRIPT}</body>`)
      : `${head}${code}${RESIZE_SCRIPT}`;
  }
  return `<!doctype html><html><head>${head}</head><body>${code}${RESIZE_SCRIPT}</body></html>`;
}

function parseTitle(meta?: string): string {
  return meta?.match(/title=["']([^"']+)["']/)?.[1] ?? "Interactive artifact";
}

/**
 * Renders ```artifact fences as a live, sandboxed HTML/JS page. The iframe has
 * no `allow-same-origin`, so artifact code cannot touch the app's cookies,
 * storage or DOM.
 */
export function InteractiveArtifact({
  code,
  isIncomplete,
  meta,
}: CustomRendererProps) {
  const [view, setView] = useState<"preview" | "code">("preview");
  const [height, setHeight] = useState(360);
  const frameRef = useRef<HTMLIFrameElement>(null);
  const title = parseTitle(meta);
  const srcDoc = useMemo(() => buildDocument(code), [code]);

  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (event.source !== frameRef.current?.contentWindow) return;
      const next = Number(event.data?.__artifactHeight);
      if (next > 0) setHeight(Math.min(Math.max(next, 120), 900));
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  const download = () => {
    const url = URL.createObjectURL(new Blob([srcDoc], { type: "text/html" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/[^\w-]+/g, "_")}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // While streaming, show the code being written instead of reloading a
  // half-finished page on every token.
  const showCode = isIncomplete || view === "code";

  return (
    <Artifact className="my-4">
      <ArtifactHeader className="py-2">
        <ArtifactTitle>
          {title}
          {isIncomplete && (
            <span className="ml-2 text-muted-foreground text-xs">
              building…
            </span>
          )}
        </ArtifactTitle>
        <ArtifactActions>
          <ArtifactAction
            disabled={isIncomplete}
            icon={view === "code" ? EyeIcon : CodeIcon}
            onClick={() => setView(view === "code" ? "preview" : "code")}
            tooltip={view === "code" ? "Preview" : "Code"}
          />
          <ArtifactAction
            disabled={isIncomplete}
            icon={MaximizeIcon}
            onClick={() => frameRef.current?.requestFullscreen()}
            tooltip="Fullscreen"
          />
          <ArtifactAction
            disabled={isIncomplete}
            icon={DownloadIcon}
            onClick={download}
            tooltip="Download .html"
          />
        </ArtifactActions>
      </ArtifactHeader>
      {showCode && (
        <CodeBlock
          className="max-h-[480px] overflow-auto rounded-none border-0"
          code={code}
          language="html"
        />
      )}
      {!isIncomplete && (
        <iframe
          className={showCode ? "hidden" : "w-full border-0 bg-white"}
          ref={frameRef}
          sandbox="allow-scripts allow-popups allow-modals allow-forms"
          srcDoc={srcDoc}
          style={{ height }}
          title={title}
        />
      )}
    </Artifact>
  );
}
