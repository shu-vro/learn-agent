"use client";

import {
  Progress,
  ProgressLabel,
  ProgressValue,
} from "@/components/ui/progress";
import type { Artifact } from "@/lib/api/chat";

function progressLabel(artifact: Artifact): string {
  if (artifact.ingestion_status === "uploading") {
    return "Uploading document…";
  }
  if (artifact.ingestion_status === "processing") {
    return artifact.ingestion_stage_label ?? "Processing document…";
  }
  return "";
}

function progressValue(artifact: Artifact): number | null {
  if (artifact.ingestion_status === "uploading") {
    return artifact.upload_progress ?? 0;
  }
  if (artifact.ingestion_status === "processing") {
    return artifact.ingestion_progress ?? null;
  }
  return null;
}

export function ArtifactProgress({ artifact }: { artifact: Artifact }) {
  const label = progressLabel(artifact);
  const value = progressValue(artifact);
  const isIndeterminate =
    artifact.ingestion_status === "processing" && value === null;

  if (!label) {
    return null;
  }

  return (
    <div className="space-y-1.5">
      <Progress value={isIndeterminate ? null : value}>
        <ProgressLabel className="text-muted-foreground text-xs">
          {label}
        </ProgressLabel>
        {!isIndeterminate && value !== null ? (
          <ProgressValue>{value}%</ProgressValue>
        ) : null}
      </Progress>
    </div>
  );
}

export function isArtifactInProgress(artifact: Artifact): boolean {
  return (
    artifact.ingestion_status === "uploading" ||
    artifact.ingestion_status === "processing"
  );
}
