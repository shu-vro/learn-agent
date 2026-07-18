import { nanoid } from "nanoid";
import {
  appendIngestionToFormData,
  type IngestionUploadOptions,
} from "@/lib/api/preferences";
import {
  type ArtifactSeed,
  type ChatMessageSeed,
  SEED_MESSAGES_BY_THREAD,
  SEED_THREADS,
  type ThreadSeed,
} from "@/lib/dummy/seed";
import { del, get, patch, post } from "@/utils/fetch";

export type Thread = ThreadSeed;
export type ChatMessage = ChatMessageSeed;
export type Artifact = ArtifactSeed;

function isArtifact(value: unknown): value is Artifact {
  return (
    value !== null &&
    typeof value === "object" &&
    "id" in value &&
    "name" in value &&
    "chunks" in value &&
    typeof (value as Artifact).chunks === "object" &&
    (value as Artifact).chunks !== null
  );
}

function isThread(value: unknown): value is Thread {
  return (
    value !== null &&
    typeof value === "object" &&
    "id" in value &&
    "thread_name" in value &&
    "extra" in value &&
    typeof (value as Thread).extra === "object" &&
    (value as Thread).extra !== null &&
    "created_at" in value &&
    typeof (value as Thread).created_at === "string" &&
    "updated_at" in value &&
    typeof (value as Thread).updated_at === "string"
  );
}
function isArtifactList(value: unknown): value is Artifact[] {
  return Array.isArray(value) && value.every(isArtifact);
}

function isThreadList(value: unknown): value is Thread[] {
  return Array.isArray(value) && value.every(isThread);
}

export function threadDisplayName(thread: Thread): string {
  const name = thread.thread_name.trim();
  return name || "New thread";
}

export async function listThreads(
  projectId: string | null | undefined,
): Promise<Thread[]> {
  if (!projectId) {
    return [...SEED_THREADS];
  }
  const res = await get({ endpoint: `/projects/${projectId}/threads` });
  if (isThreadList(res)) {
    return res;
  }
  return [...SEED_THREADS];
}

export async function createThread(projectId: string): Promise<Thread | null> {
  const res = await post({
    endpoint: `/projects/${projectId}/threads`,
    throwable: true,
  });
  if (isThread(res)) {
    return res;
  }
  return null;
}

export async function updateThread(
  projectId: string,
  threadId: string,
  threadName: string,
): Promise<Thread | null> {
  const res = await patch({
    endpoint: `/projects/${projectId}/threads/${threadId}`,
    params: { thread_name: threadName.trim() },
    throwable: true,
  });
  if (isThread(res)) {
    return res;
  }
  return null;
}

export async function deleteThread(
  projectId: string,
  threadId: string,
): Promise<boolean> {
  await del({
    endpoint: `/projects/${projectId}/threads/${threadId}`,
    throwable: true,
  });
  return true;
}

export async function listMessages(threadId: string): Promise<ChatMessage[]> {
  const res = await get({ endpoint: `/threads/${threadId}/messages` });
  if (Array.isArray(res) && res.length && "role" in (res[0] as object)) {
    return res as ChatMessage[];
  }
  return [
    ...(SEED_MESSAGES_BY_THREAD[threadId] ?? SEED_MESSAGES_BY_THREAD["t-1"]),
  ];
}

export async function listArtifacts(
  projectId: string | null | undefined,
): Promise<Artifact[]> {
  if (!projectId) {
    return [];
  }
  const res = await get({
    endpoint: `/projects/${projectId}/artifacts`,
  });
  if (isArtifactList(res)) {
    return res;
  }
  return [];
}

export async function getArtifactIngestionStatus(
  projectId: string,
  artifactId: string,
): Promise<Artifact | null> {
  const res = await get({
    endpoint: `/projects/${projectId}/artifacts/${artifactId}/ingestion-status`,
  });
  if (isArtifact(res)) {
    return res;
  }
  return null;
}

export type UploadArtifactOptions = {
  ingestion?: IngestionUploadOptions;
  onUploadProgress?: (percent: number) => void;
};

export async function uploadArtifacts(
  projectId: string,
  files: File[],
  options?: UploadArtifactOptions,
): Promise<Artifact[]> {
  if (!files.length) {
    return [];
  }
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  if (options?.ingestion) {
    appendIngestionToFormData(form, options.ingestion);
  }
  const res = await post({
    endpoint: `/projects/${projectId}/artifacts`,
    params: form,
    onUploadProgress: (event) => {
      if (!options?.onUploadProgress || !event.total) {
        return;
      }
      options.onUploadProgress(
        Math.min(100, Math.round((event.loaded / event.total) * 100)),
      );
    },
  });
  if (isArtifactList(res)) {
    return res;
  }
  return [];
}

export async function uploadArtifact(
  projectId: string,
  file: File,
  options?: UploadArtifactOptions,
): Promise<Artifact | null> {
  const created = await uploadArtifacts(projectId, [file], options);
  return created[0] ?? null;
}

export async function deleteArtifact(
  projectId: string | null | undefined,
  artifactId: string,
): Promise<boolean> {
  if (!projectId) return false;
  await del({
    endpoint: `/projects/${projectId}/artifacts/${artifactId}`,
    throwable: true,
  });
  return true;
}

export async function getArtifactNotes(
  projectId: string,
  artifactId: string,
): Promise<Record<string, string>> {
  const res = await get({
    endpoint: `/projects/${projectId}/artifacts/${artifactId}/notes`,
  });
  if (res && typeof res === "object" && "notes" in res) {
    const notes = (res as { notes?: unknown }).notes;
    if (notes && typeof notes === "object") {
      return notes as Record<string, string>;
    }
  }
  return {};
}

export async function generateChunkNote(
  projectId: string,
  artifactId: string,
  chunkId: string,
): Promise<string | null> {
  const res = await post({
    endpoint: `/projects/${projectId}/artifacts/${artifactId}/chunks/${chunkId}/note`,
    throwable: true,
  });
  if (res && typeof res === "object" && "content" in res) {
    return String((res as { content: unknown }).content ?? "");
  }
  return null;
}

export function createLocalArtifact(name: string, text: string): Artifact {
  return {
    id: nanoid(),
    name,
    chunks: { [nanoid()]: { content: text, order: 0 } },
    ingestion_status: "completed",
  };
}
