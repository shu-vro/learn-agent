import { nanoid } from "nanoid";
import {
  type ArtifactSeed,
  type ChatMessageSeed,
  SEED_MESSAGES_BY_THREAD,
  SEED_THREADS,
  type ThreadSeed,
} from "@/lib/dummy/seed";
import { get, post } from "@/utils/fetch";

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

function isArtifactList(value: unknown): value is Artifact[] {
  return Array.isArray(value) && value.every(isArtifact);
}

export async function listThreads(
  _projectId?: string | null,
): Promise<Thread[]> {
  const res = await get({ endpoint: "/threads" });
  if (
    Array.isArray(res) &&
    res.length &&
    typeof res[0] === "object" &&
    "title" in res[0]
  ) {
    return res as Thread[];
  }
  return [...SEED_THREADS];
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

export async function uploadArtifact(
  projectId: string,
  file: File,
): Promise<Artifact | null> {
  const form = new FormData();
  form.append("file", file);
  const res = await post({
    endpoint: `/projects/${projectId}/artifacts`,
    params: form,
  });
  if (isArtifact(res)) {
    return res;
  }
  return null;
}

export function createLocalArtifact(name: string, text: string): Artifact {
  return {
    id: nanoid(),
    name,
    chunks: { [nanoid()]: text },
    ingestion_status: "completed",
  };
}
