import { nanoid } from "nanoid";
import {
  type ArtifactSeed,
  type ChatMessageSeed,
  SEED_ARTIFACTS,
  SEED_MESSAGES_BY_THREAD,
  SEED_THREADS,
  type ThreadSeed,
} from "@/lib/dummy/seed";
import { get } from "@/utils/fetch";

export type Thread = ThreadSeed;
export type ChatMessage = ChatMessageSeed;
export type Artifact = ArtifactSeed;

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
  _projectId?: string | null,
): Promise<Artifact[]> {
  const res = await get({ endpoint: "/artifacts" });
  if (Array.isArray(res) && res.length && "content" in (res[0] as object)) {
    return res as Artifact[];
  }
  return [...SEED_ARTIFACTS];
}

export function createLocalArtifact(name: string, content: string): Artifact {
  return { id: nanoid(), name, content };
}
