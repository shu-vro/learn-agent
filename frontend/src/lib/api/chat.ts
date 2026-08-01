import { EventSourcePolyfill } from "event-source-polyfill";
import { nanoid } from "nanoid";
import {
  appendIngestionToFormData,
  type IngestionUploadOptions,
} from "@/lib/api/preferences";
import {
  type ArtifactSeed,
  SEED_MESSAGES_BY_THREAD,
  SEED_THREADS,
  type ThreadSeed,
} from "@/lib/dummy/seed";
import { del, get, patch, post } from "@/utils/fetch";

// Ensure EventSource exists in older browsers (GET-only; POST chat uses fetch below).
if (typeof window !== "undefined" && !("EventSource" in window)) {
  (
    window as unknown as { EventSource: typeof EventSourcePolyfill }
  ).EventSource = EventSourcePolyfill;
}

export type Thread = ThreadSeed;
export type Artifact = ArtifactSeed;

export type ChatToolCall = {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: unknown;
  state: "input-available" | "output-available" | "output-error";
  step?: number;
};

export type ChatThinkingStep = {
  id: string;
  step: number;
  text: string;
};

/** Ordered agent loop: think → tools → think → … → answer */
export type ChatTimelineItem =
  | ({ kind: "thinking" } & ChatThinkingStep)
  | ({ kind: "tool" } & ChatToolCall);

export type ChatUsageIteration = {
  iteration: number;
  input_token: number;
  cache_token: number;
  output_token: number;
  total_token: number;
};

export type ChatUsage = {
  input_token: number;
  cache_token: number;
  output_token: number;
  total_token: number;
  iterations: number;
  iteration_details: ChatUsageIteration[];
};

export type ChatBranch = {
  id: string;
  content: string;
  timeline?: ChatTimelineItem[];
  /** @deprecated prefer timeline */
  thinking?: string;
  /** @deprecated prefer timeline */
  tools?: ChatToolCall[];
  streaming?: boolean;
  /** Streaming-only: step index currently receiving thinking deltas. */
  activeThinkingStep?: number | null;
  usage?: ChatUsage | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  chatId?: string;
  groupId?: string;
  selection?: string | null;
  referenceId?: string | null;
  imageUrls?: string[] | null;
  branches?: ChatBranch[];
  activeBranch?: number;
  timeline?: ChatTimelineItem[];
  thinking?: string;
  tools?: ChatToolCall[];
  streaming?: boolean;
  /** Streaming-only: mirrors active branch's activeThinkingStep. */
  activeThinkingStep?: number | null;
  usage?: ChatUsage | null;
};

export type ChatSendOptions = {
  query?: string;
  threadId?: string | null;
  messageId?: string | null;
  referenceId?: string | null;
  selection?: string | null;
  images?: string[] | null;
};

export type ChatStreamHandlers = {
  onEvent: (event: string, data: Record<string, unknown>) => void;
  onError?: (error: Error) => void;
  onDone?: () => void;
};

type ApiThinking = { id: string; thinking: string; created_at?: string };
type ApiTool = {
  id: string;
  tool_name: string;
  tool_parameters: Record<string, unknown>;
  tool_result: unknown;
  created_at?: string;
};
type ApiUsageIteration = {
  iteration: number;
  input_token?: number;
  cache_token?: number;
  output_token?: number;
  total_token?: number;
};

type ApiUsage = {
  input_token?: number;
  cache_token?: number;
  output_token?: number;
  total_token?: number;
  iterations?: number;
  iteration_details?: ApiUsageIteration[];
};

type ApiChatMessage = {
  id: string;
  chat_id: string;
  message: string;
  selection?: string | null;
  reference_id?: string | null;
  image_urls?: string[] | null;
  input_token?: number;
  cache_token?: number;
  output_token?: number;
  total_token?: number;
  usage?: ApiUsage | null;
  thinking_messages?: ApiThinking[];
  tool_messages?: ApiTool[];
};
type ApiChat = {
  id: string;
  thread_id: string;
  type: string;
  group_id?: string | null;
  messages: ApiChatMessage[];
};
type ApiTurn = {
  group_id: string;
  user: ApiChat | null;
  assistant: ApiChat | null;
};

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

function mapTool(tool: ApiTool): ChatToolCall {
  return {
    id: tool.id,
    name: tool.tool_name,
    args:
      tool.tool_parameters && typeof tool.tool_parameters === "object"
        ? tool.tool_parameters
        : {},
    result: tool.tool_result,
    state: "output-available",
  };
}

function mapTimeline(msg: ApiChatMessage): ChatTimelineItem[] {
  const thinking = (msg.thinking_messages ?? []).map((t, index) => ({
    kind: "thinking" as const,
    id: t.id,
    step: index,
    text: t.thinking,
    at: t.created_at ?? "",
  }));
  const tools = (msg.tool_messages ?? []).map((t) => ({
    kind: "tool" as const,
    ...mapTool(t),
    at: t.created_at ?? "",
  }));
  return [...thinking, ...tools]
    .sort((a, b) => a.at.localeCompare(b.at))
    .map(({ at: _at, ...item }) => item);
}

function mapUsage(msg: ApiChatMessage): ChatUsage | null {
  const raw = msg.usage;
  const input = raw?.input_token ?? msg.input_token ?? 0;
  const cache = raw?.cache_token ?? msg.cache_token ?? 0;
  const output = raw?.output_token ?? msg.output_token ?? 0;
  const total = raw?.total_token ?? msg.total_token ?? input + output;
  const details = (raw?.iteration_details ?? []).map((item, index) => ({
    iteration: item.iteration ?? index + 1,
    input_token: item.input_token ?? 0,
    cache_token: item.cache_token ?? 0,
    output_token: item.output_token ?? 0,
    total_token: item.total_token ?? 0,
  }));
  const iterations = raw?.iterations ?? details.length;
  if (!input && !cache && !output && !iterations) {
    return null;
  }
  return {
    input_token: input,
    cache_token: cache,
    output_token: output,
    total_token: total,
    iterations,
    iteration_details: details,
  };
}

export function parseChatUsage(data: unknown): ChatUsage | null {
  if (!data || typeof data !== "object") {
    return null;
  }
  const raw = data as ApiUsage;
  const details = (raw.iteration_details ?? []).map((item, index) => ({
    iteration: item.iteration ?? index + 1,
    input_token: item.input_token ?? 0,
    cache_token: item.cache_token ?? 0,
    output_token: item.output_token ?? 0,
    total_token: item.total_token ?? 0,
  }));
  const input = raw.input_token ?? 0;
  const cache = raw.cache_token ?? 0;
  const output = raw.output_token ?? 0;
  const iterations = raw.iterations ?? details.length;
  if (!input && !cache && !output && !iterations) {
    return null;
  }
  return {
    input_token: input,
    cache_token: cache,
    output_token: output,
    total_token: raw.total_token ?? input + output,
    iterations,
    iteration_details: details,
  };
}

function mapBranch(msg: ApiChatMessage): ChatBranch {
  const timeline = mapTimeline(msg);
  const thinkingParts = timeline
    .filter(
      (i): i is Extract<ChatTimelineItem, { kind: "thinking" }> =>
        i.kind === "thinking",
    )
    .map((i) => i.text);
  const tools = timeline.filter(
    (i): i is Extract<ChatTimelineItem, { kind: "tool" }> => i.kind === "tool",
  );
  return {
    id: msg.id,
    content: msg.message,
    timeline,
    thinking: thinkingParts.join("\n\n"),
    tools,
    usage: mapUsage(msg),
  };
}

export function turnsToMessages(turns: ApiTurn[]): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (const turn of turns) {
    if (turn.user?.messages?.[0]) {
      const um = turn.user.messages[0];
      messages.push({
        id: um.id,
        role: "user",
        content: um.message,
        chatId: turn.user.id,
        groupId: turn.group_id,
        selection: um.selection,
        referenceId: um.reference_id,
        imageUrls: um.image_urls ?? null,
      });
    }
    if (turn.assistant) {
      const branches = (turn.assistant.messages ?? []).map(mapBranch);
      const active = Math.max(0, branches.length - 1);
      const current = branches[active];
      messages.push({
        id: turn.assistant.id,
        role: "assistant",
        content: current?.content ?? "",
        chatId: turn.assistant.id,
        groupId: turn.group_id,
        branches,
        activeBranch: active,
        thinking: current?.thinking,
        tools: current?.tools,
        usage: current?.usage,
      });
    }
  }
  return messages;
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

export async function listMessages(
  projectId: string | null | undefined,
  threadId: string,
): Promise<ChatMessage[]> {
  if (!projectId) {
    return (
      SEED_MESSAGES_BY_THREAD[threadId] ??
      SEED_MESSAGES_BY_THREAD["t-1"] ??
      []
    ).map((m) => ({ ...m }));
  }
  const res = await get({
    endpoint: `/projects/${projectId}/threads/${threadId}/chats`,
  });
  if (Array.isArray(res)) {
    return turnsToMessages(res as ApiTurn[]);
  }
  return [];
}

export function streamChat(
  projectId: string,
  options: ChatSendOptions,
  handlers: ChatStreamHandlers,
): () => void {
  const baseUrl = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/+$/, "");
  const url = `${baseUrl}/api/v1/projects/${projectId}/chats`;

  const body: Record<string, unknown> = {};
  if (options.query) body.query = options.query;
  if (options.threadId) body.thread_id = options.threadId;
  if (options.messageId) body.message_id = options.messageId;
  if (options.referenceId) body.reference_id = options.referenceId;
  if (options.selection) body.selection = options.selection;
  if (options.images?.length) body.images = options.images;

  const controller = new AbortController();
  let closed = false;

  const close = () => {
    if (closed) return;
    closed = true;
    controller.abort();
  };

  // event-source-polyfill is GET-only; POST SSE uses fetch + stream parsing.
  (async () => {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        credentials: "include",
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Chat stream failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (!closed) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sep = buffer.indexOf("\n\n");
        while (sep >= 0) {
          const rawEvent = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          sep = buffer.indexOf("\n\n");

          let eventName = "message";
          const dataLines: string[] = [];
          for (const line of rawEvent.split("\n")) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trim());
            }
          }
          if (!dataLines.length) continue;

          try {
            const parsed = JSON.parse(dataLines.join("\n")) as Record<
              string,
              unknown
            >;
            handlers.onEvent(eventName, parsed);
            if (eventName === "done" || eventName === "error") {
              close();
              handlers.onDone?.();
              return;
            }
          } catch (err) {
            handlers.onError?.(
              err instanceof Error
                ? err
                : new Error("Failed to parse SSE event"),
            );
          }
        }
      }

      if (!closed) {
        handlers.onDone?.();
      }
    } catch (err) {
      if (controller.signal.aborted) {
        return;
      }
      handlers.onError?.(
        err instanceof Error ? err : new Error("Chat stream connection error"),
      );
      handlers.onDone?.();
    }
  })();

  return close;
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
  regenerate = false,
): Promise<string | null> {
  const query = regenerate ? "?regenerate=true" : "";
  const res = await post({
    endpoint: `/projects/${projectId}/artifacts/${artifactId}/chunks/${chunkId}/note${query}`,
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
