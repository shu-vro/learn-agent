"use client";

import { nanoid } from "nanoid";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  type Artifact,
  type ChatArtifact,
  type ChatMessage,
  type ChatTimelineItem,
  type ChatToolCall,
  type ChatUsage,
  createLocalArtifact,
  createThread as createThreadRemote,
  deleteThread as deleteThreadRemote,
  getArtifactIngestionStatus,
  listArtifacts,
  listMessages,
  listThreads,
  parseChatArtifacts,
  parseChatUsage,
  streamChat,
  type Thread,
  updateThread as updateThreadRemote,
  uploadArtifacts,
} from "@/lib/api/chat";
import type { IngestionUploadOptions } from "@/lib/api/preferences";

type SendMessageOptions = {
  selection?: string | null;
  referenceId?: string | null;
  images?: string[] | null;
};

type ChatWorkspaceValue = {
  projectId: string | null;
  threads: Thread[];
  activeThreadId: string;
  setActiveThreadId: (id: string) => void;
  messages: ChatMessage[];
  appendUserMessage: (text: string, options?: SendMessageOptions) => void;
  regenerateMessage: (assistantChatId: string, messageId: string) => void;
  setActiveBranch: (assistantChatId: string, index: number) => void;
  isStreaming: boolean;
  newThread: () => void;
  renameThread: (threadId: string, name: string) => Promise<void>;
  deleteThread: (threadId: string) => Promise<void>;
  artifacts: Artifact[];
  selectedArtifactId: string | null;
  setSelectedArtifactId: (id: string | null) => void;
  /** Chunk the preview panel should scroll to and highlight, if any. */
  focusedChunkId: string | null;
  /** Bumped on every focusChunk call, so repeat clicks re-scroll / re-open. */
  chunkFocusSeq: number;
  /** Open a document citation in the preview panel. */
  focusChunk: (documentId: string, chunkId: string) => void;
  addArtifactFromFile: (
    file: File,
    ingestion?: IngestionUploadOptions,
  ) => Promise<void>;
  addArtifactsFromFiles: (
    files: File[],
    ingestion?: IngestionUploadOptions,
  ) => Promise<void>;
  deleteArtifact: (artifactId: string) => Promise<void>;
};

const ChatWorkspaceContext = createContext<ChatWorkspaceValue | null>(null);

export function useChatWorkspace() {
  const ctx = useContext(ChatWorkspaceContext);
  if (!ctx) {
    throw new Error(
      "useChatWorkspace must be used within ChatWorkspaceProvider",
    );
  }
  return ctx;
}

function updateThreadMessages(
  prev: Record<string, ChatMessage[]>,
  threadId: string,
  updater: (messages: ChatMessage[]) => ChatMessage[],
): Record<string, ChatMessage[]> {
  return {
    ...prev,
    [threadId]: updater(prev[threadId] ?? []),
  };
}

function patchAssistantBranch(
  messages: ChatMessage[],
  assistantChatId: string,
  branchId: string,
  patch: Partial<{
    content: string;
    timeline: ChatTimelineItem[];
    thinking: string;
    tools: ChatToolCall[];
    streaming: boolean;
    activeThinkingStep: number | null;
    usage: ChatUsage | null;
    artifacts: ChatArtifact[];
  }>,
): ChatMessage[] {
  return messages.map((msg) => {
    if (msg.id !== assistantChatId || msg.role !== "assistant") {
      return msg;
    }
    const branches = [...(msg.branches ?? [])];
    const idx = branches.findIndex((b) => b.id === branchId);
    if (idx < 0) {
      return msg;
    }
    const nextBranch = { ...branches[idx], ...patch };
    branches[idx] = nextBranch;
    const active = msg.activeBranch ?? idx;
    const current = branches[active] ?? nextBranch;
    return {
      ...msg,
      branches,
      content: current.content,
      timeline: current.timeline,
      thinking: current.thinking,
      tools: current.tools,
      streaming: current.streaming,
      activeThinkingStep: current.activeThinkingStep,
      usage: current.usage,
      artifacts: current.artifacts,
    };
  });
}

function syncBranchDerived<T extends { timeline?: ChatTimelineItem[] }>(
  branch: T,
): T & {
  thinking: string;
  tools: Extract<ChatTimelineItem, { kind: "tool" }>[];
} {
  const timeline = branch.timeline ?? [];
  return {
    ...branch,
    thinking: timeline
      .filter((i) => i.kind === "thinking")
      .map((i) => i.text)
      .join("\n\n"),
    tools: timeline.filter(
      (i): i is Extract<ChatTimelineItem, { kind: "tool" }> =>
        i.kind === "tool",
    ),
  };
}

export function ChatWorkspaceProvider({
  children,
  projectId = null,
}: {
  children: React.ReactNode;
  projectId?: string | null;
}) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState("");
  const [messagesByThread, setMessagesByThread] = useState<
    Record<string, ChatMessage[]>
  >({});
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [focusedChunk, setFocusedChunk] = useState<{
    id: string;
    seq: number;
  } | null>(null);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(
    null,
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const abortStreamRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [t, a] = await Promise.all([
        listThreads(projectId),
        listArtifacts(projectId),
      ]);
      if (cancelled) {
        return;
      }
      setThreads(t);
      const firstId = t[0]?.id ?? "";
      setActiveThreadId(firstId);
      const nextMsgs: Record<string, ChatMessage[]> = {};
      for (const th of t) {
        nextMsgs[th.id] = await listMessages(projectId, th.id);
      }
      if (cancelled) {
        return;
      }
      setMessagesByThread(nextMsgs);
      setArtifacts(a);
      setSelectedArtifactId(a[0]?.id ?? null);
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  const processingArtifactIdsRef = useRef<string[]>([]);
  const processingArtifactKey = useMemo(() => {
    const ids = artifacts
      .filter((artifact) => artifact.ingestion_status === "processing")
      .map((artifact) => artifact.id)
      .sort();
    processingArtifactIdsRef.current = ids;
    return ids.join(",");
  }, [artifacts]);

  useEffect(() => {
    if (!projectId || !processingArtifactKey) {
      return;
    }
    let cancelled = false;

    const poll = async () => {
      const ids = processingArtifactIdsRef.current;
      for (const id of ids) {
        const updated = await getArtifactIngestionStatus(projectId, id);
        if (cancelled || !updated) {
          continue;
        }
        setArtifacts((prev) => {
          const existing = prev.find((artifact) => artifact.id === updated.id);
          if (
            existing &&
            existing.ingestion_status === updated.ingestion_status &&
            existing.ingestion_stage === updated.ingestion_stage &&
            existing.ingestion_progress === updated.ingestion_progress &&
            Object.keys(existing.chunks).length ===
              Object.keys(updated.chunks).length
          ) {
            return prev;
          }
          return prev.map((artifact) =>
            artifact.id === updated.id ? updated : artifact,
          );
        });
      }
    };

    void poll();
    const interval = window.setInterval(() => {
      void poll();
    }, 3000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [projectId, processingArtifactKey]);

  const messages = messagesByThread[activeThreadId] ?? [];

  const runStream = useCallback(
    (opts: {
      query?: string;
      threadId: string | null;
      messageId?: string;
      selection?: string | null;
      referenceId?: string | null;
      images?: string[] | null;
      optimisticThreadId?: string;
    }) => {
      if (!projectId) {
        return;
      }
      abortStreamRef.current?.();
      setIsStreaming(true);

      let workingThreadId = opts.optimisticThreadId ?? opts.threadId ?? "";
      let assistantChatId = "";
      let branchId = "";

      const stop = streamChat(
        projectId,
        {
          query: opts.query,
          threadId: opts.threadId,
          messageId: opts.messageId,
          selection: opts.selection,
          referenceId: opts.referenceId,
          images: opts.images,
        },
        {
          onEvent: (event, data) => {
            if (event === "thread") {
              const thread = data.thread as Thread | undefined;
              if (thread?.id) {
                workingThreadId = thread.id;
                setThreads((prev) => {
                  const exists = prev.some((t) => t.id === thread.id);
                  if (exists) {
                    return prev.map((t) =>
                      t.id === thread.id ? { ...t, ...thread } : t,
                    );
                  }
                  return [...prev, thread];
                });
                setActiveThreadId(thread.id);
                setMessagesByThread((prev) => {
                  if (prev[thread.id]) {
                    return prev;
                  }
                  const fromKey = opts.optimisticThreadId;
                  if (fromKey && prev[fromKey]) {
                    const next = { ...prev };
                    next[thread.id] = next[fromKey];
                    delete next[fromKey];
                    return next;
                  }
                  return { ...prev, [thread.id]: [] };
                });
              }
            }

            if (event === "user_message") {
              const message = data.message as {
                id: string;
                chat_id: string;
                message: string;
                selection?: string | null;
                reference_id?: string | null;
                image_urls?: string[] | null;
              };
              setMessagesByThread((prev) =>
                updateThreadMessages(prev, workingThreadId, (msgs) => {
                  const withoutTemp = msgs.filter(
                    (m) => !(m.role === "user" && m.id.startsWith("temp-")),
                  );
                  const existingTemp = msgs.find(
                    (m) => m.role === "user" && m.id.startsWith("temp-"),
                  );
                  if (withoutTemp.some((m) => m.id === message.id)) {
                    return withoutTemp;
                  }
                  return [
                    ...withoutTemp,
                    {
                      id: message.id,
                      role: "user" as const,
                      content: message.message,
                      chatId: message.chat_id,
                      selection: message.selection,
                      referenceId: message.reference_id,
                      // Prefer optimistic local previews until S3 URLs land.
                      imageUrls: message.image_urls?.length
                        ? message.image_urls
                        : (existingTemp?.imageUrls ?? null),
                    },
                  ];
                }),
              );
            }

            if (event === "assistant_message") {
              assistantChatId = String(data.chat_id ?? "");
              branchId = String(data.message_id ?? "");
              const regenerate = Boolean(data.regenerate);
              setMessagesByThread((prev) =>
                updateThreadMessages(prev, workingThreadId, (msgs) => {
                  const cleaned = msgs.filter(
                    (m) =>
                      !(m.role === "assistant" && m.id.startsWith("temp-")),
                  );
                  if (regenerate) {
                    return cleaned.map((m) => {
                      if (m.id !== assistantChatId) {
                        return m;
                      }
                      const branches = [
                        ...(m.branches ?? []),
                        {
                          id: branchId,
                          content: "",
                          timeline: [],
                          thinking: "",
                          tools: [],
                          streaming: true,
                        },
                      ];
                      const activeBranch = branches.length - 1;
                      return {
                        ...m,
                        branches,
                        activeBranch,
                        content: "",
                        timeline: [],
                        thinking: "",
                        tools: [],
                        streaming: true,
                      };
                    });
                  }
                  if (cleaned.some((m) => m.id === assistantChatId)) {
                    return cleaned;
                  }
                  return [
                    ...cleaned,
                    {
                      id: assistantChatId,
                      role: "assistant" as const,
                      content: "",
                      chatId: assistantChatId,
                      branches: [
                        {
                          id: branchId,
                          content: "",
                          timeline: [],
                          thinking: "",
                          tools: [],
                          streaming: true,
                        },
                      ],
                      activeBranch: 0,
                      timeline: [],
                      streaming: true,
                    },
                  ];
                }),
              );
            }

            if (event === "thinking" && assistantChatId && branchId) {
              const delta = String(data.delta ?? "");
              const step = Number(data.step ?? 0);
              setMessagesByThread((prev) =>
                updateThreadMessages(prev, workingThreadId, (msgs) =>
                  msgs.map((msg) => {
                    if (msg.id !== assistantChatId) return msg;
                    const branches = [...(msg.branches ?? [])];
                    const idx = branches.findIndex((b) => b.id === branchId);
                    if (idx < 0) return msg;
                    const timeline = [...(branches[idx].timeline ?? [])];
                    const existingIdx = timeline.findIndex(
                      (i) => i.kind === "thinking" && i.step === step,
                    );
                    if (existingIdx >= 0) {
                      const cur = timeline[existingIdx];
                      if (cur.kind === "thinking") {
                        timeline[existingIdx] = {
                          ...cur,
                          text: `${cur.text}${delta}`,
                        };
                      }
                    } else {
                      timeline.push({
                        kind: "thinking",
                        id: `think-${step}`,
                        step,
                        text: delta,
                      });
                    }
                    const nextBranch = syncBranchDerived({
                      ...branches[idx],
                      timeline,
                      activeThinkingStep: step,
                    });
                    branches[idx] = nextBranch;
                    const active = msg.activeBranch ?? idx;
                    return {
                      ...msg,
                      branches,
                      timeline: active === idx ? timeline : msg.timeline,
                      thinking:
                        active === idx ? nextBranch.thinking : msg.thinking,
                      tools: active === idx ? nextBranch.tools : msg.tools,
                      activeThinkingStep:
                        active === idx
                          ? nextBranch.activeThinkingStep
                          : msg.activeThinkingStep,
                    };
                  }),
                ),
              );
            }

            if (event === "token" && assistantChatId && branchId) {
              const delta = String(data.delta ?? "");
              setMessagesByThread((prev) =>
                updateThreadMessages(prev, workingThreadId, (msgs) =>
                  msgs.map((msg) => {
                    if (msg.id !== assistantChatId) return msg;
                    const branches = [...(msg.branches ?? [])];
                    const idx = branches.findIndex((b) => b.id === branchId);
                    if (idx < 0) return msg;
                    const content = `${branches[idx].content}${delta}`;
                    branches[idx] = {
                      ...branches[idx],
                      content,
                      activeThinkingStep: null,
                    };
                    const active = msg.activeBranch ?? idx;
                    return {
                      ...msg,
                      branches,
                      content: active === idx ? content : msg.content,
                      activeThinkingStep:
                        active === idx ? null : msg.activeThinkingStep,
                    };
                  }),
                ),
              );
            }

            if (event === "tool" && assistantChatId && branchId) {
              const phase = String(data.phase ?? "");
              const toolId = String(data.id ?? nanoid());
              const step = Number(data.step ?? 0);
              const name = String(data.name ?? "tool");
              const args =
                data.args && typeof data.args === "object"
                  ? (data.args as Record<string, unknown>)
                  : {};
              setMessagesByThread((prev) =>
                updateThreadMessages(prev, workingThreadId, (msgs) =>
                  msgs.map((msg) => {
                    if (msg.id !== assistantChatId) return msg;
                    const branches = [...(msg.branches ?? [])];
                    const idx = branches.findIndex((b) => b.id === branchId);
                    if (idx < 0) return msg;
                    const timeline = [...(branches[idx].timeline ?? [])];
                    if (phase === "start") {
                      timeline.push({
                        kind: "tool",
                        id: toolId,
                        name,
                        args,
                        state: "input-available",
                        step,
                      });
                    } else if (phase === "result") {
                      const existing = timeline.findIndex(
                        (t) => t.kind === "tool" && t.id === toolId,
                      );
                      const next: ChatTimelineItem = {
                        kind: "tool",
                        id: toolId,
                        name,
                        args,
                        result: data.result,
                        state: "output-available",
                        step,
                      };
                      if (existing >= 0) {
                        timeline[existing] = next;
                      } else {
                        timeline.push(next);
                      }
                    }
                    const nextBranch = syncBranchDerived({
                      ...branches[idx],
                      timeline,
                      activeThinkingStep: null,
                    });
                    branches[idx] = nextBranch;
                    const active = msg.activeBranch ?? idx;
                    return {
                      ...msg,
                      branches,
                      timeline: active === idx ? timeline : msg.timeline,
                      thinking:
                        active === idx ? nextBranch.thinking : msg.thinking,
                      tools: active === idx ? nextBranch.tools : msg.tools,
                      activeThinkingStep:
                        active === idx ? null : msg.activeThinkingStep,
                    };
                  }),
                ),
              );
            }

            if (event === "done" && assistantChatId && branchId) {
              const finalText = String(data.message ?? "");
              const usage = parseChatUsage(data.usage);
              const artifacts = parseChatArtifacts(data.artifacts);
              setMessagesByThread((prev) =>
                updateThreadMessages(prev, workingThreadId, (msgs) =>
                  patchAssistantBranch(msgs, assistantChatId, branchId, {
                    content: finalText || undefined,
                    streaming: false,
                    activeThinkingStep: null,
                    usage,
                    artifacts,
                  }).map((msg) =>
                    msg.id === assistantChatId
                      ? {
                          ...msg,
                          streaming: false,
                          activeThinkingStep: null,
                          usage,
                          artifacts,
                        }
                      : msg,
                  ),
                ),
              );
            }

            if (event === "error") {
              const message = String(data.message ?? "Chat failed");
              setMessagesByThread((prev) =>
                updateThreadMessages(prev, workingThreadId, (msgs) =>
                  msgs.map((msg) => {
                    if (msg.id !== assistantChatId) return msg;
                    const content = msg.content || `*Error:* ${message}`;
                    const branches = (msg.branches ?? []).map((b) => ({
                      ...b,
                      streaming: false,
                      activeThinkingStep: null,
                    }));
                    return {
                      ...msg,
                      content,
                      streaming: false,
                      activeThinkingStep: null,
                      branches,
                    };
                  }),
                ),
              );
            }
          },
          onError: () => {
            setIsStreaming(false);
          },
          onDone: () => {
            setIsStreaming(false);
            abortStreamRef.current = null;
          },
        },
      );
      abortStreamRef.current = stop;
    },
    [projectId],
  );

  const appendUserMessage = useCallback(
    (text: string, options?: SendMessageOptions) => {
      const trimmed = text.trim();
      const images = (options?.images ?? []).filter(Boolean);
      if (!trimmed && images.length === 0) {
        return;
      }

      if (!projectId) {
        if (!activeThreadId) return;
        const userMsg: ChatMessage = {
          id: nanoid(),
          role: "user",
          content: trimmed,
          selection: options?.selection,
          referenceId: options?.referenceId,
          imageUrls: images.length ? images : null,
        };
        const assistantMsg: ChatMessage = {
          id: nanoid(),
          role: "assistant",
          content:
            "Connect to a project to stream real replies from the RAG agent.",
        };
        setMessagesByThread((prev) => ({
          ...prev,
          [activeThreadId]: [
            ...(prev[activeThreadId] ?? []),
            userMsg,
            assistantMsg,
          ],
        }));
        return;
      }

      const threadId = activeThreadId || null;
      const tempUserId = `temp-${nanoid()}`;
      const tempAssistantId = `temp-${nanoid()}`;

      if (threadId) {
        setMessagesByThread((prev) => ({
          ...prev,
          [threadId]: [
            ...(prev[threadId] ?? []),
            {
              id: tempUserId,
              role: "user",
              content: trimmed,
              selection: options?.selection,
              referenceId: options?.referenceId,
              imageUrls: images.length ? images : null,
            },
            {
              id: tempAssistantId,
              role: "assistant",
              content: "",
              streaming: true,
              branches: [
                {
                  id: tempAssistantId,
                  content: "",
                  streaming: true,
                },
              ],
              activeBranch: 0,
            },
          ],
        }));
      }

      runStream({
        query: trimmed || undefined,
        threadId,
        selection: options?.selection,
        referenceId: options?.referenceId,
        images: images.length ? images : undefined,
      });
    },
    [activeThreadId, projectId, runStream],
  );

  const regenerateMessage = useCallback(
    (assistantChatId: string, messageId: string) => {
      if (!projectId || !activeThreadId || isStreaming) {
        return;
      }
      runStream({
        threadId: activeThreadId,
        messageId,
      });
      void assistantChatId;
    },
    [activeThreadId, isStreaming, projectId, runStream],
  );

  const setActiveBranch = useCallback(
    (assistantChatId: string, index: number) => {
      if (!activeThreadId) return;
      setMessagesByThread((prev) =>
        updateThreadMessages(prev, activeThreadId, (msgs) =>
          msgs.map((msg) => {
            if (msg.id !== assistantChatId || !msg.branches) return msg;
            const branch = msg.branches[index];
            if (!branch) return msg;
            return {
              ...msg,
              activeBranch: index,
              content: branch.content,
              timeline: branch.timeline,
              thinking: branch.thinking,
              tools: branch.tools,
            };
          }),
        ),
      );
    },
    [activeThreadId],
  );

  const newThread = useCallback(async () => {
    if (projectId) {
      try {
        const created = await createThreadRemote(projectId);
        if (!created) {
          return;
        }
        setThreads((prev) => [...prev, created]);
        setMessagesByThread((prev) => ({ ...prev, [created.id]: [] }));
        setActiveThreadId(created.id);
      } catch {
        // best-effort
      }
      return;
    }
    const id = nanoid();
    const now = new Date();
    const thread: Thread = {
      id,
      thread_name: "",
      extra: {},
      created_at: now,
      updated_at: now,
    };
    setThreads((prev) => [...prev, thread]);
    setMessagesByThread((prev) => ({ ...prev, [id]: [] }));
    setActiveThreadId(id);
  }, [projectId]);

  const renameThread = useCallback(
    async (threadId: string, name: string) => {
      const trimmed = name.trim();
      if (!trimmed) {
        return;
      }
      setThreads((prev) =>
        prev.map((t) =>
          t.id === threadId ? { ...t, thread_name: trimmed } : t,
        ),
      );
      if (!projectId) {
        return;
      }
      try {
        const updated = await updateThreadRemote(projectId, threadId, trimmed);
        if (updated) {
          setThreads((prev) =>
            prev.map((t) => (t.id === threadId ? updated : t)),
          );
        }
      } catch {
        // best-effort: keep optimistic name
      }
    },
    [projectId],
  );

  const deleteThread = useCallback(
    async (threadId: string) => {
      setThreads((prev) => {
        const next = prev.filter((t) => t.id !== threadId);
        setActiveThreadId((active) =>
          active === threadId ? (next[0]?.id ?? "") : active,
        );
        return next;
      });
      setMessagesByThread((prev) => {
        const next = { ...prev };
        delete next[threadId];
        return next;
      });
      if (!projectId) {
        return;
      }
      try {
        await deleteThreadRemote(projectId, threadId);
      } catch {
        // best-effort
      }
    },
    [projectId],
  );

  const addArtifactsFromFiles = useCallback(
    async (files: File[], ingestion?: IngestionUploadOptions) => {
      if (!files.length) {
        return;
      }
      if (projectId) {
        const tempEntries = files.map((file) => ({
          tempId: `upload-${nanoid()}`,
          file,
        }));
        setArtifacts((prev) => [
          ...prev,
          ...tempEntries.map(({ tempId, file }) => ({
            id: tempId,
            name: file.name,
            chunks: {},
            ingestion_status: "uploading" as const,
            upload_progress: 0,
          })),
        ]);
        setSelectedArtifactId(tempEntries[0]?.tempId ?? null);

        try {
          const created = await uploadArtifacts(projectId, files, {
            ingestion,
            onUploadProgress: (percent) => {
              setArtifacts((prev) =>
                prev.map((artifact) =>
                  tempEntries.some(({ tempId }) => tempId === artifact.id)
                    ? { ...artifact, upload_progress: percent }
                    : artifact,
                ),
              );
            },
          });
          const tempIds = new Set(tempEntries.map(({ tempId }) => tempId));
          setArtifacts((prev) => {
            let next = prev.filter((artifact) => !tempIds.has(artifact.id));
            for (const artifact of created) {
              const existingIndex = next.findIndex(
                (item) => item.id === artifact.id,
              );
              if (existingIndex >= 0) {
                next = next.map((item) =>
                  item.id === artifact.id ? artifact : item,
                );
              } else {
                next = [...next, artifact];
              }
            }
            return next;
          });
          if (created[0]) {
            setSelectedArtifactId(created[0].id);
          } else {
            setArtifacts((prev) =>
              prev.map((artifact) =>
                tempIds.has(artifact.id)
                  ? {
                      ...artifact,
                      ingestion_status: "failed",
                      upload_progress: undefined,
                    }
                  : artifact,
              ),
            );
          }
        } catch {
          const tempIds = new Set(tempEntries.map(({ tempId }) => tempId));
          setArtifacts((prev) =>
            prev.map((artifact) =>
              tempIds.has(artifact.id)
                ? {
                    ...artifact,
                    ingestion_status: "failed",
                    upload_progress: undefined,
                  }
                : artifact,
            ),
          );
        }
        return;
      }

      for (const file of files) {
        let content = "";
        try {
          content = await file.text();
        } catch {
          content = `_Could not read file as text: ${file.name}_`;
        }
        const art = createLocalArtifact(file.name, content || "_Empty file_");
        setArtifacts((prev) => [...prev, art]);
        setSelectedArtifactId(art.id);
      }
    },
    [projectId],
  );

  const addArtifactFromFile = useCallback(
    async (file: File, ingestion?: IngestionUploadOptions) => {
      await addArtifactsFromFiles([file], ingestion);
    },
    [addArtifactsFromFiles],
  );

  const deleteArtifact = useCallback(
    async (artifactId: string) => {
      setArtifacts((prev) => prev.filter((a) => a.id !== artifactId));
      if (selectedArtifactId === artifactId) {
        setSelectedArtifactId(null);
      }
      if (!projectId) return;
      try {
        const { deleteArtifact: apiDelete } = await import("@/lib/api/chat");
        await apiDelete(projectId, artifactId);
      } catch (_err) {
        // best-effort
      }
    },
    [projectId, selectedArtifactId],
  );

  const focusChunk = useCallback((documentId: string, chunkId: string) => {
    setSelectedArtifactId(documentId);
    setFocusedChunk((prev) => ({ id: chunkId, seq: (prev?.seq ?? 0) + 1 }));
  }, []);

  const value = useMemo<ChatWorkspaceValue>(
    () => ({
      projectId,
      threads,
      activeThreadId,
      setActiveThreadId,
      messages,
      appendUserMessage,
      regenerateMessage,
      setActiveBranch,
      isStreaming,
      newThread,
      renameThread,
      deleteThread,
      artifacts,
      selectedArtifactId,
      setSelectedArtifactId,
      focusedChunkId: focusedChunk?.id ?? null,
      chunkFocusSeq: focusedChunk?.seq ?? 0,
      focusChunk,
      addArtifactFromFile,
      addArtifactsFromFiles,
      deleteArtifact,
    }),
    [
      projectId,
      threads,
      activeThreadId,
      messages,
      appendUserMessage,
      regenerateMessage,
      setActiveBranch,
      isStreaming,
      newThread,
      renameThread,
      deleteThread,
      artifacts,
      selectedArtifactId,
      focusedChunk,
      focusChunk,
      addArtifactFromFile,
      addArtifactsFromFiles,
      deleteArtifact,
    ],
  );

  return (
    <ChatWorkspaceContext.Provider value={value}>
      {children}
    </ChatWorkspaceContext.Provider>
  );
}
