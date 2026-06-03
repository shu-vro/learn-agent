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
  type ChatMessage,
  createLocalArtifact,
  createThread as createThreadRemote,
  deleteThread as deleteThreadRemote,
  getArtifactIngestionStatus,
  listArtifacts,
  listMessages,
  listThreads,
  type Thread,
  updateThread as updateThreadRemote,
  uploadArtifacts,
} from "@/lib/api/chat";
import type { IngestionUploadOptions } from "@/lib/api/preferences";

type ChatWorkspaceValue = {
  threads: Thread[];
  activeThreadId: string;
  setActiveThreadId: (id: string) => void;
  messages: ChatMessage[];
  appendUserMessage: (text: string) => void;
  newThread: () => void;
  renameThread: (threadId: string, name: string) => Promise<void>;
  deleteThread: (threadId: string) => Promise<void>;
  artifacts: Artifact[];
  selectedArtifactId: string | null;
  setSelectedArtifactId: (id: string | null) => void;
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

const assistantPlaceholder =
  "Here is a **placeholder** reply. Wire your model here when the backend is ready.";

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
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(
    null,
  );

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
        nextMsgs[th.id] = await listMessages(th.id);
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
      for (const artifactId of processingArtifactIdsRef.current) {
        if (cancelled) {
          return;
        }
        const updated = await getArtifactIngestionStatus(projectId, artifactId);
        if (!updated || cancelled) {
          continue;
        }
        setArtifacts((prev) => {
          const existing = prev.find((artifact) => artifact.id === updated.id);
          if (!existing) {
            return prev;
          }
          if (
            existing.ingestion_status === updated.ingestion_status &&
            existing.name === updated.name &&
            existing.ingestion_stage === updated.ingestion_stage &&
            existing.ingestion_stage_label === updated.ingestion_stage_label &&
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

  const appendUserMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !activeThreadId) {
        return;
      }
      const userMsg: ChatMessage = {
        id: nanoid(),
        role: "user",
        content: trimmed,
      };
      const assistantMsg: ChatMessage = {
        id: nanoid(),
        role: "assistant",
        content: assistantPlaceholder,
      };
      setMessagesByThread((prev) => ({
        ...prev,
        [activeThreadId]: [
          ...(prev[activeThreadId] ?? []),
          userMsg,
          assistantMsg,
        ],
      }));
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
      // Optimistically remove locally
      setArtifacts((prev) => prev.filter((a) => a.id !== artifactId));
      if (selectedArtifactId === artifactId) {
        setSelectedArtifactId(null);
      }
      if (!projectId) return;
      try {
        const { deleteArtifact: apiDelete } = await import("@/lib/api/chat");
        await apiDelete(projectId, artifactId);
      } catch (_err) {
        // best-effort: ignore failures for now
      }
    },
    [projectId, selectedArtifactId],
  );

  const value = useMemo<ChatWorkspaceValue>(
    () => ({
      threads,
      activeThreadId,
      setActiveThreadId,
      messages,
      appendUserMessage,
      newThread,
      renameThread,
      deleteThread,
      artifacts,
      selectedArtifactId,
      setSelectedArtifactId,
      addArtifactFromFile,
      addArtifactsFromFiles,
      deleteArtifact,
    }),
    [
      threads,
      activeThreadId,
      messages,
      appendUserMessage,
      newThread,
      renameThread,
      deleteThread,
      artifacts,
      selectedArtifactId,
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
