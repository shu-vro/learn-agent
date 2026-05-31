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
  getArtifactIngestionStatus,
  listArtifacts,
  listMessages,
  listThreads,
  type Thread,
  uploadArtifact,
} from "@/lib/api/chat";
import type { IngestionUploadOptions } from "@/lib/api/preferences";

type ChatWorkspaceValue = {
  threads: Thread[];
  activeThreadId: string;
  setActiveThreadId: (id: string) => void;
  messages: ChatMessage[];
  appendUserMessage: (text: string) => void;
  newThread: () => void;
  artifacts: Artifact[];
  selectedArtifactId: string | null;
  setSelectedArtifactId: (id: string | null) => void;
  addArtifactFromFile: (
    file: File,
    ingestion?: IngestionUploadOptions,
  ) => Promise<void>;
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

  const newThread = useCallback(() => {
    const id = nanoid();
    const title = "New thread";
    setThreads((prev) => [...prev, { id, title }]);
    setMessagesByThread((prev) => ({ ...prev, [id]: [] }));
    setActiveThreadId(id);
  }, []);

  const addArtifactFromFile = useCallback(
    async (file: File, ingestion?: IngestionUploadOptions) => {
      if (projectId) {
        const tempId = `upload-${nanoid()}`;
        setArtifacts((prev) => [
          ...prev,
          {
            id: tempId,
            name: file.name,
            chunks: {},
            ingestion_status: "uploading",
            upload_progress: 0,
          },
        ]);
        setSelectedArtifactId(tempId);

        try {
          const created = await uploadArtifact(projectId, file, {
            ingestion,
            onUploadProgress: (percent) => {
              setArtifacts((prev) =>
                prev.map((artifact) =>
                  artifact.id === tempId
                    ? { ...artifact, upload_progress: percent }
                    : artifact,
                ),
              );
            },
          });
          if (created) {
            setArtifacts((prev) => {
              const withoutTemp = prev.filter(
                (artifact) => artifact.id !== tempId,
              );
              const existingIndex = withoutTemp.findIndex(
                (artifact) => artifact.id === created.id,
              );
              if (existingIndex >= 0) {
                return withoutTemp.map((artifact) =>
                  artifact.id === created.id ? created : artifact,
                );
              }
              return [...withoutTemp, created];
            });
            setSelectedArtifactId(created.id);
          } else {
            setArtifacts((prev) =>
              prev.map((artifact) =>
                artifact.id === tempId
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
          setArtifacts((prev) =>
            prev.map((artifact) =>
              artifact.id === tempId
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
      let content = "";
      try {
        content = await file.text();
      } catch {
        content = `_Could not read file as text: ${file.name}_`;
      }
      const art = createLocalArtifact(file.name, content || "_Empty file_");
      setArtifacts((prev) => [...prev, art]);
      setSelectedArtifactId(art.id);
    },
    [projectId],
  );

  const value = useMemo<ChatWorkspaceValue>(
    () => ({
      threads,
      activeThreadId,
      setActiveThreadId,
      messages,
      appendUserMessage,
      newThread,
      artifacts,
      selectedArtifactId,
      setSelectedArtifactId,
      addArtifactFromFile,
    }),
    [
      threads,
      activeThreadId,
      messages,
      appendUserMessage,
      newThread,
      artifacts,
      selectedArtifactId,
      addArtifactFromFile,
    ],
  );

  return (
    <ChatWorkspaceContext.Provider value={value}>
      {children}
    </ChatWorkspaceContext.Provider>
  );
}
