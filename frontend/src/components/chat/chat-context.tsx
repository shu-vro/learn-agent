"use client";

import { nanoid } from "nanoid";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  type Artifact,
  type ChatMessage,
  createLocalArtifact,
  listArtifacts,
  listMessages,
  listThreads,
  type Thread,
  uploadArtifact,
} from "@/lib/api/chat";

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
  addArtifactFromFile: (file: File) => Promise<void>;
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
    async (file: File) => {
      if (projectId) {
        const created = await uploadArtifact(projectId, file);
        if (created) {
          setArtifacts((prev) => [...prev, created]);
          setSelectedArtifactId(created.id);
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
