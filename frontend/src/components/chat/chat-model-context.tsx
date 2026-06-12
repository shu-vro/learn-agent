"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import { useAuth } from "@/components/auth/auth-provider";
import {
  fetchModelPresets,
  type ModelPreset,
  modelSupportsReasoning,
  type ReasoningEffort,
} from "@/lib/api/models";

const DEFAULT_MODEL_ID = "omlx:gemma-4-e4b-it-4bit";

type ChatModelValue = {
  models: ModelPreset[];
  reasoningEfforts: ReasoningEffort[];
  selectedModelId: string;
  reasoningEffort: ReasoningEffort;
  setSelectedModelId: (modelId: string) => void;
  setReasoningEffort: (effort: ReasoningEffort) => void;
  loading: boolean;
};

const ChatModelContext = createContext<ChatModelValue | null>(null);

export function useChatModel() {
  const ctx = useContext(ChatModelContext);
  if (!ctx) {
    throw new Error("useChatModel must be used within ChatModelProvider");
  }
  return ctx;
}

function resolveReasoningEffort(
  userEffort: ReasoningEffort | undefined,
  model: ModelPreset | undefined,
): ReasoningEffort {
  if (!modelSupportsReasoning(model)) {
    return null;
  }
  if (userEffort) {
    return userEffort;
  }
  return model?.reasoning_effort ?? null;
}

export function ChatModelProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const [models, setModels] = useState<ModelPreset[]>([]);
  const [reasoningEfforts, setReasoningEfforts] = useState<ReasoningEffort[]>(
    [],
  );
  const [selectedModelId, setSelectedModelId] = useState(DEFAULT_MODEL_ID);
  const [reasoningEffort, setReasoningEffort] = useState<ReasoningEffort>(null);
  const [loading, setLoading] = useState(true);
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const presets = await fetchModelPresets();
      if (cancelled || !presets) {
        setLoading(false);
        return;
      }
      setModels(presets.models);
      setReasoningEfforts(presets.reasoning_efforts);
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (initialized || loading || models.length === 0) {
      return;
    }

    const preferredModelId =
      user?.preferences.chat?.default_model ?? DEFAULT_MODEL_ID;
    const model =
      models.find((item) => item.id === preferredModelId) ?? models[0];
    const modelId = model?.id ?? preferredModelId;

    setSelectedModelId(modelId);
    setReasoningEffort(
      resolveReasoningEffort(user?.preferences.chat?.reasoning_effort, model),
    );
    setInitialized(true);
  }, [initialized, loading, models, user]);

  const handleModelChange = useCallback(
    (modelId: string) => {
      setSelectedModelId(modelId);
      const model = models.find((item) => item.id === modelId);
      if (!modelSupportsReasoning(model)) {
        setReasoningEffort(null);
        return;
      }
      setReasoningEffort(
        (current) =>
          current ??
          user?.preferences.chat?.reasoning_effort ??
          model?.reasoning_effort ??
          null,
      );
    },
    [models, user?.preferences.chat?.reasoning_effort],
  );

  const value = useMemo<ChatModelValue>(
    () => ({
      models,
      reasoningEfforts,
      selectedModelId,
      reasoningEffort,
      setSelectedModelId: handleModelChange,
      setReasoningEffort,
      loading,
    }),
    [
      models,
      reasoningEfforts,
      selectedModelId,
      reasoningEffort,
      handleModelChange,
      loading,
    ],
  );

  return (
    <ChatModelContext.Provider value={value}>
      {children}
    </ChatModelContext.Provider>
  );
}
