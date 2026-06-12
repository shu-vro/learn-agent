"use client";

import { ModelPicker } from "@/components/chat/model-picker";
import {
  type ModelPreset,
  modelSupportsReasoning,
  type ReasoningEffort,
} from "@/lib/api/models";
import type { ChatModelPreferences } from "@/lib/api/preferences";

export function ChatModelSettingsFields({
  value,
  onChange,
  models,
  reasoningEfforts,
}: {
  value: ChatModelPreferences;
  onChange: (value: ChatModelPreferences) => void;
  models: ModelPreset[];
  reasoningEfforts: ReasoningEffort[];
}) {
  return (
    <ModelPicker
      models={models}
      reasoningEfforts={reasoningEfforts}
      selectedModelId={value.default_model}
      reasoningEffort={value.reasoning_effort}
      onModelChange={(default_model) => {
        const model = models.find((item) => item.id === default_model);
        onChange({
          ...value,
          default_model,
          reasoning_effort: modelSupportsReasoning(model)
            ? (value.reasoning_effort ?? model?.reasoning_effort ?? null)
            : null,
        });
      }}
      onReasoningEffortChange={(reasoning_effort) =>
        onChange({ ...value, reasoning_effort })
      }
    />
  );
}
