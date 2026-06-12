"use client";

import { CheckIcon, ChevronDownIcon } from "lucide-react";
import { useMemo, useState } from "react";

import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorEmpty,
  ModelSelectorGroup,
  ModelSelectorInput,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorLogo,
  ModelSelectorName,
  ModelSelectorSeparator,
  ModelSelectorTrigger,
} from "@/components/ai-elements/model-selector";
import { PromptInputButton } from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import {
  formatReasoningEffort,
  type ModelPreset,
  modelSupportsReasoning,
  type ReasoningEffort,
  selectableReasoningEfforts,
} from "@/lib/api/models";
import { cn } from "@/lib/utils";

type ModelPickerProps = {
  models: ModelPreset[];
  reasoningEfforts: ReasoningEffort[];
  selectedModelId: string;
  reasoningEffort: ReasoningEffort;
  onModelChange: (modelId: string) => void;
  onReasoningEffortChange: (effort: ReasoningEffort) => void;
  className?: string;
  compact?: boolean;
  variant?: "default" | "toolbar";
};

export function ModelPicker({
  models,
  reasoningEfforts,
  selectedModelId,
  reasoningEffort,
  onModelChange,
  onReasoningEffortChange,
  className,
  compact = false,
  variant = "default",
}: ModelPickerProps) {
  const isToolbar = variant === "toolbar";
  const [open, setOpen] = useState(false);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) ?? models[0],
    [models, selectedModelId],
  );

  const modelsByProvider = useMemo(() => {
    const grouped = new Map<string, ModelPreset[]>();
    for (const model of models) {
      const list = grouped.get(model.provider) ?? [];
      list.push(model);
      grouped.set(model.provider, list);
    }
    return grouped;
  }, [models]);

  const effortOptions = useMemo(
    () => selectableReasoningEfforts(reasoningEfforts),
    [reasoningEfforts],
  );

  const showReasoning =
    selectedModel != null && modelSupportsReasoning(selectedModel);

  if (!selectedModel) {
    return null;
  }

  const handleModelSelect = (modelId: string) => {
    onModelChange(modelId);
    const model = models.find((item) => item.id === modelId);
    if (!modelSupportsReasoning(model)) {
      setOpen(false);
    }
  };

  return (
    <div className={className}>
      <ModelSelector open={open} onOpenChange={setOpen}>
        <ModelSelectorTrigger
          render={
            isToolbar ? (
              <PromptInputButton
                className={cn(
                  "h-8 max-w-44 justify-between gap-1.5 px-2 sm:max-w-96",
                )}
                size="sm"
              />
            ) : (
              <Button
                type="button"
                variant="outline"
                size={compact ? "sm" : "default"}
                className={cn(
                  "max-w-full justify-between gap-2 rounded-xl",
                  compact && "h-8 px-2.5 text-xs",
                )}
              />
            )
          }
        >
          <span className="flex min-w-0 items-center gap-2">
            <ModelSelectorLogo provider={selectedModel.provider} />
            <ModelSelectorName className="text-xs sm:text-sm min-w-0 overflow-visible whitespace-normal text-clip">
              {selectedModel.model}
              {showReasoning && reasoningEffort ? (
                <span className="text-muted-foreground">
                  {" · "}
                  {formatReasoningEffort(reasoningEffort)}
                </span>
              ) : null}
            </ModelSelectorName>
          </span>
          <ChevronDownIcon className="size-3.5 shrink-0 opacity-60" />
        </ModelSelectorTrigger>
        <ModelSelectorContent className="w-[min(100vw-2rem,28rem)]">
          <ModelSelectorInput placeholder="Search models…" />
          <ModelSelectorList>
            <ModelSelectorEmpty>No models found.</ModelSelectorEmpty>
            {[...modelsByProvider.entries()].map(
              ([provider, providerModels]) => (
                <ModelSelectorGroup
                  key={provider}
                  heading={provider.charAt(0).toUpperCase() + provider.slice(1)}
                >
                  {providerModels.map((model) => (
                    <ModelSelectorItem
                      key={model.id}
                      value={`${model.id} ${model.model} ${provider}`}
                      onSelect={() => handleModelSelect(model.id)}
                    >
                      <ModelSelectorLogo provider={model.provider} />
                      <ModelSelectorName>{model.model}</ModelSelectorName>
                      {selectedModelId === model.id ? (
                        <CheckIcon className="ml-auto size-4 opacity-70" />
                      ) : null}
                    </ModelSelectorItem>
                  ))}
                </ModelSelectorGroup>
              ),
            )}
          </ModelSelectorList>
          {showReasoning ? (
            <>
              <ModelSelectorSeparator />
              <div className="px-3 py-3">
                <p className="mb-2 font-medium text-muted-foreground text-xs">
                  Reasoning effort
                </p>
                <div className="flex flex-wrap gap-2">
                  {effortOptions.map((effort) => (
                    <Button
                      key={effort}
                      type="button"
                      size="sm"
                      variant={
                        reasoningEffort === effort ? "default" : "outline"
                      }
                      className="h-8 rounded-lg px-3 text-xs"
                      onClick={() => {
                        onReasoningEffortChange(effort);
                        setOpen(false);
                      }}
                    >
                      {formatReasoningEffort(effort)}
                    </Button>
                  ))}
                </div>
              </div>
            </>
          ) : null}
        </ModelSelectorContent>
      </ModelSelector>
    </div>
  );
}
