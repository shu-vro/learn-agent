import { get } from "@/utils/fetch";

export type ReasoningEffort = "low" | "medium" | "high" | null;

export type ModelPreset = {
  id: string;
  model: string;
  provider: string;
  context_window: number;
  reasoning_effort: ReasoningEffort;
};

export type ModelPickerPresets = {
  models: ModelPreset[];
  reasoning_efforts: ReasoningEffort[];
};

export async function fetchModelPresets(): Promise<ModelPickerPresets | null> {
  const data = await get({ endpoint: "/models/" });
  if (data && typeof data === "object" && "models" in data) {
    return data as ModelPickerPresets;
  }
  return null;
}

export function modelSupportsReasoning(
  model: ModelPreset | undefined,
): boolean {
  return model?.reasoning_effort != null;
}

export function selectableReasoningEfforts(
  efforts: ReasoningEffort[],
): Exclude<ReasoningEffort, null>[] {
  return efforts.filter((effort): effort is Exclude<ReasoningEffort, null> =>
    Boolean(effort),
  );
}

export function formatReasoningEffort(effort: ReasoningEffort): string {
  if (!effort) {
    return "None";
  }
  return effort.charAt(0).toUpperCase() + effort.slice(1);
}
