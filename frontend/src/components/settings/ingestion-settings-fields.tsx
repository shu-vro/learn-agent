"use client";

import { useEffect, useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  fetchIngestionConfig,
  type IngestionPreferences,
} from "@/lib/api/preferences";
import { cn } from "@/lib/utils";

export type IngestionSettingsValue = IngestionPreferences & {
  rebuild?: boolean;
};

type IngestionSettingsFieldsProps = {
  value: IngestionSettingsValue;
  onChange: (value: IngestionSettingsValue) => void;
  showRebuild?: boolean;
  layout?: "row" | "stacked";
  className?: string;
};

function SettingRow({
  label,
  description,
  children,
  layout = "row",
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
  layout?: "row" | "stacked";
}) {
  if (layout === "stacked") {
    return (
      <div className="flex items-start justify-between gap-4 border-border/30 border-b py-4 last:border-0">
        <div className="min-w-0 flex-1">
          <p className="font-medium text-sm">{label}</p>
          {description ? (
            <p className="mt-0.5 text-muted-foreground text-xs">
              {description}
            </p>
          ) : null}
        </div>
        <div className="shrink-0 pt-0.5">{children}</div>
      </div>
    );
  }
  return (
    <div className="flex items-start justify-between gap-4 py-2">
      <div className="min-w-0 flex-1">
        <p className="font-medium text-sm">{label}</p>
        {description ? (
          <p className="text-muted-foreground text-xs">{description}</p>
        ) : null}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

export function IngestionSettingsFields({
  value,
  onChange,
  showRebuild = false,
  layout = "row",
  className,
}: IngestionSettingsFieldsProps) {
  const [ocrOptions, setOcrOptions] = useState<string[]>(["local", "llm"]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const config = await fetchIngestionConfig();
      if (!cancelled) {
        setOcrOptions(config.equation_ocr_options);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const patch = (partial: Partial<IngestionSettingsValue>) => {
    const next = { ...value, ...partial };
    if (partial.use_vision_model === false) {
      next.use_image_descriptions = false;
      next.use_formula_transcription = false;
    }
    onChange(next);
  };

  const visionDisabled = !value.use_vision_model;

  const rowLayout = layout;

  return (
    <div className={cn(className)}>
      <SettingRow
        layout={rowLayout}
        label="Use vision model"
        description="Enable vision features for images and formulas."
      >
        <Switch
          checked={value.use_vision_model}
          onCheckedChange={(checked) =>
            patch({ use_vision_model: Boolean(checked) })
          }
        />
      </SettingRow>
      <SettingRow
        layout={rowLayout}
        label="Image descriptions"
        description="Describe figures with the vision model."
      >
        <Switch
          checked={value.use_image_descriptions}
          disabled={visionDisabled}
          onCheckedChange={(checked) =>
            patch({ use_image_descriptions: Boolean(checked) })
          }
        />
      </SettingRow>
      <SettingRow
        layout={rowLayout}
        label="Formula transcription"
        description="Transcribe formula images to LaTeX."
      >
        <Switch
          checked={value.use_formula_transcription}
          disabled={visionDisabled}
          onCheckedChange={(checked) =>
            patch({ use_formula_transcription: Boolean(checked) })
          }
        />
      </SettingRow>
      {layout === "stacked" ? (
        <div className="grid gap-2 border-border/30 border-b py-4 last:border-0">
          <div>
            <p className="font-medium text-sm">Equation OCR</p>
            <p className="mt-0.5 text-muted-foreground text-xs">
              Local uses pix2tex; LLM uses Ollama vision.
            </p>
          </div>
          <Select
            value={value.equation_ocr_lib}
            onValueChange={(v) =>
              patch({
                equation_ocr_lib: v as IngestionPreferences["equation_ocr_lib"],
              })
            }
            disabled={visionDisabled || !value.use_formula_transcription}
          >
            <SelectTrigger className="w-full" size="sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ocrOptions.map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {opt}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      ) : (
        <SettingRow
          layout={rowLayout}
          label="Equation OCR"
          description="Local uses pix2tex; LLM uses Ollama vision."
        >
          <Select
            value={value.equation_ocr_lib}
            onValueChange={(v) =>
              patch({
                equation_ocr_lib: v as IngestionPreferences["equation_ocr_lib"],
              })
            }
            disabled={visionDisabled || !value.use_formula_transcription}
          >
            <SelectTrigger className="w-36" size="sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ocrOptions.map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {opt}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingRow>
      )}
      {showRebuild ? (
        <SettingRow
          layout={rowLayout}
          label="Rebuild index"
          description="Recreate the vector index before indexing this upload."
        >
          <Switch
            checked={Boolean(value.rebuild)}
            onCheckedChange={(checked) => patch({ rebuild: Boolean(checked) })}
          />
        </SettingRow>
      ) : null}
    </div>
  );
}
