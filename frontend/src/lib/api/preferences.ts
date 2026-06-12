import type { ReasoningEffort } from "@/lib/api/models";
import { get, patch } from "@/utils/fetch";

export type ThemeChoice = "system" | "light" | "dark";
export type EquationOcrLib = "local" | "llm";

export type ChatModelPreferences = {
  default_model: string;
  reasoning_effort: ReasoningEffort;
};

export type IngestionPreferences = {
  use_vision_model: boolean;
  use_image_descriptions: boolean;
  use_formula_transcription: boolean;
  equation_ocr_lib: EquationOcrLib;
};

export type UserPreferences = {
  theme: ThemeChoice;
  ingestion: IngestionPreferences;
  chat: ChatModelPreferences;
};

export type UserProfile = {
  id: string;
  name: string;
  email: string;
  preferences: UserPreferences;
};

export type IngestionUploadOptions = IngestionPreferences & {
  rebuild?: boolean;
};

export type IngestionConfig = {
  equation_ocr_options: string[];
  default_equation_ocr_lib: string;
};

export async function fetchIngestionConfig(): Promise<IngestionConfig> {
  const data = await get({ endpoint: "/config/ingestion" });
  if (data && typeof data === "object" && "equation_ocr_options" in data) {
    return data as IngestionConfig;
  }
  return {
    equation_ocr_options: ["local", "llm"],
    default_equation_ocr_lib: "local",
  };
}

export async function updatePreferences(
  body: Partial<UserPreferences> & {
    ingestion?: Partial<IngestionPreferences>;
    chat?: Partial<ChatModelPreferences>;
  },
): Promise<UserPreferences | null> {
  const data = await patch({
    endpoint: "/auth/preferences",
    params: body,
    throwable: true,
  });
  if (data && typeof data === "object" && "theme" in data) {
    return data as UserPreferences;
  }
  return null;
}

export async function updateProfile(body: {
  name?: string;
  email?: string;
}): Promise<UserProfile | null> {
  const data = await patch({
    endpoint: "/auth/profile",
    params: body,
    throwable: true,
  });
  if (data && typeof data === "object" && "email" in data) {
    return data as UserProfile;
  }
  return null;
}

export async function updatePassword(body: {
  current_password: string;
  new_password: string;
}): Promise<void> {
  await patch({
    endpoint: "/auth/password",
    params: body,
    throwable: true,
  });
}

export function appendIngestionToFormData(
  form: FormData,
  options: IngestionUploadOptions,
): void {
  form.append("use_vision_model", String(options.use_vision_model));
  form.append("use_image_descriptions", String(options.use_image_descriptions));
  form.append(
    "use_formula_transcription",
    String(options.use_formula_transcription),
  );
  form.append("equation_ocr_lib", options.equation_ocr_lib);
  if (options.rebuild) {
    form.append("rebuild", "true");
  }
}
