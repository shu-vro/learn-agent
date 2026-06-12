"use client";

import {
  BotIcon,
  ChevronRightIcon,
  FileScanIcon,
  LockIcon,
  PaletteIcon,
  UserIcon,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { useAuth } from "@/components/auth/auth-provider";
import { ChatModelSettingsFields } from "@/components/settings/chat-model-settings-fields";
import { IngestionSettingsFields } from "@/components/settings/ingestion-settings-fields";
import { SettingsField } from "@/components/settings/settings-field";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchModelPresets } from "@/lib/api/models";
import type {
  ChatModelPreferences,
  IngestionPreferences,
  ThemeChoice,
} from "@/lib/api/preferences";
import {
  updatePassword,
  updatePreferences,
  updateProfile,
} from "@/lib/api/preferences";
import { formatRequestError } from "@/lib/api-error";
import { cn } from "@/lib/utils";

type SettingsSection =
  | "account"
  | "security"
  | "appearance"
  | "chat"
  | "ingestion";

const NAV: {
  id: SettingsSection;
  label: string;
  description: string;
  icon: typeof UserIcon;
  iconClass: string;
}[] = [
  {
    id: "account",
    label: "Account",
    description: "Name and email",
    icon: UserIcon,
    iconClass: "bg-rose-500/15 text-rose-500",
  },
  {
    id: "security",
    label: "Security",
    description: "Password",
    icon: LockIcon,
    iconClass: "bg-violet-500/15 text-violet-500",
  },
  {
    id: "appearance",
    label: "Appearance",
    description: "Theme",
    icon: PaletteIcon,
    iconClass: "bg-teal-500/15 text-teal-500",
  },
  {
    id: "chat",
    label: "Chat model",
    description: "Default model & reasoning",
    icon: BotIcon,
    iconClass: "bg-sky-500/15 text-sky-500",
  },
  {
    id: "ingestion",
    label: "Document ingestion",
    description: "PDF upload defaults",
    icon: FileScanIcon,
    iconClass: "bg-amber-500/15 text-amber-500",
  },
];

function SettingsNavItem({
  item,
  active,
  onSelect,
}: {
  item: (typeof NAV)[number];
  active: boolean;
  onSelect: () => void;
}) {
  const Icon = item.icon;
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "relative flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left transition-colors",
        active ? "bg-muted/80" : "hover:bg-muted/50",
      )}
    >
      <span
        className={cn(
          "flex size-9 shrink-0 items-center justify-center rounded-xl",
          item.iconClass,
        )}
      >
        <Icon className="size-4" aria-hidden />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block font-medium text-sm">{item.label}</span>
        <span className="block truncate text-muted-foreground text-xs">
          {item.description}
        </span>
      </span>
      <ChevronRightIcon
        className="size-4 shrink-0 text-muted-foreground"
        aria-hidden
      />
    </button>
  );
}

export function SettingsPage() {
  const router = useRouter();
  const { setTheme } = useTheme();
  const { user, loading, refreshProfile, setPreferences } = useAuth();

  const [activeSection, setActiveSection] =
    useState<SettingsSection>("account");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [theme, setThemeLocal] = useState<ThemeChoice>("system");
  const [ingestion, setIngestion] = useState<IngestionPreferences>({
    use_vision_model: true,
    use_image_descriptions: true,
    use_formula_transcription: true,
    equation_ocr_lib: "local",
  });
  const [chat, setChat] = useState<ChatModelPreferences>({
    default_model: "omlx:gemma-4-e4b-it-4bit",
    reasoning_effort: null,
  });
  const [modelPresets, setModelPresets] =
    useState<Awaited<ReturnType<typeof fetchModelPresets>>>(null);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/auth/login");
    }
  }, [loading, user, router]);

  useEffect(() => {
    if (user) {
      setName(user.name);
      setEmail(user.email);
      setThemeLocal(user.preferences.theme);
      setIngestion(user.preferences.ingestion);
      if (user.preferences.chat) {
        setChat(user.preferences.chat);
      }
    }
  }, [user]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const presets = await fetchModelPresets();
      if (!cancelled) {
        setModelPresets(presets);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading || !user) {
    return (
      <div className="flex min-h-dvh items-center justify-center text-muted-foreground text-sm">
        Loading…
      </div>
    );
  }

  async function saveActiveSection() {
    if (!user) {
      return;
    }
    setError(null);
    setMessage(null);
    setSaving(true);
    try {
      switch (activeSection) {
        case "account": {
          const updated = await updateProfile({
            name: name.trim() || user.name,
            email: email.trim() || user.email,
          });
          if (updated) {
            setPreferences(updated.preferences);
          }
          break;
        }
        case "security": {
          await updatePassword({
            current_password: currentPassword,
            new_password: newPassword,
          });
          setCurrentPassword("");
          setNewPassword("");
          break;
        }
        case "appearance": {
          const prefs = await updatePreferences({ theme });
          if (prefs) {
            setPreferences(prefs);
            setTheme(theme);
          }
          break;
        }
        case "chat": {
          const prefs = await updatePreferences({ chat });
          if (prefs) {
            setPreferences(prefs);
          }
          break;
        }
        case "ingestion": {
          const prefs = await updatePreferences({ ingestion });
          if (prefs) {
            setPreferences(prefs);
          }
          break;
        }
      }
      await refreshProfile();
      setMessage("Changes saved.");
    } catch (err) {
      setError(formatRequestError(err));
    } finally {
      setSaving(false);
    }
  }

  const activeNav = NAV.find((n) => n.id === activeSection) ?? NAV[0];

  return (
    <div className="flex min-h-dvh flex-col bg-background">
      <header className="shrink-0 border-border/40 border-b">
        <div className="mx-auto flex w-full max-w-6xl items-center gap-3 px-4 py-4 md:px-6">
          <Link
            href="/"
            className="text-muted-foreground text-sm transition-colors hover:text-foreground"
          >
            ← Projects
          </Link>
          <h1 className="font-semibold text-lg tracking-tight">Settings</h1>
        </div>
      </header>

      <div className="mx-auto flex w-full max-w-6xl min-h-0 flex-1 flex-col gap-4 px-4 py-4 md:flex-row md:gap-6 md:px-6 md:py-6">
        <aside className="shrink-0 md:w-72">
          <nav className="flex gap-2 overflow-x-auto pb-1 md:flex-col md:overflow-visible md:pb-0">
            {NAV.map((item) => (
              <SettingsNavItem
                key={item.id}
                item={item}
                active={activeSection === item.id}
                onSelect={() => {
                  setActiveSection(item.id);
                  setMessage(null);
                  setError(null);
                }}
              />
            ))}
          </nav>
        </aside>

        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="mb-3 md:hidden">
            <p className="font-medium text-sm">{activeNav.label}</p>
            <p className="text-muted-foreground text-xs">
              {activeNav.description}
            </p>
          </div>

          {(error || message) && (
            <div className="mb-4 shrink-0">
              {error ? (
                <p className="rounded-xl bg-destructive/10 px-3 py-2 text-destructive text-sm">
                  {error}
                </p>
              ) : null}
              {message ? (
                <p className="rounded-xl bg-primary/10 px-3 py-2 text-primary text-sm">
                  {message}
                </p>
              ) : null}
            </div>
          )}

          <div className="min-h-0 flex-1 overflow-y-auto pb-24">
            <Accordion
              multiple
              defaultValue={["panel"]}
              className="border-border/60 bg-card/40"
            >
              <AccordionItem value="panel">
                <AccordionTrigger className="px-5 py-4 hover:no-underline">
                  <span className="font-medium text-base">
                    {activeNav.label}
                  </span>
                </AccordionTrigger>
                <AccordionContent className="px-5">
                  {activeSection === "account" && (
                    <div className="grid gap-5 pb-2">
                      <SettingsField label="Full name">
                        <Input
                          value={name}
                          onChange={(e) => setName(e.target.value)}
                          className="w-full"
                        />
                      </SettingsField>
                      <SettingsField
                        label="Email"
                        hint="Used to sign in to your account."
                      >
                        <Input
                          type="email"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          className="w-full"
                        />
                      </SettingsField>
                    </div>
                  )}

                  {activeSection === "security" && (
                    <div className="grid gap-5 pb-2">
                      <SettingsField label="Current password">
                        <Input
                          type="password"
                          autoComplete="current-password"
                          value={currentPassword}
                          onChange={(e) => setCurrentPassword(e.target.value)}
                          className="w-full"
                        />
                      </SettingsField>
                      <SettingsField
                        label="New password"
                        hint="At least 8 characters."
                      >
                        <Input
                          type="password"
                          autoComplete="new-password"
                          value={newPassword}
                          onChange={(e) => setNewPassword(e.target.value)}
                          className="w-full"
                        />
                      </SettingsField>
                    </div>
                  )}

                  {activeSection === "appearance" && (
                    <div className="grid gap-5 pb-2">
                      <SettingsField
                        label="Theme"
                        hint="Choose how Learn Agent looks on this device."
                      >
                        <Select
                          value={theme}
                          onValueChange={(v) => setThemeLocal(v as ThemeChoice)}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="system">System</SelectItem>
                            <SelectItem value="light">Light</SelectItem>
                            <SelectItem value="dark">Dark</SelectItem>
                          </SelectContent>
                        </Select>
                      </SettingsField>
                    </div>
                  )}

                  {activeSection === "chat" && (
                    <div className="grid gap-5 pb-2">
                      <SettingsField
                        label="Default model"
                        hint="Used when you open chat. You can still switch models per session."
                      >
                        {modelPresets ? (
                          <ChatModelSettingsFields
                            value={chat}
                            onChange={setChat}
                            models={modelPresets.models}
                            reasoningEfforts={modelPresets.reasoning_efforts}
                          />
                        ) : (
                          <p className="text-muted-foreground text-sm">
                            Loading models…
                          </p>
                        )}
                      </SettingsField>
                    </div>
                  )}

                  {activeSection === "ingestion" && (
                    <div className="pb-2">
                      <p className="mb-2 text-muted-foreground text-xs">
                        Defaults for PDF uploads. Override per upload from the
                        files panel gear icon.
                      </p>
                      <IngestionSettingsFields
                        layout="stacked"
                        value={ingestion}
                        onChange={setIngestion}
                      />
                    </div>
                  )}
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>

          <div className="sticky bottom-0 -mx-4 border-border/40 border-t bg-background/95 px-4 py-4 backdrop-blur-sm md:-mx-0 md:px-0">
            <Button
              type="button"
              className="h-11 w-full rounded-2xl font-medium"
              disabled={saving}
              onClick={() => void saveActiveSection()}
            >
              {saving ? "Saving…" : "Save changes"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
