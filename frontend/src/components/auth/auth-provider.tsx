"use client";

import { useTheme } from "next-themes";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import { logoutRequest, profileRequest } from "@/lib/api/auth";
import type {
  IngestionPreferences,
  ThemeChoice,
  UserPreferences,
  UserProfile,
} from "@/lib/api/preferences";

type AuthContextValue = {
  user: UserProfile | null;
  loading: boolean;
  preferences: UserPreferences | null;
  refreshProfile: () => Promise<void>;
  setPreferences: (prefs: UserPreferences) => void;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

function applyTheme(theme: ThemeChoice, setTheme: (t: string) => void) {
  setTheme(theme);
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const { setTheme } = useTheme();
  const [user, setUser] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshProfile = useCallback(async () => {
    const profile = await profileRequest();
    setUser(profile);
    if (profile?.preferences.theme) {
      applyTheme(profile.preferences.theme, setTheme);
    }
  }, [setTheme]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const profile = await profileRequest();
        if (!cancelled) {
          setUser(profile);
          if (profile?.preferences.theme) {
            applyTheme(profile.preferences.theme, setTheme);
          }
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setTheme]);

  const setPreferences = useCallback(
    (prefs: UserPreferences) => {
      setUser((prev) => (prev ? { ...prev, preferences: prefs } : prev));
      applyTheme(prefs.theme, setTheme);
    },
    [setTheme],
  );

  const logout = useCallback(async () => {
    await logoutRequest();
    setUser(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      preferences: user?.preferences ?? null,
      refreshProfile,
      setPreferences,
      logout,
    }),
    [user, loading, refreshProfile, setPreferences, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}

export function defaultIngestionPreferences(): IngestionPreferences {
  return {
    use_vision_model: true,
    use_image_descriptions: true,
    use_formula_transcription: true,
    equation_ocr_lib: "local",
  };
}
