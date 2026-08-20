"use client";

import { Loader2Icon, PauseIcon, PlayIcon, XIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";

/** DOM id used to scroll a message into view when read-aloud starts. */
export function messageAnchorId(messageId: string): string {
  return `message-${messageId}`;
}

function voiceStreamUrl(
  projectId: string,
  chatId: string,
  messageId: string,
): string {
  const baseUrl = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/+$/, "");
  const query = new URLSearchParams({
    chat_id: chatId,
    message_id: messageId,
  });
  return `${baseUrl}/api/v1/projects/${projectId}/chats/voice/?${query}`;
}

export type ReadAloud = {
  activeId: string | null;
  playing: boolean;
  loading: boolean;
  error: string | null;
  start: (messageId: string, chatId: string) => void;
  toggle: () => void;
  stop: () => void;
};

/**
 * Single audio player shared by every message in the conversation — starting a
 * new message stops whatever was playing before.
 */
export function useReadAloud(projectId: string | null): ReadAloud {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const release = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
    audioRef.current = null;
  }, []);

  const stop = useCallback(() => {
    release();
    setActiveId(null);
    setPlaying(false);
    setLoading(false);
    setError(null);
  }, [release]);

  useEffect(() => release, [release]);

  const start = useCallback(
    (messageId: string, chatId: string) => {
      if (!projectId || !chatId) {
        return;
      }
      release();

      const audio = new Audio();
      // Cookie auth: the stream endpoint is credentialed and cross-origin.
      audio.crossOrigin = "use-credentials";
      audio.src = voiceStreamUrl(projectId, chatId, messageId);
      audio.onplaying = () => {
        setLoading(false);
        setPlaying(true);
      };
      audio.onpause = () => setPlaying(false);
      audio.onended = () => {
        setPlaying(false);
        setLoading(false);
      };
      audio.onerror = () => {
        setLoading(false);
        setPlaying(false);
        setError("Could not play this message.");
      };
      audioRef.current = audio;

      setActiveId(messageId);
      setLoading(true);
      setError(null);
      void audio.play().catch(() => {
        setLoading(false);
        setError("Could not play this message.");
      });

      document
        .getElementById(messageAnchorId(messageId))
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [projectId, release],
  );

  const toggle = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    if (audio.paused) {
      void audio.play().catch(() => setError("Could not play this message."));
    } else {
      audio.pause();
    }
  }, []);

  return { activeId, playing, loading, error, start, toggle, stop };
}

/**
 * Play/pause bar for the message being read. Sticky inside the message, so it
 * floats above the prompt only while that message is on screen.
 */
export function ReadAloudControls({ reader }: { reader: ReadAloud }) {
  return (
    <div className="pointer-events-none sticky bottom-3 z-20 flex justify-center">
      <div className="pointer-events-auto flex items-center gap-1 rounded-full border border-border/60 bg-background/95 py-1 pr-1 pl-2 shadow-lg backdrop-blur">
        <Button
          type="button"
          size="icon-sm"
          variant="ghost"
          className="rounded-full"
          onClick={reader.toggle}
          disabled={reader.loading}
        >
          {reader.loading ? (
            <Loader2Icon className="size-3.5 animate-spin" />
          ) : reader.playing ? (
            <PauseIcon className="size-3.5" />
          ) : (
            <PlayIcon className="size-3.5" />
          )}
          <span className="sr-only">
            {reader.playing ? "Pause" : "Play"} read aloud
          </span>
        </Button>
        <span className="px-1 text-muted-foreground text-xs">
          {reader.error ?? (reader.loading ? "Loading…" : "Reading aloud")}
        </span>
        <Button
          type="button"
          size="icon-sm"
          variant="ghost"
          className="rounded-full"
          onClick={reader.stop}
        >
          <XIcon className="size-3.5" />
          <span className="sr-only">Stop read aloud</span>
        </Button>
      </div>
    </div>
  );
}
