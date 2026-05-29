"use client";

import Link from "next/link";

import { ArtifactsPanel } from "@/components/chat/artifacts-panel";
import { ChatMain } from "@/components/chat/chat-main";
import { ThreadsPanel } from "@/components/chat/threads-panel";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function ChatShell({ projectId = null }: { projectId?: string | null }) {
  return (
    <div className="flex h-dvh min-h-0 flex-col bg-background">
      <header className="shrink-0 border-border/40 border-b">
        <div className="mx-auto flex w-full max-w-[1600px] items-center gap-3 px-4 py-3 md:px-6">
          <Link
            href="/"
            className="text-muted-foreground text-sm transition-colors hover:text-foreground"
          >
            ← Projects
          </Link>
          <span className="font-medium text-sm">Learn Agent</span>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col">
        <div className="hidden min-h-0 flex-1 flex-col md:flex">
          <ResizablePanelGroup
            id="learn-agent-chat-layout"
            orientation="horizontal"
            defaultLayout={{ threads: 22, chat: 35, artifacts: 43 }}
            resizeTargetMinimumSize={{ fine: 6, coarse: 10 }}
          >
            <ResizablePanel
              id="threads"
              minSize="16%"
              maxSize="40%"
              className="min-h-0 min-w-0"
            >
              <ThreadsPanel className="h-full min-h-0" />
            </ResizablePanel>
            <ResizableHandle />
            <ResizablePanel
              id="chat"
              minSize="32%"
              maxSize="72%"
              className="min-h-0 min-w-0"
            >
              <ChatMain className="h-full min-h-0" />
            </ResizablePanel>
            <ResizableHandle />
            <ResizablePanel
              id="artifacts"
              minSize="16%"
              maxSize="40%"
              className="min-h-0 min-w-0"
            >
              <ArtifactsPanel
                className="h-full min-h-0"
                projectId={projectId}
              />
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>

        <div className="flex min-h-0 flex-1 flex-col md:hidden">
          <Tabs
            defaultValue="chat"
            className="flex min-h-0 flex-1 flex-col px-2 pt-2"
          >
            <TabsList className="mb-2 grid w-full shrink-0 grid-cols-3 rounded-2xl">
              <TabsTrigger value="threads">Threads</TabsTrigger>
              <TabsTrigger value="chat">Chat</TabsTrigger>
              <TabsTrigger value="files">Files</TabsTrigger>
            </TabsList>
            <TabsContent
              value="threads"
              className="min-h-0 flex-1 overflow-hidden"
            >
              <ThreadsPanel className="h-full rounded-xl border border-border/40" />
            </TabsContent>
            <TabsContent
              value="chat"
              className="min-h-0 flex-1 overflow-hidden"
            >
              <ChatMain className="h-full rounded-xl border border-border/40" />
            </TabsContent>
            <TabsContent
              value="files"
              className="min-h-0 flex-1 overflow-hidden"
            >
              <ArtifactsPanel
                className="h-full rounded-xl border border-border/40"
                projectId={projectId}
              />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
