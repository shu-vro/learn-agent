"use client";

import { SettingsIcon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { PromptInputProvider } from "@/components/ai-elements/prompt-input";
import { ArtifactsPanel } from "@/components/chat/artifacts-panel";
import { ChatMain } from "@/components/chat/chat-main";
import { ThreadsPanel } from "@/components/chat/threads-panel";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useMediaQuery } from "@/hooks/use-media-query";
import {
  deleteProjectRemote,
  listProjects,
  type Project,
  updateProjectRemote,
} from "@/lib/api/projects";

type MobileTab = "threads" | "chat" | "files";

export function ChatShell({ projectId = null }: { projectId?: string | null }) {
  const [project, setProject] = useState<Project | null>(null);
  const [editOpen, setEditOpen] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteConfirmationValue, setDeleteConfirmationValue] = useState("");
  const [mobileTab, setMobileTab] = useState<MobileTab>("chat");
  const isMobile = useMediaQuery("(max-width: 767px)");
  const router = useRouter();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!projectId) return;
      const all = await listProjects();
      if (cancelled) return;
      const p = all.find((x) => x.id === projectId) ?? null;
      setProject(p);
      setEditTitle(p?.name ?? "");
      setEditDescription(p?.description ?? "");
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  return (
    <PromptInputProvider>
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
            <div className="ml-auto">
              {projectId ? (
                <DropdownMenu>
                  <DropdownMenuTrigger>
                    <button
                      type="button"
                      className="rounded-xl p-1 text-muted-foreground hover:text-foreground"
                    >
                      <SettingsIcon className="size-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => setEditOpen(true)}>
                      Update
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      data-variant="destructive"
                      onClick={() => setDeleteOpen(true)}
                    >
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              ) : null}
            </div>
          </div>
        </header>

        <div className="flex min-h-0 flex-1 flex-col">
          {isMobile ? (
            <div className="flex min-h-0 flex-1 flex-col px-2 pt-2">
              <Tabs
                value={mobileTab}
                onValueChange={(value) => setMobileTab(value as MobileTab)}
                className="flex min-h-0 flex-1 flex-col"
              >
                <TabsList className="mb-2 grid w-full shrink-0 grid-cols-3 rounded-2xl">
                  <TabsTrigger value="threads">Threads</TabsTrigger>
                  <TabsTrigger value="chat">Chat</TabsTrigger>
                  <TabsTrigger value="files">Files</TabsTrigger>
                </TabsList>
                <div className="min-h-0 flex-1 overflow-hidden">
                  {mobileTab === "threads" ? (
                    <ThreadsPanel className="h-full rounded-xl border border-border/40" />
                  ) : null}
                  {mobileTab === "chat" ? (
                    <ChatMain
                      className="h-full rounded-xl border border-border/40"
                      promptGlobalDrop
                    />
                  ) : null}
                  {mobileTab === "files" ? (
                    <ArtifactsPanel
                      className="h-full rounded-xl border border-border/40"
                      projectId={projectId}
                    />
                  ) : null}
                </div>
              </Tabs>
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col">
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
                  <ChatMain className="h-full min-h-0" promptGlobalDrop />
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
          )}
        </div>

        <Dialog open={editOpen} onOpenChange={setEditOpen}>
          <DialogContent showCloseButton>
            <DialogHeader>
              <DialogTitle>Update project</DialogTitle>
            </DialogHeader>
            <div className="grid gap-3">
              <Input
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
              />
              <Input
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
              />
            </div>
            <DialogFooter>
              <Button variant="ghost" onClick={() => setEditOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={async () => {
                  if (!project) return;
                  const updated = await updateProjectRemote(project.id, {
                    name: editTitle,
                    description: editDescription,
                    extra: project.extra,
                  });
                  if (updated) {
                    setProject(updated as Project);
                  }
                  setEditOpen(false);
                }}
              >
                Save
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
          <DialogContent showCloseButton>
            <DialogHeader>
              <DialogTitle>Delete project</DialogTitle>
            </DialogHeader>
            <div className="grid gap-3">
              <p className="text-sm text-muted-foreground">
                Type DELETE (capital letters) to confirm deletion.
              </p>
              <Input
                value={deleteConfirmationValue}
                onChange={(e) => setDeleteConfirmationValue(e.target.value)}
              />
            </div>
            <DialogFooter>
              <Button variant="ghost" onClick={() => setDeleteOpen(false)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={async () => {
                  if (!projectId) return;
                  if (deleteConfirmationValue !== "DELETE") return;
                  await deleteProjectRemote(projectId);
                  router.push("/");
                }}
              >
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </PromptInputProvider>
  );
}
