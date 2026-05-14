"use client";

import { PencilIcon, PlusIcon, Trash2Icon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ProjectCard } from "@/components/projects/project-card";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  createProjectRemote,
  deleteProjectRemote,
  listProjects,
  type Project,
  updateProjectRemote,
} from "@/lib/api/projects";
import { cn } from "@/lib/utils";

type MenuState =
  | { open: false }
  | { open: true; x: number; y: number; project: Project };

export function ProjectsHome() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [menu, setMenu] = useState<MenuState>({ open: false });
  const [editOpen, setEditOpen] = useState(false);
  const [editProject, setEditProject] = useState<Project | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editDescription, setEditDescription] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const list = await listProjects();
      if (!cancelled) {
        setProjects(list);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const openMenuAt = useCallback(
    (project: Project, clientX: number, clientY: number) => {
      setMenu({ open: true, x: clientX, y: clientY, project });
    },
    [],
  );

  const closeMenu = useCallback(() => setMenu({ open: false }), []);

  const openProject = useCallback(
    (p: Project) => {
      closeMenu();
      router.push(`/chat?project=${encodeURIComponent(p.id)}`);
    },
    [router, closeMenu],
  );

  const handleNewProject = useCallback(async () => {
    const p = await createProjectRemote();
    setProjects((prev) => [...prev, p]);
    router.push(`/chat?project=${encodeURIComponent(p.id)}`);
  }, [router]);

  const handleDelete = useCallback(
    async (p: Project) => {
      closeMenu();
      if (!window.confirm(`Delete “${p.name}”?`)) {
        return;
      }
      await deleteProjectRemote(p.id);
      setProjects((prev) => prev.filter((x) => x.id !== p.id));
    },
    [closeMenu],
  );

  const startEdit = useCallback(
    (p: Project) => {
      closeMenu();
      setEditProject(p);
      setEditTitle(p.name);
      setEditDescription(p.description);
      setEditOpen(true);
    },
    [closeMenu],
  );

  const saveEdit = useCallback(async () => {
    if (!editProject) {
      return;
    }
    const updated = await updateProjectRemote(editProject.id, {
      name: editTitle.trim() || editProject.name,
      description: editDescription,
      extra: editProject.extra,
    });
    if (updated) {
      setProjects((prev) =>
        prev.map((x) => (x.id === updated.id ? updated : x)),
      );
    }
    setEditOpen(false);
    setEditProject(null);
  }, [editProject, editTitle, editDescription]);

  return (
    <div className="flex min-h-dvh flex-col bg-background">
      <header className="border-border/40 border-b">
        <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-4 py-4 md:px-8">
          <div>
            <h1 className="font-semibold text-lg tracking-tight">
              Learn Agent
            </h1>
            <p className="text-muted-foreground text-sm">Your projects</p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              nativeButton={false}
              render={<Link href="/auth/login" />}>
              Log in
            </Button>
            <Button
              size="sm"
              nativeButton={false}
              render={<Link href="/auth/register" />}>
              Register
            </Button>
            <Button
              type="button"
              size="icon"
              className="rounded-full"
              onClick={handleNewProject}>
              <PlusIcon className="size-5" />
              <span className="sr-only">New project</span>
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 md:px-8">
        <div className="grid gap-4 sm:grid-cols-2">
          {projects.map((p) => (
            <ProjectCard
              key={p.id}
              project={p}
              onOpen={openProject}
              onContextOpen={(proj, x, y) => openMenuAt(proj, x, y)}
            />
          ))}
        </div>
      </main>

      {menu.open && (
        <>
          <button
            type="button"
            aria-label="Close menu"
            className="fixed inset-0 z-40 cursor-default bg-transparent"
            onClick={closeMenu}
          />
          <div
            role="menu"
            className="fixed z-50 min-w-44 rounded-2xl bg-popover p-1 text-popover-foreground shadow-2xl ring-1 ring-foreground/5"
            style={{
              left: (() => {
                const vw =
                  typeof window !== "undefined" ? window.innerWidth : 400;
                return Math.min(Math.max(8, menu.x), Math.max(8, vw - 188));
              })(),
              top: (() => {
                const vh =
                  typeof window !== "undefined" ? window.innerHeight : 400;
                return Math.min(Math.max(8, menu.y), Math.max(8, vh - 140));
              })(),
            }}>
            <MenuRow onClick={() => openProject(menu.project)}>Open</MenuRow>
            <MenuRow onClick={() => startEdit(menu.project)}>
              <PencilIcon className="size-4 opacity-70" />
              Edit
            </MenuRow>
            <MenuRow
              variant="destructive"
              onClick={() => void handleDelete(menu.project)}>
              <Trash2Icon className="size-4 opacity-70" />
              Delete
            </MenuRow>
          </div>
        </>
      )}

      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent showCloseButton>
          <DialogHeader>
            <DialogTitle>Edit project</DialogTitle>
          </DialogHeader>
          <div className="grid gap-3">
            <div className="grid gap-1.5">
              <span className="text-muted-foreground text-xs">Title</span>
              <Input
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
              />
            </div>
            <div className="grid gap-1.5">
              <span className="text-muted-foreground text-xs">Description</span>
              <Input
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter className="gap-2 sm:justify-end">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setEditOpen(false)}>
              Cancel
            </Button>
            <Button type="button" onClick={() => void saveEdit()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function MenuRow({
  children,
  onClick,
  variant = "default",
}: {
  children: React.ReactNode;
  onClick: () => void;
  variant?: "default" | "destructive";
}) {
  return (
    <button
      type="button"
      role="menuitem"
      className={cn(
        "flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm transition-colors",
        variant === "destructive"
          ? "text-destructive hover:bg-destructive/10"
          : "hover:bg-accent hover:text-accent-foreground",
      )}
      onClick={() => {
        onClick();
      }}>
      {children}
    </button>
  );
}
