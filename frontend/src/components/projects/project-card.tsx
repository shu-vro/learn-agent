"use client";

import { EllipsisVertical, FolderIcon } from "lucide-react";
import { CardDescription, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useLongPress } from "@/hooks/use-long-press";
import type { Project } from "@/lib/api/projects";
import { cn } from "@/lib/utils";

export function ProjectCard({
  project,
  onOpen,
  onContextOpen,
  onEdit,
  onDelete,
}: {
  project: Project;
  onOpen: (p: Project) => void;
  onContextOpen: (p: Project, x: number, y: number) => void;
  onEdit: (p: Project) => void;
  onDelete: (p: Project) => void;
}) {
  const longPress = useLongPress({
    onLongPress: (e) => {
      e.preventDefault();
      onContextOpen(project, e.clientX, e.clientY);
    },
  });

  return (
    <button
      type="button"
      className={cn(
        "group/card flex w-full flex-col gap-6 overflow-hidden rounded-2xl border border-border/50 bg-card/80 py-6 text-left text-card-foreground ring-1 ring-foreground/10 transition-colors hover:bg-card",
      )}
      onClick={() => onOpen(project)}
      onContextMenu={(e) => {
        e.preventDefault();
        onContextOpen(project, e.clientX, e.clientY);
      }}
      {...longPress}
    >
      <div className="grid auto-rows-min items-start gap-2 px-6">
        <div className="flex items-start gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-2xl bg-secondary text-muted-foreground">
            <FolderIcon className="size-5" />
          </div>
          <div className="min-w-0 space-y-1">
            <CardTitle className="text-base leading-snug">
              {project.name || "Untitled project"}
            </CardTitle>
            <CardDescription className="line-clamp-3">
              {project.description || "No description"}
            </CardDescription>
          </div>
          <div className="ml-auto mr-4 -mt-2 flex shrink-0">
            <DropdownMenu>
              <DropdownMenuTrigger>
                <button
                  type="button"
                  onClick={(e) => e.stopPropagation()}
                  className="rounded-full p-1 text-muted-foreground hover:text-foreground"
                  aria-label="Project menu"
                >
                  <EllipsisVertical className="size-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem
                  onClick={(e: React.MouseEvent) => {
                    e.stopPropagation();
                    onEdit(project);
                  }}
                >
                  Edit
                </DropdownMenuItem>
                <DropdownMenuItem
                  data-variant="destructive"
                  onClick={(e: React.MouseEvent) => {
                    e.stopPropagation();
                    onDelete(project);
                  }}
                >
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </button>
  );
}
