import { nanoid } from "nanoid";
import { type ProjectSeed, SEED_PROJECTS } from "@/lib/dummy/seed";
import { del, get, post, put } from "@/utils/fetch";

export type Project = ProjectSeed;

function isProject(value: unknown): value is Project {
  return (
    typeof value === "object" &&
    value !== null &&
    "id" in value &&
    "name" in value &&
    "description" in value &&
    typeof (value as Project).id === "string"
  );
}

function isProjectList(value: unknown): value is Project[] {
  return Array.isArray(value) && value.every(isProject);
}

export async function listProjects(): Promise<Project[]> {
  const res = await get({ endpoint: "/projects" });
  console.log(res);
  if (isProjectList(res)) {
    return res;
  }
  return [...SEED_PROJECTS];
}

export async function createProjectRemote(
  partial?: Pick<Project, "name" | "description" | "extra">,
): Promise<Project> {
  const payload = {
    name: partial?.name ?? "",
    description: partial?.description ?? "",
    extra: partial?.extra ?? {},
  };
  const res = await post({ endpoint: "/projects", params: payload });
  if (isProject(res)) {
    return res;
  }
  return {
    id: nanoid(),
    name: payload.name,
    description: payload.description || "No description yet.",
    extra: payload.extra || {},
  };
}

export async function deleteProjectRemote(id: string): Promise<boolean> {
  const res = await del({ endpoint: `/projects/${id}` });
  if (res !== null && typeof res === "object") {
    return true;
  }
  return true;
}

export async function updateProjectRemote(
  id: string,
  patch: Pick<Project, "name" | "description" | "extra">,
): Promise<Project | null> {
  const res = await put({
    endpoint: `/projects/${id}`,
    params: patch,
  });
  if (isProject(res)) {
    return res;
  }
  return { id, ...patch };
}
