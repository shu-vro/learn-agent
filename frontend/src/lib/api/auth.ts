import type { UserProfile } from "@/lib/api/preferences";
import { get, post } from "@/utils/fetch";

export type UserPublic = {
  id: string;
  name: string;
  email: string;
};

export async function loginRequest(body: {
  email: string;
  password: string;
}): Promise<UserPublic> {
  const data = await post({
    endpoint: "/auth/login",
    params: body,
    throwable: true,
  });
  return data as UserPublic;
}

export async function registerRequest(body: {
  name: string;
  email: string;
  password: string;
}): Promise<UserPublic> {
  const data = await post({
    endpoint: "/auth/register",
    params: body,
    throwable: true,
  });
  return data as UserPublic;
}

export async function profileRequest(): Promise<UserProfile | null> {
  const data = await get({ endpoint: "/auth/profile" });
  if (
    data &&
    typeof data === "object" &&
    "email" in data &&
    "preferences" in data
  ) {
    return data as UserProfile;
  }
  return null;
}

export async function logoutRequest(): Promise<void> {
  await post({ endpoint: "/auth/logout", params: {} });
}
