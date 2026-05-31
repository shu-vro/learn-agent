import axios from "axios";

type RequestOptions = {
  endpoint: string;
  params?: unknown;
  token?: string;
  full?: boolean;
  throwable?: boolean;
  version?: string;
  baseUrl?: string;
  overrideEncryptedResponsesOnly?: boolean;
};

function isSuccessPayload(data: unknown): boolean {
  if (data === null || typeof data !== "object") return false;
  const d = data as Record<string, unknown>;
  if (d.status === "success") return true;
  if (d.success === true) return true;
  return false;
}

function extractData<T = unknown>(data: unknown): T | null {
  if (data === null || typeof data !== "object") return null;
  const d = data as Record<string, unknown>;
  if ("data" in d) return d.data as T;
  return null;
}

const request = async (
  method: "get" | "put" | "post" | "patch" | "delete" = "get",
  {
    endpoint = "",
    params = {},
    token = "",
    full = false,
    throwable = false,
    version = "v1",
    baseUrl = process.env.NEXT_PUBLIC_API_URL ?? "",
    overrideEncryptedResponsesOnly: _unusedOverride = false,
  }: RequestOptions,
) => {
  const path = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  const root = baseUrl.replace(/\/+$/, "");
  const url = `${root}/api/${version}${path}`;
  let response = null;
  try {
    const headers: Record<string, string> = {};
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const requestPayload = method !== "get" ? params : undefined;
    if (!(requestPayload instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }

    response = await axios({
      method,
      headers,
      url,
      data: requestPayload,
      params: method === "get" ? params : undefined,
      withCredentials: true,
    });

    const body = response.data;

    if (isSuccessPayload(body)) {
      const payload = extractData(body);
      if (full) return response;
      return payload;
    }

    if (throwable) {
      throw new Error(
        (body as { message?: string })?.message || "API request failed",
      );
    }
    return null;
  } catch (error: unknown) {
    const err = error as {
      message?: string;
      response?: {
        status?: number;
        data?: unknown;
        headers?: Record<string, string>;
      };
    };
    if (throwable) {
      throw error;
    }
    // Avoid logging full Axios stacks on every 404 — Next dev forwards browser
    // console output and repeated errors contribute to runaway memory use.
    if (process.env.NODE_ENV === "development") {
      const status = err.response?.status;
      if (status !== 404) {
        console.warn(`API ${method.toUpperCase()} ${url} failed:`, err.message);
      }
    }
    return err?.response?.data ?? null;
  }
};

export const get = async ({
  endpoint = "",
  params = {},
  token = "",
  full = false,
  throwable = false,
  version = "v1",
  baseUrl,
  overrideEncryptedResponsesOnly = false,
}: Partial<RequestOptions>) => {
  return await request("get", {
    endpoint,
    params,
    token,
    full,
    throwable,
    version,
    baseUrl,
    overrideEncryptedResponsesOnly,
  });
};

export const post = async ({
  endpoint = "",
  params = {},
  token = "",
  full = false,
  throwable = false,
  version = "v1",
  baseUrl,
  overrideEncryptedResponsesOnly = false,
}: Partial<RequestOptions>) => {
  return await request("post", {
    endpoint,
    params,
    token,
    full,
    throwable,
    version,
    baseUrl,
    overrideEncryptedResponsesOnly,
  });
};

export const patch = async ({
  endpoint = "",
  params = {},
  token = "",
  full = false,
  throwable = false,
  version = "v1",
  baseUrl,
  overrideEncryptedResponsesOnly = false,
}: Partial<RequestOptions>) => {
  return await request("patch", {
    endpoint,
    params,
    token,
    full,
    throwable,
    version,
    baseUrl,
    overrideEncryptedResponsesOnly,
  });
};

export const put = async ({
  endpoint = "",
  params = {},
  token = "",
  full = false,
  throwable = false,
  version = "v1",
  baseUrl,
  overrideEncryptedResponsesOnly = false,
}: Partial<RequestOptions>) => {
  return await request("put", {
    endpoint,
    params,
    token,
    full,
    throwable,
    version,
    baseUrl,
    overrideEncryptedResponsesOnly,
  });
};

export const del = async ({
  endpoint = "",
  params = {},
  token = "",
  full = false,
  throwable = false,
  version = "v1",
  baseUrl,
  overrideEncryptedResponsesOnly = false,
}: Partial<RequestOptions>) => {
  return await request("delete", {
    endpoint,
    params,
    token,
    full,
    throwable,
    version,
    baseUrl,
    overrideEncryptedResponsesOnly,
  });
};
