export type ProjectSeed = {
  id: string;
  name: string;
  description: string;
  extra: Record<string, any>;
};

export type ThreadSeed = { id: string; title: string };

export type ChatMessageSeed = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type ArtifactSeed = {
  id: string;
  name: string;
  chunks: Record<string, string>;
  ingestion_status?: string;
  upload_progress?: number;
  ingestion_stage?: string;
  ingestion_stage_label?: string;
  ingestion_progress?: number;
};

export const SEED_PROJECTS: ProjectSeed[] = [
  {
    id: "p-demo-1",
    name: "Research notes",
    description: "Summaries and drafts for the Q1 literature review.",
    extra: {},
  },
  {
    id: "p-demo-2",
    name: "Side project",
    description: "Ideas and todos for the weekend build.",
    extra: {},
  },
];

export const SEED_THREADS: ThreadSeed[] = [
  { id: "t-1", title: "Getting started" },
  { id: "t-2", title: "API design questions" },
  { id: "t-3", title: "Markdown sample" },
];

export const SEED_MESSAGES_BY_THREAD: Record<string, ChatMessageSeed[]> = {
  "t-1": [
    {
      id: "m-1",
      role: "user",
      content: "What can you help me with in this workspace?",
    },
    {
      id: "m-2",
      role: "assistant",
      content:
        "I can help you **organize** threads, draft content, and explain uploaded files. Try the **Files** panel to preview markdown artifacts.",
    },
  ],
  "t-2": [
    {
      id: "m-3",
      role: "user",
      content: "How should I version my REST API?",
    },
    {
      id: "m-4",
      role: "assistant",
      content:
        "Common patterns:\n\n1. Path prefix: `/v1/...`\n2. Header: `Accept-Version`\n3. Subdomain\n\nPick one and stay consistent.",
    },
  ],
  "t-3": [
    {
      id: "m-5",
      role: "user",
      content: "Show me a tiny markdown demo.",
    },
    {
      id: "m-6",
      role: "assistant",
      content:
        "Here is **bold**, a list:\n\n- one\n- two\n\nAnd a block:\n\n```ts\nconst x = 1;\n```",
    },
  ],
};

export const SEED_ARTIFACTS: ArtifactSeed[] = [
  {
    id: "a-1",
    name: "README.md",
    chunks: {
      "seed-1":
        "# Sample artifact\n\nThis preview uses the same **Streamdown** renderer as chat messages.",
    },
    ingestion_status: "completed",
  },
];
