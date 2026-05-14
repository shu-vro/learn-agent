import { ChatWorkspaceProvider } from "@/components/chat/chat-context";
import { ChatShell } from "@/components/chat/chat-shell";

type ChatPageProps = {
  searchParams: Promise<{ project?: string }>;
};

export default async function ChatPage({ searchParams }: ChatPageProps) {
  const { project } = await searchParams;
  return (
    <ChatWorkspaceProvider projectId={project ?? null}>
      <ChatShell />
    </ChatWorkspaceProvider>
  );
}
