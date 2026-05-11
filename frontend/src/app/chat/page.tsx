import { ChatWorkspaceProvider } from "@/components/chat/chat-context";
import { ChatShell } from "@/components/chat/chat-shell";

export default function ChatPage() {
  return (
    <ChatWorkspaceProvider>
      <ChatShell />
    </ChatWorkspaceProvider>
  );
}
