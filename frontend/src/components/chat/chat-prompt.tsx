"use client";

import {
  Attachment,
  AttachmentPreview,
  AttachmentRemove,
  Attachments,
} from "@/components/ai-elements/attachments";
import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";
import {
  PromptInput,
  PromptInputActionAddScreenshot,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputActionTakePhoto,
  PromptInputActionUploadFile,
  PromptInputActionUploadPhoto,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputProvider,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbarCameraPhoto,
  PromptInputToolbarScreenshot,
  PromptInputToolbarUploadFile,
  PromptInputToolbarUploadPhoto,
  PromptInputTools,
  usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
import type { FileUIPart } from "ai";
import { GlobeIcon } from "lucide-react";
import type { FormEvent } from "react";
import { memo, useCallback, useState } from "react";

const SUBMITTING_TIMEOUT = 200;
const STREAMING_TIMEOUT = 2000;

interface AttachmentItemProps {
  attachment: FileUIPart & { id: string };
  onRemove: (id: string) => void;
}

const AttachmentItem = memo(({ attachment, onRemove }: AttachmentItemProps) => {
  const handleRemove = useCallback(
    () => onRemove(attachment.id),
    [onRemove, attachment.id],
  );
  return (
    <Attachment data={attachment} key={attachment.id} onRemove={handleRemove}>
      <AttachmentPreview />
      <AttachmentRemove />
    </Attachment>
  );
});

AttachmentItem.displayName = "AttachmentItem";

const PromptInputAttachmentsDisplay = () => {
  const attachments = usePromptInputAttachments();

  const handleRemove = useCallback(
    (id: string) => attachments.remove(id),
    [attachments],
  );

  if (attachments.files.length === 0) {
    return null;
  }

  return (
    <Attachments className="px-3 pt-3 pb-2" variant="grid">
      {attachments.files.map((attachment) => (
        <AttachmentItem
          attachment={attachment}
          key={attachment.id}
          onRemove={handleRemove}
        />
      ))}
    </Attachments>
  );
};

export const ChatPrompt = ({
  onSubmit,
}: {
  onSubmit: (text: string, e: FormEvent<HTMLFormElement>) => void;
}) => {
  const [status, setStatus] = useState<
    "submitted" | "streaming" | "ready" | "error"
  >("ready");

  const handleSubmit = useCallback(
    (message: PromptInputMessage, event: FormEvent<HTMLFormElement>) => {
      const hasText = Boolean(message.text);
      const hasAttachments = Boolean(message.files?.length);

      if (!(hasText || hasAttachments)) {
        return;
      }

      setStatus("submitted");

      onSubmit(message.text, event);

      setTimeout(() => {
        setStatus("streaming");
      }, SUBMITTING_TIMEOUT);

      setTimeout(() => {
        setStatus("ready");
      }, STREAMING_TIMEOUT);
    },
    [onSubmit],
  );

  return (
    <div className="size-full">
      <PromptInputProvider>
        <PromptInput globalDrop multiple onSubmit={handleSubmit}>
          <PromptInputAttachmentsDisplay />
          <PromptInputBody>
            <PromptInputTextarea />
          </PromptInputBody>
          <PromptInputFooter>
            <PromptInputTools className="min-w-0 flex-wrap">
              <PromptInputActionMenu>
                <PromptInputActionMenuTrigger />
                <PromptInputActionMenuContent>
                  <PromptInputActionUploadFile />
                  <PromptInputActionUploadPhoto />
                  <PromptInputActionAddScreenshot />
                  <PromptInputActionTakePhoto />
                </PromptInputActionMenuContent>
              </PromptInputActionMenu>
              {/* <PromptInputToolbarUploadFile />
              <PromptInputToolbarUploadPhoto />
              <PromptInputToolbarScreenshot />
              <PromptInputToolbarCameraPhoto /> */}
              <PromptInputButton>
                <GlobeIcon size={16} />
                <span className="max-sm:sr-only">Search</span>
              </PromptInputButton>
            </PromptInputTools>
            <PromptInputSubmit status={status} />
          </PromptInputFooter>
        </PromptInput>
      </PromptInputProvider>
    </div>
  );
};
