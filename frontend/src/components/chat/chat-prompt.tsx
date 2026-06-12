"use client";

import type { FileUIPart } from "ai";
import type { FormEvent } from "react";
import { memo, useCallback, useState } from "react";
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
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbarCameraPhoto,
  PromptInputToolbarScreenshot,
  PromptInputToolbarUploadFile,
  PromptInputToolbarUploadPhoto,
  PromptInputTools,
  usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";

import { useChatModel } from "@/components/chat/chat-model-context";
import { ModelPicker } from "@/components/chat/model-picker";

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
  globalDrop = true,
}: {
  onSubmit: (text: string, e: FormEvent<HTMLFormElement>) => void;
  globalDrop?: boolean;
}) => {
  const {
    models,
    reasoningEfforts,
    selectedModelId,
    reasoningEffort,
    setSelectedModelId,
    setReasoningEffort,
    loading: modelsLoading,
  } = useChatModel();
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
      <PromptInput globalDrop={globalDrop} multiple onSubmit={handleSubmit}>
        <PromptInputAttachmentsDisplay />
        <PromptInputBody>
          <PromptInputTextarea />
        </PromptInputBody>
        <PromptInputFooter className="items-start">
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
            {!modelsLoading && models.length > 0 ? (
              <ModelPicker
                variant="toolbar"
                models={models}
                reasoningEfforts={reasoningEfforts}
                selectedModelId={selectedModelId}
                reasoningEffort={reasoningEffort}
                onModelChange={setSelectedModelId}
                onReasoningEffortChange={setReasoningEffort}
              />
            ) : null}
          </PromptInputTools>
          <PromptInputSubmit status={status} />
        </PromptInputFooter>
      </PromptInput>
    </div>
  );
};
