"use client";

import type { FileUIPart } from "ai";
import { QuoteIcon, XIcon } from "lucide-react";
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
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
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
  disabled = false,
  reference = null,
  onClearReference,
}: {
  onSubmit: (
    text: string,
    e: FormEvent<HTMLFormElement>,
    images?: string[],
  ) => void;
  globalDrop?: boolean;
  disabled?: boolean;
  reference?: { id: string; text: string } | null;
  onClearReference?: () => void;
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
      if (disabled) {
        return;
      }
      const hasText = Boolean(message.text?.trim());
      const imageUrls = (message.files ?? [])
        .filter(
          (file) =>
            Boolean(file.url) &&
            (file.mediaType?.startsWith("image/") ||
              file.url.startsWith("data:image/")),
        )
        .map((file) => file.url);

      if (!(hasText || imageUrls.length > 0)) {
        return;
      }

      setStatus("submitted");

      onSubmit(message.text ?? "", event, imageUrls);

      setTimeout(() => {
        setStatus("streaming");
      }, SUBMITTING_TIMEOUT);

      setTimeout(() => {
        setStatus("ready");
      }, STREAMING_TIMEOUT);
    },
    [disabled, onSubmit],
  );

  const submitStatus = disabled ? "streaming" : status;

  return (
    <div className="size-full">
      <PromptInput globalDrop={globalDrop} multiple onSubmit={handleSubmit}>
        <PromptInputAttachmentsDisplay />
        {reference ? (
          <div className="flex items-start gap-2 border-border/50 border-b px-3 py-2 text-muted-foreground text-xs">
            <QuoteIcon className="mt-0.5 size-3 shrink-0" />
            <span className="line-clamp-2 min-w-0 flex-1">
              {reference.text}
            </span>
            <button
              type="button"
              aria-label="Remove reference"
              className="shrink-0 hover:text-foreground"
              onClick={onClearReference}
            >
              <XIcon className="size-3" />
            </button>
          </div>
        ) : null}
        <PromptInputBody>
          <PromptInputTextarea disabled={disabled} />
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
          <PromptInputSubmit status={submitStatus} disabled={disabled} />
        </PromptInputFooter>
      </PromptInput>
    </div>
  );
};
