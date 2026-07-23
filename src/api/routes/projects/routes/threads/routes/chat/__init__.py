"""Chat SSE endpoint and thread chat history."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.agent.checkpointer import init_checkpointer
from src.agent.rag_core import (
    RagAppConfig,
    build_rag_agent,
    format_human_prompt,
    qdrant_filter_for_doc_ids,
    stream_rag_events,
    truncate_checkpoint_before_turn,
)
from src.db import get_session
from src.db.models.chat import Chat, ChatMessage, ThinkingMessage, ToolMessage
from src.db.models.document import Document
from src.db.models.project import Project
from src.db.models.project_document import ProjectDocument
from src.db.models.thread import Thread
from src.utils.api.BaseResponse import BaseResponse
from src.utils.usage_aggregator_callback import UsageAggregatorCallback

router = APIRouter(tags=["chats"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str | None = None
    thread_id: str | None = None
    message_id: str | None = None  # assistant ChatMessage id → regenerate
    reference_id: str | None = None  # ChatMessage id being quoted
    selection: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "ChatRequest":
        if self.message_id:
            return self
        if not (self.query and self.query.strip()):
            raise ValueError("query is required unless regenerating with message_id")
        if (self.reference_id is None) ^ (self.selection is None):
            raise ValueError("reference_id and selection must be provided together")
        if self.selection is not None and not self.selection.strip():
            raise ValueError("selection must not be empty")
        return self


class ThinkingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    thinking: str
    created_at: datetime


class ToolRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tool_name: str
    tool_parameters: dict[str, Any] | list[Any] | str | int | float | bool | None
    tool_result: dict[str, Any] | list[Any] | str | int | float | bool | None
    created_at: datetime


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chat_id: str
    message: str
    selection: str | None = None
    reference_id: str | None = None
    input_token: int = 0
    output_token: int = 0
    total_token: int = 0
    thinking_messages: list[ThinkingRead] = Field(default_factory=list)
    tool_messages: list[ToolRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ChatRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    thread_id: str
    type: str
    group_id: str | None = None
    messages: list[ChatMessageRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ThreadChatTurn(BaseModel):
    group_id: str
    user: ChatRead | None = None
    assistant: ChatRead | None = None


class ThreadBrief(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    thread_name: str
    created_at: datetime
    updated_at: datetime


ThreadChatsResponse = BaseResponse[list[ThreadChatTurn]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps({"event": event, **data}, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _chat_message_read(msg: ChatMessage) -> ChatMessageRead:
    """Build ChatMessageRead without triggering lazy relationship loads."""
    state = sa_inspect(msg)
    thinking: list[ThinkingRead] = []
    tools: list[ToolRead] = []
    if "thinking_messages" not in state.unloaded:
        thinking = [ThinkingRead.model_validate(t) for t in msg.thinking_messages]
    if "tool_messages" not in state.unloaded:
        tools = [ToolRead.model_validate(t) for t in msg.tool_messages]
    return ChatMessageRead(
        id=msg.id,
        chat_id=msg.chat_id,
        message=msg.message,
        selection=msg.selection,
        reference_id=msg.reference_id,
        input_token=msg.input_token or 0,
        output_token=msg.output_token or 0,
        total_token=msg.total_token or 0,
        thinking_messages=thinking,
        tool_messages=tools,
        created_at=msg.created_at,
        updated_at=msg.updated_at,
    )


async def _project_doc_sha256s(session: AsyncSession, project_id: str) -> list[str]:
    """Qdrant ``metadata.doc_id`` values for this project's uploaded assets.

    Ingestion stores content fingerprints (``documents.sha256``) as ``doc_id``
    in the vector store — not the documents.id UUID.
    """
    result = await session.execute(
        select(Document.sha256)
        .join(ProjectDocument, ProjectDocument.document_id == Document.id)
        .where(
            ProjectDocument.project_id == project_id,
            Document.sha256.is_not(None),
            Document.sha256 != "",
        )
    )
    return [row[0] for row in result.all() if row[0]]


async def _load_thread_chats(
    session: AsyncSession, thread_id: str
) -> list[ThreadChatTurn]:
    stmt = (
        select(Chat)
        .where(Chat.thread_id == thread_id)
        .options(
            selectinload(Chat.messages).selectinload(ChatMessage.thinking_messages),
            selectinload(Chat.messages).selectinload(ChatMessage.tool_messages),
        )
        .order_by(Chat.created_at.asc())
    )
    chats = list((await session.execute(stmt)).scalars().all())

    by_group: dict[str, dict[str, Chat]] = {}
    order: list[str] = []
    for chat in chats:
        gid = chat.group_id or chat.id
        if gid not in by_group:
            by_group[gid] = {}
            order.append(gid)
        by_group[gid][chat.type] = chat

    turns: list[ThreadChatTurn] = []
    for gid in order:
        pair = by_group[gid]
        user_chat = pair.get("user")
        assistant_chat = pair.get("assistant")
        turns.append(
            ThreadChatTurn(
                group_id=gid,
                user=ChatRead.model_validate(user_chat) if user_chat else None,
                assistant=(
                    ChatRead.model_validate(assistant_chat) if assistant_chat else None
                ),
            )
        )
    return turns


def _sum_usage(usage_aggregator: UsageAggregatorCallback | None) -> dict[str, int]:
    if not usage_aggregator:
        return {"input_token": 0, "output_token": 0, "total_token": 0}
    entries = usage_aggregator.get_aggregated_usage().get(
        usage_aggregator.task_name, []
    )
    inp = out = total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        inp += int(entry.get("input_tokens") or entry.get("prompt_tokens") or 0)
        out += int(entry.get("output_tokens") or entry.get("completion_tokens") or 0)
        total += int(entry.get("total_tokens") or (inp + out) or 0)
    if total == 0:
        total = inp + out
    return {"input_token": inp, "output_token": out, "total_token": total}


async def _count_user_turns_up_to_group(
    session: AsyncSession, thread_id: str, group_id: str
) -> int:
    """1-based index of this group's user turn among user chats in the thread."""
    stmt = (
        select(Chat)
        .where(Chat.thread_id == thread_id, Chat.type == "user")
        .order_by(Chat.created_at.asc())
    )
    user_chats = list((await session.execute(stmt)).scalars().all())
    for idx, chat in enumerate(user_chats, start=1):
        if chat.group_id == group_id:
            return idx
    return len(user_chats)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/{project_id}/threads/{thread_id}/chats")
async def list_thread_chats(
    request: Request,
    project_id: str,
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> ThreadChatsResponse:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    thread = await Thread.get_by_id_for_user(session, thread_id, user.id)
    if not thread or thread.project_id != project_id:
        raise HTTPException(status_code=404, detail="Thread not found")

    turns = await _load_thread_chats(session, thread_id)
    return ThreadChatsResponse.ok(data=turns)


@router.post("/{project_id}/chats")
async def chat_endpoint(
    request: Request,
    project_id: str,
    payload: ChatRequest,
    session: AsyncSession = Depends(get_session),
):
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Resolve thread / regeneration context before opening the SSE stream.
    thread: Thread | None = None
    thread_created = False
    regenerate = False
    query: str = ""
    selection: str | None = None
    reference_id: str | None = None
    user_chat: Chat | None = None
    user_message: ChatMessage | None = None
    assistant_chat: Chat | None = None
    assistant_message: ChatMessage | None = None
    keep_human_turns: int | None = None

    if payload.message_id:
        regenerate = True
        msg_stmt = (
            select(ChatMessage)
            .where(ChatMessage.id == payload.message_id)
            .options(selectinload(ChatMessage.chat))
        )
        target_msg = (await session.execute(msg_stmt)).scalar_one_or_none()
        if not target_msg:
            raise HTTPException(status_code=404, detail="Message not found")

        assistant_chat = target_msg.chat
        if assistant_chat.type != "assistant":
            raise HTTPException(
                status_code=400, detail="message_id must be an assistant message"
            )

        thread = await Thread.get_by_id_for_user(
            session, assistant_chat.thread_id, user.id
        )
        if not thread or thread.project_id != project_id:
            raise HTTPException(status_code=404, detail="Thread not found")

        group_id = assistant_chat.group_id
        if not group_id:
            raise HTTPException(status_code=400, detail="Chat group missing")

        user_chat_stmt = select(Chat).where(
            Chat.thread_id == thread.id,
            Chat.group_id == group_id,
            Chat.type == "user",
        )
        user_chat = (await session.execute(user_chat_stmt)).scalar_one_or_none()
        if not user_chat:
            raise HTTPException(status_code=400, detail="Paired user chat not found")

        user_msgs = (
            (
                await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.chat_id == user_chat.id)
                    .order_by(ChatMessage.created_at.asc())
                )
            )
            .scalars()
            .all()
        )
        if not user_msgs:
            raise HTTPException(status_code=400, detail="User message not found")
        user_message = user_msgs[0]
        query = user_message.message
        selection = user_message.selection
        reference_id = user_message.reference_id

        keep_human_turns = await _count_user_turns_up_to_group(
            session, thread.id, group_id
        )

        # Drop later turns so app history matches truncated checkpointer state.
        later_chats = (
            (
                await session.execute(
                    select(Chat).where(
                        Chat.thread_id == thread.id,
                        Chat.created_at > assistant_chat.created_at,
                    )
                )
            )
            .scalars()
            .all()
        )
        for later in later_chats:
            await session.delete(later)

        assistant_message = ChatMessage(chat_id=assistant_chat.id, message="")
        session.add(assistant_message)
        await session.commit()
        await session.refresh(assistant_message)
        await session.refresh(assistant_chat)
        await session.refresh(user_chat)
        await session.refresh(user_message)

    else:
        query = (payload.query or "").strip()
        selection = payload.selection
        reference_id = payload.reference_id

        if payload.thread_id:
            thread = await Thread.get_by_id_for_user(
                session, payload.thread_id, user.id
            )
            if not thread or thread.project_id != project_id:
                raise HTTPException(status_code=404, detail="Thread not found")
        else:
            thread = await Thread.create(session, project_id, user.id)
            thread_created = True

        group_id = str(uuid.uuid4())
        user_chat = Chat(
            thread_id=thread.id,
            user_id=user.id,
            type="user",
            group_id=group_id,
        )
        assistant_chat = Chat(
            thread_id=thread.id,
            user_id=user.id,
            type="assistant",
            group_id=group_id,
        )
        session.add(user_chat)
        session.add(assistant_chat)
        await session.flush()

        user_message = ChatMessage(
            chat_id=user_chat.id,
            message=query,
            selection=selection,
            reference_id=reference_id,
        )
        assistant_message = ChatMessage(chat_id=assistant_chat.id, message="")
        session.add(user_message)
        session.add(assistant_message)
        await session.commit()
        await session.refresh(user_chat)
        await session.refresh(assistant_chat)
        await session.refresh(user_message)
        await session.refresh(assistant_message)
        await session.refresh(thread)

    assert (
        thread and user_chat and user_message and assistant_chat and assistant_message
    )

    doc_sha256s = await _project_doc_sha256s(session, project_id)
    thread_id = thread.id
    thread_brief = ThreadBrief.model_validate(thread)
    user_msg_read = _chat_message_read(user_message)
    assistant_msg_id = assistant_message.id
    assistant_chat_id = assistant_chat.id
    user_chat_id = user_chat.id
    human_prompt = format_human_prompt(
        query, selection=selection, reference_id=reference_id
    )

    async def event_stream() -> AsyncIterator[str]:
        yield _sse(
            "thread",
            {
                "thread": thread_brief.model_dump(mode="json"),
                "created": thread_created,
            },
        )
        if not regenerate:
            yield _sse(
                "user_message",
                {
                    "chat_id": user_chat_id,
                    "message": user_msg_read.model_dump(mode="json"),
                },
            )
        yield _sse(
            "assistant_message",
            {
                "chat_id": assistant_chat_id,
                "message_id": assistant_msg_id,
                "regenerate": regenerate,
            },
        )

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        answer_holder: dict[str, Any] = {"text": "", "usage": {}}
        loop = asyncio.get_running_loop()

        def emit(item: str | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def run_agent() -> None:
            try:
                checkpointer = init_checkpointer()
                usage_aggregator = UsageAggregatorCallback("rag_agent_calls")
                agent, _ = build_rag_agent(
                    RagAppConfig(),
                    checkpointer=checkpointer,
                    qdrant_filter=qdrant_filter_for_doc_ids(doc_sha256s),
                    usage_aggregator=usage_aggregator,
                    summarization_aggregator=usage_aggregator,
                )

                if regenerate and keep_human_turns is not None:
                    truncate_checkpoint_before_turn(
                        agent,
                        thread_id=thread_id,
                        keep_human_turns=keep_human_turns,
                    )

                thinking_by_step: dict[int, str] = {}
                pending_tools: dict[str, dict[str, Any]] = {}
                completed_tools: list[dict[str, Any]] = []

                for event in stream_rag_events(
                    agent,
                    question=query,
                    thread_id=thread_id,
                    human_prompt=human_prompt,
                ):
                    if event.type == "thinking":
                        delta = event.data.get("delta", "")
                        step = int(event.data.get("step") or 0)
                        thinking_by_step[step] = thinking_by_step.get(step, "") + delta
                        emit(
                            _sse(
                                "thinking",
                                {
                                    "delta": delta,
                                    "step": step,
                                    "message_id": assistant_msg_id,
                                },
                            )
                        )
                    elif event.type == "token":
                        delta = event.data.get("delta", "")
                        answer_holder["text"] += delta
                        emit(
                            _sse(
                                "token",
                                {"delta": delta, "message_id": assistant_msg_id},
                            )
                        )
                    elif event.type == "tool":
                        phase = event.data.get("phase")
                        tool_id = event.data.get("id")
                        step = int(event.data.get("step") or 0)
                        if phase == "start" and tool_id:
                            pending_tools[tool_id] = {
                                "name": event.data.get("name"),
                                "args": event.data.get("args") or {},
                                "step": step,
                            }
                        emit(
                            _sse(
                                "tool",
                                {**event.data, "message_id": assistant_msg_id},
                            )
                        )
                        if phase == "result":
                            start = pending_tools.pop(tool_id, {}) if tool_id else {}
                            completed_tools.append(
                                {
                                    "name": event.data.get("name") or start.get("name"),
                                    "args": event.data.get("args")
                                    or start.get("args")
                                    or {},
                                    "result": event.data.get("result"),
                                    "step": event.data.get(
                                        "step", start.get("step", 0)
                                    ),
                                }
                            )
                    elif event.type == "summarizing":
                        emit(_sse("summarizing", {}))
                    elif event.type == "done":
                        answer_holder["text"] = event.data.get(
                            "answer", answer_holder["text"]
                        )
                        answer_holder["usage"] = _sum_usage(usage_aggregator)
                        answer_holder["thinking_by_step"] = thinking_by_step
                        answer_holder["tools"] = completed_tools

            except Exception as exc:  # noqa: BLE001
                emit(
                    _sse(
                        "error",
                        {"message": str(exc), "message_id": assistant_msg_id},
                    )
                )
            finally:
                emit(None)

        worker = loop.run_in_executor(None, run_agent)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        await worker

        # Persist final assistant message + thinking/tools
        try:
            from src.db import session_factory

            async with session_factory()() as persist_session:
                msg = await persist_session.get(ChatMessage, assistant_msg_id)
                if msg:
                    msg.message = answer_holder.get("text") or ""
                    usage = answer_holder.get("usage") or {}
                    msg.input_token = usage.get("input_token", 0)
                    msg.output_token = usage.get("output_token", 0)
                    msg.total_token = usage.get("total_token", 0)

                    thinking_by_step: dict[int, str] = (
                        answer_holder.get("thinking_by_step") or {}
                    )
                    tools: list[dict[str, Any]] = answer_holder.get("tools") or []
                    # Persist in loop order: thinking step N, then tools for step N.
                    step_keys = sorted(
                        {
                            *thinking_by_step.keys(),
                            *(int(t.get("step") or 0) for t in tools),
                        }
                    )
                    for step in step_keys:
                        text = thinking_by_step.get(step) or ""
                        if text:
                            persist_session.add(
                                ThinkingMessage(
                                    chat_message_id=msg.id,
                                    thinking=text,
                                )
                            )
                            await persist_session.flush()
                        for tool in tools:
                            if int(tool.get("step") or 0) != step:
                                continue
                            persist_session.add(
                                ToolMessage(
                                    chat_message_id=msg.id,
                                    tool_name=str(tool.get("name") or "unknown"),
                                    tool_parameters=tool.get("args") or {},
                                    tool_result={"result": tool.get("result")},
                                )
                            )
                            await persist_session.flush()

                    await persist_session.commit()
                    await persist_session.refresh(msg)

                    done_payload: dict[str, Any] = {
                        "message_id": msg.id,
                        "chat_id": assistant_chat_id,
                        "message": msg.message,
                        "usage": {
                            "input_token": msg.input_token,
                            "output_token": msg.output_token,
                            "total_token": msg.total_token,
                        },
                    }

                    # Name the thread once, on the first successful AI reply.
                    if (msg.message or "").strip() and not regenerate:
                        thread_row = await persist_session.get(Thread, thread_id)
                        if thread_row and not str(thread_row.thread_name or "").strip():
                            try:
                                from src.side_agents.update_thread_name_agent import (
                                    generate_thread_name,
                                )

                                new_name = await asyncio.to_thread(
                                    generate_thread_name,
                                    query=query,
                                    answer=msg.message,
                                )
                                thread_row.thread_name = new_name
                                await persist_session.commit()
                                await persist_session.refresh(thread_row)
                                done_payload["thread"] = ThreadBrief.model_validate(
                                    thread_row
                                ).model_dump(mode="json")
                                yield _sse(
                                    "thread",
                                    {
                                        "thread": done_payload["thread"],
                                        "created": False,
                                        "renamed": True,
                                    },
                                )
                            except Exception:  # noqa: BLE001
                                # Naming is best-effort; never fail the chat stream.
                                pass

                    yield _sse("done", done_payload)
                else:
                    yield _sse(
                        "done",
                        {
                            "message_id": assistant_msg_id,
                            "chat_id": assistant_chat_id,
                            "message": answer_holder.get("text") or "",
                            "usage": answer_holder.get("usage") or {},
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": f"Failed to persist: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
