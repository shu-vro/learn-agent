from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from src.config.model_config import model_selector
from src.config.constants import DEFAULT_PROJECT_NAME_AND_DESCRIPTION_MODEL
from src.utils.usage_aggregator_callback import UsageAggregatorCallback
from src.utils.side_agents import extract_json_payload

PROMPT_PATH = Path(__file__).with_name("prompt.md")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()


def _document_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = json.dumps(
            document.metadata,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        content = document.page_content.strip()
        blocks.append(
            "\n".join(
                [
                    f"[Document {index}]",
                    f"metadata: {metadata}",
                    "content:",
                    content if content else "[empty]",
                ]
            )
        )
    return "\n\n".join(blocks)


def generate_project_name_and_description(documents: list[Document]) -> dict[str, str]:
    usage_aggregator: UsageAggregatorCallback = UsageAggregatorCallback(
        "project_name_update_agent_usage"
    )

    llm = model_selector(
        DEFAULT_PROJECT_NAME_AND_DESCRIPTION_MODEL,
        temperature=0,
        callbacks=[usage_aggregator],
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    response = llm.invoke(
        [
            *messages,
            HumanMessage(
                content=(
                    "Generate a project name and description from these artifact chunks.\n"
                    "Return JSON only.\n\n"
                    f"{_document_context(documents)}"
                )
            ),
        ]
    )

    payload = extract_json_payload(response.content)
    name = str(payload.get("name", "")).strip()
    description = str(payload.get("description", "")).strip()

    if not name or not description:
        raise ValueError("Model response must include non-empty name and description.")

    return {"name": name, "description": description}


__all__ = [
    "generate_project_name_and_description",
    "llm",
]
