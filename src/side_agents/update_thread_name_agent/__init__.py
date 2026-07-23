from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.constants import DEFAULT_PROJECT_NAME_AND_DESCRIPTION_MODEL
from src.config.model_config import model_selector
from src.utils.side_agents import extract_json_payload
from src.utils.usage_aggregator_callback import UsageAggregatorCallback

PROMPT_PATH = Path(__file__).with_name("prompt.md")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()


def generate_thread_name(*, query: str, answer: str) -> str:
    """Derive a short thread title from the first user query and AI answer."""
    usage_aggregator = UsageAggregatorCallback("thread_name_update_agent_usage")
    llm = model_selector(
        DEFAULT_PROJECT_NAME_AND_DESCRIPTION_MODEL,
        temperature=0,
        callbacks=[usage_aggregator],
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Generate a thread name from this first exchange.\n"
                    "Return JSON only.\n\n"
                    f"User question:\n{query.strip()}\n\n"
                    f"Assistant answer:\n{answer.strip()}"
                )
            ),
        ]
    )

    payload = extract_json_payload(response.content)
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("Model response must include a non-empty name.")
    # Soft cap so UI/sidebar stays readable.
    if len(name) > 80:
        name = name[:77].rstrip() + "..."
    return name


__all__ = ["generate_thread_name"]
