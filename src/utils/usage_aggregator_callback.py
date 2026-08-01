from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _usage_from_llm_output(llm_output: dict[str, Any] | None) -> dict[str, Any] | None:
    if not llm_output:
        return None
    token_usage = llm_output.get("token_usage")
    if not isinstance(token_usage, dict) or not token_usage:
        return None
    return {
        "input_tokens": token_usage.get("prompt_tokens", 0),
        "output_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        **token_usage,
    }


def _cache_tokens_from_entry(entry: dict[str, Any]) -> int:
    """Extract cached input tokens from a usage metadata entry."""
    details = entry.get("input_token_details")
    if isinstance(details, dict):
        cache_read = int(details.get("cache_read") or 0)
        cache_creation = int(details.get("cache_creation") or 0)
        if cache_read or cache_creation:
            return cache_read + cache_creation

    # OpenAI-style nested prompt token details
    prompt_details = entry.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        return int(prompt_details.get("cached_tokens") or 0)

    return int(
        entry.get("cache_read_input_tokens")
        or entry.get("cached_tokens")
        or entry.get("cache_tokens")
        or 0
    )


def _normalize_entry(entry: dict[str, Any]) -> dict[str, int] | None:
    if not isinstance(entry, dict) or entry.get("warning"):
        return None
    inp = int(entry.get("input_tokens") or entry.get("prompt_tokens") or 0)
    out = int(entry.get("output_tokens") or entry.get("completion_tokens") or 0)
    cache = _cache_tokens_from_entry(entry)
    total = int(entry.get("total_tokens") or 0)
    if total == 0:
        total = inp + out
    if inp == 0 and out == 0 and cache == 0:
        return None
    return {
        "input_token": inp,
        "cache_token": cache,
        "output_token": out,
        "total_token": total,
    }


def summarize_usage(
    usage_aggregator: "UsageAggregatorCallback | None",
) -> dict[str, Any]:
    """Aggregate per-call usage into totals + per-iteration breakdown."""
    empty: dict[str, Any] = {
        "input_token": 0,
        "cache_token": 0,
        "output_token": 0,
        "total_token": 0,
        "iterations": 0,
        "iteration_details": [],
    }
    if not usage_aggregator:
        return empty

    entries = usage_aggregator.get_aggregated_usage().get(
        usage_aggregator.task_name, []
    )
    details: list[dict[str, int]] = []
    inp = cache = out = total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_entry(entry)
        if not normalized:
            continue
        details.append(
            {
                "iteration": len(details) + 1,
                "input_token": normalized["input_token"],
                "cache_token": normalized["cache_token"],
                "output_token": normalized["output_token"],
                "total_token": normalized["total_token"],
            }
        )
        inp += normalized["input_token"]
        cache += normalized["cache_token"]
        out += normalized["output_token"]
        total += normalized["total_token"]

    if total == 0:
        total = inp + out

    return {
        "input_token": inp,
        "cache_token": cache,
        "output_token": out,
        "total_token": total,
        "iterations": len(details),
        "iteration_details": details,
    }


class UsageAggregatorCallback(BaseCallbackHandler):
    """Callback handler that aggregates usage metadata."""

    def __init__(self, task_name: str = "default_task"):
        self.aggregated_usage = {task_name: []}
        self.task_name = task_name

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        llm_output_usage = _usage_from_llm_output(response.llm_output)

        for generation in response.generations[0]:
            usage_metadata = generation.message.usage_metadata
            if usage_metadata:
                self.aggregated_usage[self.task_name].append(usage_metadata)
            elif llm_output_usage:
                self.aggregated_usage[self.task_name].append(llm_output_usage)
            else:
                self.aggregated_usage[self.task_name].append(
                    {"warning": "No usage_metadata found"}
                )

    def get_aggregated_usage(self) -> List[Dict[str, list[Dict[str, Any]]]]:
        """Return the aggregated usage metadata."""
        return self.aggregated_usage

    def clear_aggregated_usage(self) -> None:
        """Clear the aggregated usage metadata."""
        self.aggregated_usage = {self.task_name: []}
