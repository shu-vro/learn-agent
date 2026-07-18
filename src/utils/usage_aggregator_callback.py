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
