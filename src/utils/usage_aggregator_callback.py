from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import List, Dict, Any


class UsageAggregatorCallback(BaseCallbackHandler):
    """Callback handler that aggregates usage metadata."""

    def __init__(self, task_name: str = "default_task"):
        self.aggregated_usage = {task_name: []}
        self.task_name = task_name

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

        for messages in response.generations[0]:
            if messages.message.usage_metadata:
                self.aggregated_usage[self.task_name].append(
                    messages.message.usage_metadata
                )
            else:
                self.aggregated_usage[self.task_name].append(
                    {"warning": "No usage_metadata found"}
                )

    def get_aggregated_usage(self) -> List[Dict[str, Any]]:
        """Return the aggregated usage metadata."""
        return self.aggregated_usage

    def clear_aggregated_usage(self) -> None:
        """Clear the aggregated usage metadata."""
        self.aggregated_usage = {self.task_name: []}
