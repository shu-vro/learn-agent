from langchain.chat_models import init_chat_model
from src.config.constants import DEFAULT_LLM_MODEL
from src.utils.usage_aggregator_callback import UsageAggregatorCallback
from src.vector_store.qdrant_store import (
    embeddings,
)


global_usage_aggregator = UsageAggregatorCallback("mem0_calls")
llm = init_chat_model(
    model=DEFAULT_LLM_MODEL,
    temperature=0,
    configurable_fields="any",
    callbacks=[global_usage_aggregator],
)

config = {
    "llm": {"provider": "langchain", "config": {"model": llm}},
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": embeddings,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            "host": "localhost",
            "port": 6333,
        },
    },
}
