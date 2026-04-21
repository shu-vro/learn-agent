from pathlib import Path
import sys
from mem0 import Memory
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from src.utils.usage_aggregator_callback import UsageAggregatorCallback
from src.lib.embeddings import build_embeddings
import src.config.bootstrap  # noqa: F401

# Ensure top-level `src` imports resolve when this file is executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


global_usage_aggregator = UsageAggregatorCallback("mem0_task")

# LLM
llm = ChatOpenAI(
    model="gemma4:e2b",
    temperature=0,
    callbacks=[global_usage_aggregator],
)

# Embeddings
embeddings = build_embeddings()  # example

# Plain LangChain Qdrant vector store
qdrant_client = QdrantClient(
    url="http://localhost:6333",
)

# qdrant_client.create_collection(
#     collection_name="mem0",
#     vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
# )

vector_store_mem0 = QdrantVectorStore(
    client=qdrant_client,
    collection_name="mem0",  # required by Mem0 langchain vector-store provider
    embedding=embeddings,
)

config = {
    "llm": {
        "provider": "langchain",
        "config": {"model": llm},
    },
    "embedder": {
        "provider": "langchain",
        "config": {"model": embeddings},
    },
    # "vector_store": {
    #     "provider": "langchain",
    #     "config": {"client": vector_store_mem0, "collection_name": "mem0"},
    # },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            "host": "localhost",
            "port": 6333,
        },
    },
}

memory = Memory.from_config(config)

# memory.add(
#     messages=[
#         {"role": "user", "content": "I'm a vegetarian and allergic to nuts."},
#         {
#             "role": "assistant",
#             "content": "Got it! I'll remember your dietary preferences.",
#         },
#     ],
#     metadata={"source": "test"},
#     user_id="test_user",
# )

pref = memory.search("What are my favourite sports?", user_id="test_user")
print(pref)

print(global_usage_aggregator.aggregated_usage)
