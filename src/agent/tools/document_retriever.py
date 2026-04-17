from langchain.tools import tool
from langchain_core.documents import Document
from src.vector_store.qdrant_store import vector_store


def _format_context(documents: list[Document]) -> str:
    """this is for llm prompt."""
    context_blocks: list[str] = []

    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        block_header = (
            f"[Source {idx}] type={meta.get('type', 'unknown')}, "
            f"similarity_score={meta.get('similarity_score', 'n/a')}, "
            f"source={meta.get('source', 'unknown')}, "
            f"page={meta.get('page', 'n/a')}, "
            f"image_path={meta.get('path', 'n/a')}"
        )
        context_blocks.append(f"{block_header}\n{doc.page_content}")

    return "\n\n".join(context_blocks)


@tool(response_format="content_and_artifact")
def retrieve_context(
    query: str, score_threshold: float = 0.5
) -> tuple[str, list[Document]]:
    """Retrieve information to help answer a query. score threshold is between 0 and 1, higher means more relevant."""
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(
        query=query, k=5, score_threshold=score_threshold
    )

    retrieved_docs = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "similarity_score": score},
        )
        for doc, score in retrieved_docs_with_scores
    ]

    if not retrieved_docs:
        return "No documents found above the score threshold.", []

    serialized = _format_context(retrieved_docs)
    return serialized, retrieved_docs
