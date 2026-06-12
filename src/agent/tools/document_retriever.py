from langchain.tools import tool
from langchain_core.documents import Document
from typing import Any

from src.vector_store.qdrant_store import vector_store


def _format_context(documents: list[Document]) -> str:
    """this is for llm prompt."""
    context_blocks: list[str] = []

    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        block_header = (
            f"[Source {idx}] reference_id={meta.get('doc_id', 'n/a')}:{meta.get('chunk_id', 'n/a')}, "
            f"type={meta.get('type', 'unknown')}, "
            f"similarity_score={meta.get('similarity_score', 'n/a')}, "
            f"page={meta.get('page', 'n/a')}"
        )
        if meta.get("type") == "image":
            block_header += (
                f", image_path={meta.get('path', 'n/a')}, "
                f"caption={meta.get('caption', 'n/a')}"
            )
        context_blocks.append(f"{block_header}\n{doc.page_content}")

    return "\n\n".join(context_blocks)


def retrieve_context_tool(filters: dict[str, Any] = {}):

    @tool(response_format="content_and_artifact")
    def retrieve_context(
        query: str, score_threshold: float = 0.25
    ) -> tuple[str, list[Document]]:
        """Search uploaded documents for passages relevant to the query.

        Uses hybrid dense+sparse retrieval with RRF score fusion. Returned
        similarity_score values are RRF ranks, not cosine similarity — top hits
        are typically 0.25–0.5. Set score_threshold lower to include more
        results (e.g. 0.5 strict, 0.35 relaxed, 0.25 broad); higher values
        filter out weaker matches.
        """
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(
            query=query, k=5, score_threshold=score_threshold, filter=filters
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

    return retrieve_context
