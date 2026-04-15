from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
    SparseVectorParams,
)

from src.config.constants import DEFAULT_EMBEDDING_MODEL
from src.config.env import (
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
)


from src.lib.embeddings import build_embeddings


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)


def _build_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, https=False
    )


def _get_collection_names(qdrant_client: QdrantClient) -> set[str]:
    return {
        collection.name for collection in qdrant_client.get_collections().collections
    }


def _get_embedding_dimension(embedding_model_name: str) -> int:
    embedding = build_embeddings(embedding_model_name)
    return len(embedding.embed_query("dimension probe"))


def qdrant_collection_exists(
    collection_name: str,
    qdrant_client: QdrantClient | None = None,
) -> bool:
    target_client = qdrant_client or client
    return collection_name in _get_collection_names(target_client)


def ensure_qdrant_collection(
    collection_name: str,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    qdrant_client: QdrantClient | None = None,
) -> None:
    target_client = qdrant_client or client
    if qdrant_collection_exists(collection_name, qdrant_client=target_client):
        return

    target_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=_get_embedding_dimension(embedding_model_name),
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )


def build_hybrid_qdrant_store(
    collection_name: str,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    qdrant_client: QdrantClient | None = None,
) -> QdrantVectorStore:
    target_client = qdrant_client or client
    ensure_qdrant_collection(
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        qdrant_client=target_client,
    )
    return QdrantVectorStore(
        client=target_client,
        collection_name=collection_name,
        embedding=build_embeddings(embedding_model_name),
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def qdrant_paper_hash_exists(
    collection_name: str,
    paper_sha256: str,
    qdrant_client: QdrantClient | None = None,
) -> bool:
    target_client = qdrant_client or client
    if not qdrant_collection_exists(collection_name, qdrant_client=target_client):
        return False

    points, _ = target_client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.paper_sha256",
                    match=MatchValue(value=paper_sha256),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


def create_qdrant_index(
    documents: list[Document],
    embedding_model_name: str,
    collection_name: str,
    recreate: bool = False,
    qdrant_client: QdrantClient | None = None,
) -> QdrantVectorStore:
    target_client = qdrant_client or client

    if recreate and qdrant_collection_exists(
        collection_name,
        qdrant_client=target_client,
    ):
        target_client.delete_collection(collection_name=collection_name)

    vectorstore = build_hybrid_qdrant_store(
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        qdrant_client=target_client,
    )
    if documents:
        vectorstore.add_documents(documents)
    return vectorstore


def load_qdrant_index(
    embedding_model_name: str,
    collection_name: str,
    qdrant_client: QdrantClient | None = None,
) -> QdrantVectorStore:
    target_client = qdrant_client or client

    if not qdrant_collection_exists(collection_name, qdrant_client=target_client):
        raise ValueError(
            f"Qdrant collection '{collection_name}' does not exist. Ingest first or change the collection name."
        )

    return build_hybrid_qdrant_store(
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        qdrant_client=target_client,
    )


client = _build_qdrant_client()
embeddings = build_embeddings()

COLLECTIONS = {"store": "store", "chats": "chats"}
for collection_alias in COLLECTIONS.values():
    ensure_qdrant_collection(collection_alias)

vector_store = build_hybrid_qdrant_store(
    collection_name=COLLECTIONS["store"],
    qdrant_client=client,
)
