from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.lib.embeddings import build_embeddings


def faiss_index_exists(index_dir: str | Path) -> bool:
    index_path = Path(index_dir)
    return (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists()


def create_faiss_index(
    documents: list[Document],
    embedding_model_name: str,
    index_dir: str | Path,
) -> FAISS:
    embeddings = build_embeddings(embedding_model_name)
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    return vectorstore


def load_faiss_index(
    embedding_model_name: str,
    index_dir: str | Path,
) -> FAISS:
    embeddings = build_embeddings(embedding_model_name)
    return FAISS.load_local(
        folder_path=str(Path(index_dir)),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
