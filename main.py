import argparse
import os
import src.config.bootstrap  # noqa: F401
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OCR_LIB,
    DEFAULT_PAPER_SOURCES,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VISION_MODEL,
    ENVIRONMENT,
)
from src.utils.time_utils import measure_time
from src.db import create_all_tables

if TYPE_CHECKING:
    from src.agent.rag_agent import RagAppConfig


def _check_database_connection() -> None:
    try:
        create_all_tables()
        print("Successfully connected to the database.")
    except Exception as e:
        print(f"Warning: Failed to connect to the database. Error: {e}")


def _default_worker_count() -> int:
    return max(1, os.cpu_count() or 1)


def _positive_int(value: str) -> int:
    int_value = int(value)
    if int_value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return int_value


def _ingest_documents(
    config: "RagAppConfig",
    *,
    rebuild: bool,
    use_vision_model: bool,
    use_image_descriptions: bool,
    use_formula_transcription: bool,
) -> dict[str, Any]:
    from src.module.upload_docs import ingest_paper_to_qdrant

    return ingest_paper_to_qdrant(
        source=config.sources,
        collection_name=config.collection_name,
        artifacts_root=config.artifacts_root,
        embedding_model_name=config.embedding_model,
        vision_model_name=config.vision_model,
        use_vision_model=use_vision_model,
        use_image_descriptions=use_image_descriptions,
        use_formula_transcription=use_formula_transcription,
        equation_ocr_lib=config.equation_ocr_lib,
        recreate_collection=rebuild,
    )


def _print_ingestion_summary(ingest_info: dict[str, Any]) -> None:
    if ingest_info.get("skipped_existing_paper"):
        print("Paper already indexed. Skipped ingestion.")
    else:
        print("Ingestion complete.")

    print(f"Indexed documents: {ingest_info['documents_indexed']}")
    paper_hashes = ingest_info.get("paper_sha256_list") or [ingest_info["paper_sha256"]]
    print("Paper SHA256 hashes:")
    for paper_hash in paper_hashes:
        print(f"- {paper_hash}")
    print(f"Qdrant collection: {ingest_info['collection_name']}")
    print(f"Artifacts dir: {ingest_info['artifacts_root']}")


def _run_api_server(*, host: str, port: int, workers: int, log_level: str) -> None:
    import uvicorn

    uvicorn.run(
        "src.api:create_api",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        proxy_headers=True,
        reload=ENVIRONMENT == "development",
    )


@measure_time
def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal RAG over Attention Is All You Need using Docling + Qdrant + Ollama."
    )
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        default=None,
        help="Paper source URL or local file path. Repeat this flag to ingest multiple papers.",
    )
    parser.add_argument(
        "--collection-name",
        "--index-dir",
        dest="collection_name",
        default=DEFAULT_QDRANT_COLLECTION,
        help="Qdrant collection name for indexed paper documents.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="data/artifacts",
        help="Directory for extracted markdown and images.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Ollama text generation model for QA.",
    )
    parser.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help="Ollama vision model used for image descriptions.",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="Number of retrieved chunks for each question.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable all vision features (image descriptions and formula transcription).",
    )
    parser.add_argument(
        "--no-image-description",
        action="store_true",
        help="Disable image descriptions while keeping other vision features enabled.",
    )
    parser.add_argument(
        "--no-formula-transcription",
        action="store_true",
        help="Disable formula LaTeX transcription from formula images.",
    )
    parser.add_argument(
        "--equation-ocr-lib",
        choices=("local", "llm"),
        default=DEFAULT_OCR_LIB,
        help=(
            "Formula OCR backend for LaTeX transcription "
            "(local=pix2tex, llm=Ollama vision)."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest the paper and write vectors to Qdrant."
    )
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index from scratch.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask one question to the RAG agent.")
    ask_parser.add_argument("question", help="Question about the paper.")
    ask_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index before answering.",
    )

    chat_parser = subparsers.add_parser(
        "chat", help="Run an interactive RAG chat session."
    )
    chat_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index before starting chat.",
    )

    api_parser = subparsers.add_parser(
        "api", help="Run the fastapi server for RAG API endpoints."
    )
    api_parser.add_argument(
        "--host",
        default=os.environ.get("API_HOST", "0.0.0.0"),
        help="API bind host (default: API_HOST or 0.0.0.0).",
    )
    api_parser.add_argument(
        "--port",
        type=_positive_int,
        default=int(os.environ.get("API_PORT", "8000")),
        help="API bind port (default: API_PORT or 8000).",
    )
    api_parser.add_argument(
        "--workers",
        type=_positive_int,
        default=_default_worker_count(),
        help="Number of Uvicorn worker processes.",
    )
    api_parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
        help="Uvicorn log level.",
    )

    for command_parser in (ingest_parser, ask_parser, chat_parser):
        command_parser.add_argument(
            "--no-vision",
            action="store_true",
            help="Disable all vision features (image descriptions and formula transcription).",
        )
        command_parser.add_argument(
            "--no-image-description",
            action="store_true",
            help="Disable image descriptions while keeping other vision features enabled.",
        )
        command_parser.add_argument(
            "--no-formula-transcription",
            action="store_true",
            help="Disable formula LaTeX transcription from formula images.",
        )
        command_parser.add_argument(
            "--equation-ocr-lib",
            choices=("local", "llm"),
            default=DEFAULT_OCR_LIB,
            help=(
                "Formula OCR backend for LaTeX transcription "
                "(local=pix2tex, llm=Ollama vision)."
            ),
        )

    return parser


@measure_time
def main() -> None:
    try:
        parser = _build_cli_parser()
        args = parser.parse_args()

        if args.command == "api":
            _check_database_connection()
            _run_api_server(
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_level=args.log_level,
            )
            return

        from src.agent.rag_agent import RagAppConfig, answer_question, interactive_chat

        use_vision_model = not args.no_vision
        use_image_descriptions = use_vision_model and not args.no_image_description
        use_formula_transcription = (
            use_vision_model and not args.no_formula_transcription
        )
        equation_ocr_lib = args.equation_ocr_lib
        selected_sources = args.sources or list(DEFAULT_PAPER_SOURCES)
        config = RagAppConfig(
            sources=selected_sources,
            collection_name=args.collection_name,
            artifacts_root=Path(args.artifacts_dir),
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            vision_model=args.vision_model,
            equation_ocr_lib=equation_ocr_lib,
            top_k=args.top_k,
        )

        if args.command == "ingest":
            ingest_info = _ingest_documents(
                config,
                rebuild=args.rebuild,
                use_vision_model=use_vision_model,
                use_image_descriptions=use_image_descriptions,
                use_formula_transcription=use_formula_transcription,
            )
            _print_ingestion_summary(ingest_info)
            return

        if args.rebuild:
            print(f"Rebuilding index before '{args.command}'...")
            ingest_info = _ingest_documents(
                config,
                rebuild=True,
                use_vision_model=use_vision_model,
                use_image_descriptions=use_image_descriptions,
                use_formula_transcription=use_formula_transcription,
            )
            _print_ingestion_summary(ingest_info)

        if args.command == "ask":
            answer_question(
                question=args.question,
                config=config,
            )
            return

        if args.command == "chat":
            interactive_chat(config=config)
            return
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        # if ENVIRONMENT == "development":
        #     import subprocess

        #     subprocess.run("killall -9 ollama", shell=True)


if __name__ == "__main__":
    main()
