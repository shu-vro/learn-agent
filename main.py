import argparse
import src.config.bootstrap
from pathlib import Path

from src.module.rag_agent import RagAppConfig, answer_question, interactive_chat
from src.module.upload_docs import ingest_paper_to_qdrant
from src.config.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OCR_LIB,
    DEFAULT_PAPER_SOURCES,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VISION_MODEL,
)
from src.utils.time_utils import measure_time


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
    parser = _build_cli_parser()
    args = parser.parse_args()

    use_vision_model = not args.no_vision
    use_image_descriptions = use_vision_model and not args.no_image_description
    use_formula_transcription = use_vision_model and not args.no_formula_transcription
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
        ingest_info = ingest_paper_to_qdrant(
            source=config.sources,
            collection_name=config.collection_name,
            artifacts_root=config.artifacts_root,
            embedding_model_name=config.embedding_model,
            vision_model_name=config.vision_model,
            use_vision_model=use_vision_model,
            use_image_descriptions=use_image_descriptions,
            use_formula_transcription=use_formula_transcription,
            equation_ocr_lib=config.equation_ocr_lib,
            recreate_collection=args.rebuild,
        )
        if ingest_info.get("skipped_existing_paper"):
            print("Paper already indexed. Skipped ingestion.")
        else:
            print("Ingestion complete.")
        print(f"Indexed documents: {ingest_info['documents_indexed']}")
        paper_hashes = ingest_info.get("paper_sha256_list") or [
            ingest_info["paper_sha256"]
        ]
        print("Paper SHA256 hashes:")
        for paper_hash in paper_hashes:
            print(f"- {paper_hash}")
        print(f"Qdrant collection: {ingest_info['collection_name']}")
        print(f"Artifacts dir: {ingest_info['artifacts_root']}")
        return

    if args.command == "ask":
        result = answer_question(
            question=args.question,
            config=config,
            rebuild_index=args.rebuild,
            use_vision_model=use_vision_model,
            use_image_descriptions=use_image_descriptions,
            use_formula_transcription=use_formula_transcription,
            equation_ocr_lib=config.equation_ocr_lib,
        )
        print("Answer:\n")
        print(result["answer"])
        print("\nSources:")
        for line in result["source_summaries"]:
            print(f"- {line}")
        return

    if args.command == "chat":
        interactive_chat(
            config=config,
            rebuild_index=args.rebuild,
            use_vision_model=use_vision_model,
            use_image_descriptions=use_image_descriptions,
            use_formula_transcription=use_formula_transcription,
            equation_ocr_lib=config.equation_ocr_lib,
        )
        return


if __name__ == "__main__":
    main()
