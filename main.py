import argparse
import src.config.bootstrap
from pathlib import Path

from src.module.rag_agent import RagAppConfig, answer_question, interactive_chat
from src.module.upload_docs import ingest_paper_to_faiss
from src.utils.time_utils import measure_time


@measure_time
def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal RAG over Attention Is All You Need using Docling + FAISS + Ollama."
    )
    parser.add_argument(
        "--source",
        default="https://arxiv.org/pdf/1706.03762",
        help="Paper source URL or local file path.",
    )
    parser.add_argument(
        "--index-dir",
        default="data/faiss_db",
        help="Directory where the FAISS index is stored.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="data/artifacts",
        help="Directory for extracted markdown and images.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument(
        "--llm-model",
        default="gemma4:e4b",
        help="Ollama text generation model for QA.",
    )
    parser.add_argument(
        "--vision-model",
        default="gemma4:e4b",
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
        help="Disable vision model and skip image descriptions.",
    )
    parser.add_argument(
        "--no-auto-pull",
        action="store_true",
        help="Do not auto-pull missing Ollama models.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest the paper and build FAISS index."
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
            help="Disable vision model and skip image descriptions.",
        )
        command_parser.add_argument(
            "--no-auto-pull",
            action="store_true",
            help="Do not auto-pull missing Ollama models.",
        )

    return parser


@measure_time
def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    use_vision_model = not args.no_vision
    auto_pull_models = not args.no_auto_pull
    config = RagAppConfig(
        source=args.source,
        index_dir=Path(args.index_dir),
        artifacts_root=Path(args.artifacts_dir),
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        vision_model=args.vision_model,
        top_k=args.top_k,
    )

    if args.command == "ingest":
        ingest_info = ingest_paper_to_faiss(
            source=config.source,
            index_dir=config.index_dir,
            artifacts_root=config.artifacts_root,
            embedding_model_name=config.embedding_model,
            vision_model_name=config.vision_model,
            use_vision_model=use_vision_model,
            auto_pull_models=auto_pull_models,
        )
        print("Ingestion complete.")
        print(f"Indexed documents: {ingest_info['documents_indexed']}")
        print(f"FAISS index dir: {ingest_info['index_dir']}")
        print(f"Artifacts dir: {ingest_info['artifacts_root']}")
        return

    if args.command == "ask":
        result = answer_question(
            question=args.question,
            config=config,
            rebuild_index=args.rebuild,
            use_vision_model=use_vision_model,
            auto_pull_models=auto_pull_models,
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
            auto_pull_models=auto_pull_models,
        )
        return


if __name__ == "__main__":
    main()
