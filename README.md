# Multimodal RAG over Attention Is All You Need

This project builds a local multimodal RAG pipeline for the paper below:

- https://arxiv.org/pdf/1706.03762

It uses:

- Docling for PDF parsing and figure extraction
- FAISS as the vector store
- all-MiniLM-L6-v2 for embeddings
- Ollama gemma4:e4b for answer generation
- An Ollama vision model (default: moondream) for image descriptions

## Project Structure

- src/lib/docling_lib.py: document conversion, chunking, image extraction, multimodal doc creation
- src/lib/ollama_vision.py: local vision-captioning client through Ollama API
- src/lib/faiss_store.py: FAISS build/load helpers
- src/module/upload_docs.py: ingestion workflow from source PDF into FAISS
- src/module/rag_agent.py: retrieval + Gemma4 answer agent
- main.py: CLI entrypoint (ingest, ask, chat)

## Prerequisites

1. Install dependencies:

```bash
uv sync
```

2. Make sure Ollama is running.

3. Pull required Ollama models:

```bash
ollama pull gemma4:e4b
ollama pull moondream
```

If you prefer a different vision model, set it with `--vision-model`.

## Build the Index

```bash
uv run main.py ingest --rebuild
```

The command writes:

- FAISS index to `data/faiss_db`
- Extracted markdown and images to `data/artifacts`

## Ask a Single Question

```bash
uv run main.py ask "What is the core idea of scaled dot-product attention?"
```

## Start Interactive Chat

```bash
uv run main.py chat
```

## Useful Options

- Disable image vision enrichment:

```bash
uv run main.py ingest --rebuild --no-vision
```

- Use a custom source:

```bash
uv run main.py ask "Summarize encoder-decoder attention" --source https://arxiv.org/pdf/1706.03762
```

- Disable automatic model pull:

```bash
uv run main.py ask "How are positional encodings defined?" --no-auto-pull
```
