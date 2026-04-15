# Multimodal RAG over Attention Is All You Need

This project builds a local multimodal RAG pipeline for the paper below:

- https://arxiv.org/pdf/1706.03762

It uses:

- Docling for PDF parsing and figure extraction
- FAISS as the vector store
- all-MiniLM-L6-v2 for embeddings
- Ollama gemma4:e2b for answer generation
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

- Disable all vision enrichment (both image descriptions and formula transcription):

```bash
uv run main.py ingest --rebuild --no-vision
```

- Keep formula LaTeX transcription but skip figure descriptions (faster than full vision mode):

```bash
uv run main.py ingest --rebuild --no-image-description
```

- Disable formula transcription only (keeps image descriptions enabled):

```bash
uv run main.py ingest --rebuild --no-formula-transcription
```

- Select formula OCR backend (`local` uses pix2tex and is default, `llm` uses Ollama vision):

```bash
uv run main.py ingest --rebuild --equation-ocr-lib llm
```

- Set default formula OCR backend from environment:

```bash
DEFAULT_OCR_LIB=local
```

- Use a custom source:

```bash
uv run main.py ask "Summarize encoder-decoder attention" --source https://arxiv.org/pdf/1706.03762
```

## Help

```bash
uv run main.py --help
```

```
usage: main.py [-h] [--source SOURCES] [--collection-name COLLECTION_NAME] [--artifacts-dir ARTIFACTS_DIR]
               [--embedding-model EMBEDDING_MODEL] [--llm-model LLM_MODEL] [--vision-model VISION_MODEL] [--top-k TOP_K]
               [--no-vision] [--no-image-description] [--no-formula-transcription] [--equation-ocr-lib {local,llm}]
               {ingest,ask,chat} ...

Multimodal RAG over Attention Is All You Need using Docling + Qdrant + Ollama.

positional arguments:
  {ingest,ask,chat}
    ingest              Ingest the paper and write vectors to Qdrant.
    ask                 Ask one question to the RAG agent.
    chat                Run an interactive RAG chat session.

options:
  -h, --help            show this help message and exit
  --source SOURCES      Paper source URL or local file path. Repeat this flag to ingest multiple papers.
  --collection-name COLLECTION_NAME, --index-dir COLLECTION_NAME
                        Qdrant collection name for indexed paper documents.
  --artifacts-dir ARTIFACTS_DIR
                        Directory for extracted markdown and images.
  --embedding-model EMBEDDING_MODEL
                        SentenceTransformer embedding model name.
  --llm-model LLM_MODEL
                        Ollama text generation model for QA.
  --vision-model VISION_MODEL
                        Ollama vision model used for image descriptions.
  --top-k TOP_K         Number of retrieved chunks for each question.
  --no-vision           Disable all vision features (image descriptions and formula transcription).
  --no-image-description
                        Disable image descriptions while keeping other vision features enabled.
  --no-formula-transcription
                        Disable formula LaTeX transcription from formula images.
  --equation-ocr-lib {local,llm}
                        Formula OCR backend for LaTeX transcription (local=pix2tex, llm=Ollama vision).
```

> [!CAUTION]
> This project is a work in progress and may contain incomplete features, bugs, or suboptimal implementations. It is intended for educational and experimental purposes only. Use at your own risk.
