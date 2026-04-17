# Multimodal RAG over Research Papers

This project builds a local multimodal RAG pipeline over one or more papers using Docling ingestion, Qdrant retrieval, and Ollama generation.

Default paper sources:

- https://arxiv.org/pdf/1706.03762
- https://arxiv.org/pdf/2603.15031

It uses:

- Docling for PDF parsing, markdown extraction, and artifact/image generation
- Qdrant (hybrid dense + sparse retrieval via `langchain-qdrant`)
- `Octen/Octen-Embedding-0.6B` for dense embeddings
- Ollama `gemma4:e2b` for response generation
- Optional formula OCR (`pix2tex` or Ollama vision, controlled by `--equation-ocr-lib`)

## Project Structure

- `src/module/upload_docs.py`: ingestion workflow from source PDF(s) into Qdrant
- `src/module/rag_agent.py`: retrieval + strict context-grounded QA/chat agent
- `src/vector_store/qdrant_store.py`: Qdrant client, collection helpers, hybrid vector store
- `src/lib/docling_lib.py`: Docling conversion, chunking, and artifact extraction
- `main.py`: CLI entrypoint (`ingest`, `ask`, `chat`)

## Prerequisites

1. Install dependencies:

```bash
uv sync
```

2. Make sure Qdrant is running (default: `localhost:6333`).

Example local run:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

3. Make sure Ollama is running.

4. Pull required Ollama model(s):

```bash
ollama pull gemma4:e2b
```

## Build the Index

```bash
uv run main.py ingest --rebuild
```

This ingests documents into the Qdrant collection (default: `store`) and writes extracted artifacts under `data/artifacts`.

## Ask a Single Question

```bash
uv run main.py ask "What is the core idea of scaled dot-product attention?"
```

Optional: force re-ingestion before asking.

```bash
uv run main.py ask "What is the core idea of scaled dot-product attention?" --rebuild
```

## Start Interactive Chat

```bash
uv run main.py chat
```

Optional: force re-ingestion before chat.

```bash
uv run main.py chat --rebuild
```

## Current Agent Behavior

- Answers are constrained to retrieved context; if information is missing, the agent explicitly says it could not find it in indexed context.
- Responses are streamed token-by-token in the terminal.
- Source lines are printed after each answer (`type`, `source`, `page`, `image`).
- `chat` mode keeps in-memory conversation state and enables `SummarizationMiddleware` (trigger: 500 tokens, keep last 2 messages).
- Usage metadata is printed in `ask` mode per call and aggregated at the end of `chat` mode.

## Useful Options

- Ingest multiple sources:

```bash
uv run main.py ingest --source https://arxiv.org/pdf/1706.03762 --source https://arxiv.org/pdf/2603.15031
```

- Set retrieval depth:

```bash
uv run main.py ask "Summarize encoder-decoder attention" --top-k 8
```

- Disable all vision enrichment:

```bash
uv run main.py ingest --rebuild --no-vision
```

- Disable only image descriptions:

```bash
uv run main.py ingest --rebuild --no-image-description
```

- Disable only formula transcription:

```bash
uv run main.py ingest --rebuild --no-formula-transcription
```

- Select formula OCR backend:

```bash
uv run main.py ingest --rebuild --equation-ocr-lib llm
```

Note: current retrieval in `rag_agent` reads from the default collection (`store`) during `ask/chat`.

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
