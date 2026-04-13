---
description: "Use when editing Python RAG ingestion or retrieval code in this repo (Docling, FAISS, Ollama). Applies preferred uv workflow, Docling image safety, and targeted validation patterns."
name: "Python RAG Preferences"
applyTo: ["main.py", "src/**/*.py"]
---

# Python RAG Preferences

- Treat these as preferred defaults, not hard constraints. If a task needs a different approach, follow the task and explain the tradeoff.
- Prefer `uv run ...` commands for local execution in this repository.
- For syntax validation, compile only touched files with `python -m py_compile <files>` instead of running `compileall` over the full environment.
- If code calls `PictureItem.get_image(...)` or formula `get_image(...)`, default to keeping Docling image generation enabled:
  - `PdfPipelineOptions.generate_page_images = True`
  - `PdfPipelineOptions.generate_picture_images = True`
- Default to guarding image extraction before saving:
  - `img = element.get_image(doc)`
  - `if img is not None: img.save(...)`
- Prefer preserving stable `Document.metadata` keys unless a task explicitly requires schema changes: `source`, `doc_id`, `type`, and relevant path/page keys.
- Prefer `pathlib.Path` for artifact output paths.
- Keep exception handling localized to integration boundaries (OCR/model/network). If handling a broad exception, return a safe fallback.
