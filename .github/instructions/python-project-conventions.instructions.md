---
description: "Use when adding or updating Python dependencies, editing pyproject.toml, or creating/refactoring project folders and modules. Enforces uv add usage and clean src-based folder organization."
name: "Python Dependencies and Folder Structure"
applyTo:
  - "**/*.py"
  - "pyproject.toml"
  - "src/**"
---

# Python Dependency and Folder Structure Rules

## Dependency management (hard rule)

- Install packages with `uv add <package>` only.
- Install development-only packages with `uv add --dev <package>`.
- Do not hand-edit `[project].dependencies` in `pyproject.toml` unless the user explicitly asks for it.
- Keep `uv.lock` in sync after dependency updates.

## Folder structure (strong preference)

- Treat these structure rules as defaults, and adapt when existing architecture or team conventions require it.
- Keep importable application code inside `src/`.
- Organize code by feature or domain using clear names (for example `src/documents/`, `src/ingestion/`, `src/vectorstore/`) instead of generic buckets when possible.
- Prefer `src/lib/` for reusable utilities with minimal side effects.
- Prefer `src/module/` for feature workflows and orchestration entrypoints.
- Keep scripts and entrypoints thin; business logic should live in `src/`.
- Mirror package structure in tests (for example `tests/documents/test_parser.py`).

## When proposing new files

- Prefer creating a small package directory with `__init__.py` when a module grows beyond one file.
- Keep each folder focused on one feature concern.
- Avoid deep nesting unless it adds clear separation of concerns.
