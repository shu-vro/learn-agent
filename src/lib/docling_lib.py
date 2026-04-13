import os
import re
from pathlib import Path
from typing import Callable
from src.utils.time_utils import measure_time

import nltk
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    OcrMacOptions,
    PdfPipelineOptions,
)
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import FormulaItem, ImageRefMode, PictureItem
from langchain_core.documents import Document
from langchain_text_splitters import NLTKTextSplitter


@measure_time
def _replace_formula_placeholders(
    markdown_text: str, latex_replacements: list[str]
) -> str:
    updated_text = markdown_text
    for latex in latex_replacements:
        updated_text = updated_text.replace(
            "<!-- formula-not-decoded -->", f"$${latex}$$", 1
        )
    return updated_text


@measure_time
def _sanitize_latex_expression(latex_expression: str) -> str:
    expression = latex_expression.strip()
    if not expression:
        return expression

    aligned_env_markers = (
        r"\\begin{align",
        r"\\begin{aligned}",
        r"\\begin{array}",
        r"\\begin{matrix}",
        r"\\begin{pmatrix}",
        r"\\begin{bmatrix}",
        r"\\begin{cases}",
    )
    uses_alignment_env = any(marker in expression for marker in aligned_env_markers)

    # OCR/VLM often emits stray alignment markers (&) even when not using align environments.
    if not uses_alignment_env:
        expression = re.sub(r"(?<!\\)\s*&\s*", " ", expression)

    expression = re.sub(r"\s*\\\\\s*", r" \\\\ ", expression)
    expression = re.sub(r"\s{2,}", " ", expression)
    return expression.strip()


@measure_time
def _sanitize_markdown_formulas(markdown_text: str) -> str:
    def _sanitize_block_formula(match: re.Match[str]) -> str:
        formula = match.group(1)
        return f"$${_sanitize_latex_expression(formula)}$$"

    def _sanitize_inline_formula(match: re.Match[str]) -> str:
        formula = match.group(1)
        return f"${_sanitize_latex_expression(formula)}$"

    sanitized = re.sub(
        r"\$\$(.*?)\$\$",
        _sanitize_block_formula,
        markdown_text,
        flags=re.DOTALL,
    )
    sanitized = re.sub(
        r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
        _sanitize_inline_formula,
        sanitized,
        flags=re.DOTALL,
    )
    return sanitized


@measure_time
def _build_docling_converter() -> DocumentConverter:
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.generate_page_images = True
    pdf_pipeline_options.generate_picture_images = True
    pdf_pipeline_options.images_scale = 2.0

    # Enable formula enrichment to get LaTeX extraction.
    pdf_pipeline_options.do_code_enrichment = False
    pdf_pipeline_options.do_formula_enrichment = True
    pdf_pipeline_options.do_picture_description = False

    # Force Transformers runtime for code/formula stage to avoid MLX auto-runtime fallback logs.
    pdf_pipeline_options.code_formula_options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2",
        # "granite_docling",
        engine_options=TransformersVlmEngineOptions(),
    )

    # only if user is in mac
    if os.name == "posix" and "darwin" in os.uname().sysname.lower():
        pdf_pipeline_options.ocr_options = OcrMacOptions()

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )


@measure_time
def _ensure_nltk_resources() -> bool:
    resources = ["tokenizers/punkt", "tokenizers/punkt_tab"]
    for resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            return False

    return True


@measure_time
def _chunk_text(
    text: str, chunk_size: int = 1800, chunk_overlap: int = 250
) -> list[str]:
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    has_nltk_resources = _ensure_nltk_resources()
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if has_nltk_resources:
        chunks = text_splitter.split_text(text)
    else:
        sentence_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        sentence_splits = sentence_tokenizer.tokenize(text)
        chunks = text_splitter._merge_splits(sentence_splits, "\n\n")

    return [chunk.strip() for chunk in chunks if chunk.strip()]


@measure_time
def docling_pdf_extractor(
    file_path: str,
    artifacts_root: str | Path = "data/artifacts",
    image_describer: Callable[[Path, str], str] | None = None,
    formula_transcriber: Callable[[Path, str], str] | None = None,
    chunk_size: int = 1800,
    chunk_overlap: int = 250,
) -> list[Document]:
    converter = _build_docling_converter()
    conv_res = converter.convert(file_path)

    doc_filename = conv_res.input.file.stem
    artifacts_root_path = Path(artifacts_root)
    markdown_dir = artifacts_root_path / "markdown" / doc_filename
    image_output_dir = artifacts_root_path / "images" / doc_filename
    formula_output_dir = artifacts_root_path / "formulas" / doc_filename
    markdown_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)
    formula_output_dir.mkdir(parents=True, exist_ok=True)

    md_filename = markdown_dir / f"{doc_filename}.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
    full_markdown = md_filename.read_text(encoding="utf-8")

    documents: list[Document] = []

    formula_counter = 0
    formula_placeholder_replacements: list[str] = []
    for element, _level in conv_res.document.iterate_items():
        if not isinstance(element, FormulaItem):
            continue

        formula_text = _sanitize_latex_expression((element.text or "").strip())
        formula_orig = (element.orig or "").strip()
        had_placeholder = not formula_text and bool(formula_orig)

        formula_image_path: str | None = None
        formula_image = element.get_image(conv_res.document)
        if formula_image is not None:
            formula_image_filename = (
                formula_output_dir / f"{doc_filename}-formula-{formula_counter + 1}.png"
            )
            formula_image.save(formula_image_filename, "PNG")
            formula_image_path = str(formula_image_filename)

            if not formula_text and formula_transcriber is not None:
                try:
                    transcribed = formula_transcriber(
                        formula_image_filename, formula_orig
                    )
                    if transcribed:
                        formula_text = _sanitize_latex_expression(transcribed)
                except Exception:  # noqa: BLE001
                    pass

        if not formula_text and not formula_orig:
            continue

        formula_counter += 1
        page_no = element.prov[0].page_no if element.prov else None
        if formula_text:
            page_content = f"Formula LaTeX: {formula_text}"
            formula_status = "transcribed" if had_placeholder else "decoded"
        else:
            page_content = f"Formula source (not decoded): {formula_orig}"
            formula_status = "not_decoded"

        if had_placeholder and formula_text:
            formula_placeholder_replacements.append(formula_text)

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": file_path,
                    "doc_id": doc_filename,
                    "type": "formula",
                    "formula_id": formula_counter,
                    "formula_status": formula_status,
                    "page": page_no,
                    "path": formula_image_path,
                },
            )
        )

    if formula_placeholder_replacements:
        full_markdown = _replace_formula_placeholders(
            full_markdown,
            formula_placeholder_replacements,
        )

    # Final formula cleanup applies to both markdown export and chunking source.
    full_markdown = _sanitize_markdown_formulas(full_markdown)
    md_filename.write_text(full_markdown, encoding="utf-8")

    markdown_chunks = _chunk_text(
        text=full_markdown,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    for idx, chunk in enumerate(markdown_chunks, start=1):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "doc_id": doc_filename,
                    "type": "text_chunk",
                    "chunk_id": idx,
                    "chunk_total": len(markdown_chunks),
                    "markdown_path": str(md_filename),
                },
            )
        )

    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if not isinstance(element, PictureItem):
            continue

        picture_image = element.get_image(conv_res.document)
        if picture_image is None:
            continue

        picture_counter += 1
        element_image_filename = (
            image_output_dir / f"{doc_filename}-picture-{picture_counter}.png"
        )
        picture_image.save(element_image_filename, "PNG")

        caption = element.caption_text(conv_res.document).strip()
        picture_description = ""
        if image_describer is not None:
            try:
                picture_description = image_describer(element_image_filename, caption)
            except Exception as err:  # noqa: BLE001
                # picture_description = f"Image description failed: {err}"
                picture_description = ""
                print(f"Image description failed for {element_image_filename}: {err}")

        page_no = element.prov[0].page_no if element.prov else None
        image_context_parts = [
            f"Image extracted from: {file_path}",
            f"Image path: {element_image_filename}",
            f"Page: {page_no if page_no is not None else 'unknown'}",
        ]
        if caption:
            image_context_parts.append(f"Caption: {caption}")
        else:
            image_context_parts.append("Caption: No caption available.")
        if picture_description:
            image_context_parts.append(f"Visual description: {picture_description}")

        print(
            f"Processed image {element_image_filename},\n caption: {caption},\n description: {picture_description}",
            image_context_parts,
        )

        documents.append(
            Document(
                page_content="\n".join(image_context_parts),
                metadata={
                    "source": file_path,
                    "doc_id": doc_filename,
                    "type": "image",
                    "path": str(element_image_filename),
                    "page": page_no,
                    "caption": caption,
                },
            )
        )

    return documents
