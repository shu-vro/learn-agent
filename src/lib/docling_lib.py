import os
import re
import shutil
from pathlib import Path
from typing import Callable
from src.utils.time_utils import measure_time

from src.lib.paper_fingerprint import _sha256_for_file
from src.utils.textsplitters import chunk_text
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    PdfPipelineOptions,
    OcrMacOptions,
)
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import FormulaItem, ImageRefMode, PictureItem
from langchain_core.documents import Document


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
def _build_markdown_image_alt_text(description: str) -> str:
    description_clean = re.sub(r"\s+", " ", description).strip()

    # Avoid breaking markdown alt-text delimiters.
    description_clean = description_clean.replace("[", "(").replace("]", ")")

    if description_clean:
        return f"{description_clean}"

    return "Image"


@measure_time
def _replace_markdown_image_alt_texts(
    markdown_text: str, image_alt_texts: list[str]
) -> str:
    if not image_alt_texts:
        return markdown_text

    image_index = 0

    def _replacement(match: re.Match[str]) -> str:
        nonlocal image_index

        if image_index >= len(image_alt_texts):
            return match.group(0)

        image_path = match.group("path")
        alt_text = image_alt_texts[image_index]
        image_index += 1
        return f"![{alt_text}]({image_path})"

    return re.sub(
        r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\n]+)\)",
        _replacement,
        markdown_text,
    )


@measure_time
def _extract_markdown_image_paths(markdown_text: str) -> list[str]:
    return re.findall(
        r"!\[[^\]]*\]\((?P<path>[^)\n]+)\)",
        markdown_text,
    )


@measure_time
def _normalize_markdown_image_paths(markdown_text: str) -> str:
    """Rewrite image links to stable paths under the document's images/ folder."""

    def _rewrite(match: re.Match[str]) -> str:
        alt = match.group("alt")
        image_name = Path(match.group("path")).name
        return f"![{alt}](images/{image_name})"

    return re.sub(
        r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\n]+)\)",
        _rewrite,
        markdown_text,
    )


def _cleanup_docling_export_junk(doc_dir: Path, doc_filename: str) -> None:
    """Remove nested paths created by older Docling export defaults."""
    legacy_artifacts_dir = doc_dir / f"{doc_filename}_artifacts"
    if legacy_artifacts_dir.is_dir():
        shutil.rmtree(legacy_artifacts_dir)

    nested_data_dir = doc_dir / "data"
    if nested_data_dir.is_dir():
        shutil.rmtree(nested_data_dir)


@measure_time
def _build_docling_converter() -> DocumentConverter:
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.generate_page_images = True
    pdf_pipeline_options.generate_picture_images = True
    pdf_pipeline_options.images_scale = 2.0

    # Keep native formula enrichment disabled for speed; undecoded formulas can
    # be transcribed on demand from formula images during post-processing.
    pdf_pipeline_options.do_ocr = True
    pdf_pipeline_options.do_code_enrichment = False
    pdf_pipeline_options.do_formula_enrichment = False
    pdf_pipeline_options.do_picture_description = False
    pdf_pipeline_options.do_picture_classification = False

    # Force Transformers runtime for code/formula stage to avoid MLX auto-runtime fallback logs.
    pdf_pipeline_options.code_formula_options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2",
        # "granite_docling",
        engine_options=TransformersVlmEngineOptions(),
    )

    pdf_pipeline_options.ocr_options = OcrMacOptions(
        lang=["en-US", "bn-BD"],
        force_full_page_ocr=True,
    )

    # only if user is in mac
    if os.name == "posix" and "darwin" in os.uname().sysname.lower():
        pdf_pipeline_options.ocr_options = OcrMacOptions()

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )


# the extractor takes a massive amount of time (200+s for 15 page)
@measure_time
def docling_pdf_extractor(
    file_path: str,
    artifacts_root: str | Path = "data/artifacts",
    image_describer: Callable[[Path, str], str] | None = None,
    formula_transcriber: Callable[[Path, str], str] | None = None,
    chunk_size: int = 1800,
    chunk_overlap: int = 250,
    content_hash: str | None = None,
    upload_mode: bool = False,
) -> list[Document]:
    time_tracker: dict[str, float] = {}
    documents: list[Document] = []

    with measure_time("total_extraction", tracker=time_tracker):
        with measure_time("converter_build_time", tracker=time_tracker):
            converter = _build_docling_converter()
            conv_res = converter.convert(file_path)

        resolved_file_path = Path(file_path).expanduser().resolve()
        doc_id = content_hash or _sha256_for_file(resolved_file_path)
        artifacts_root_path = Path(artifacts_root)
        doc_artifacts_dir = artifacts_root_path / doc_id
        formula_output_dir = doc_artifacts_dir / "formulas"
        image_output_dir = doc_artifacts_dir / "images"
        doc_artifacts_dir.mkdir(parents=True, exist_ok=True)
        formula_output_dir.mkdir(parents=True, exist_ok=True)
        image_output_dir.mkdir(parents=True, exist_ok=True)

        md_filename = doc_artifacts_dir / f"{doc_id}.md"

        with measure_time("markdown_export", tracker=time_tracker):
            conv_res.document.save_as_markdown(
                md_filename,
                artifacts_dir=Path("images"),
                image_mode=ImageRefMode.REFERENCED,
            )
            full_markdown = md_filename.read_text(encoding="utf-8")
            _cleanup_docling_export_junk(doc_artifacts_dir, doc_id)
            markdown_image_paths = _extract_markdown_image_paths(full_markdown)
        formula_placeholder_token = "<!-- formula-not-decoded -->"
        pending_placeholders = full_markdown.count(formula_placeholder_token)

        formula_counter = 0
        processed_formula_counter = 0
        formula_placeholder_replacements: list[str] = []
        image_documents: list[Document] = []
        markdown_image_alt_texts: list[str] = []

        with measure_time("formula_processing", tracker=time_tracker):
            for element, _level in conv_res.document.iterate_items():
                if not isinstance(element, FormulaItem):
                    continue

                processed_formula_counter += 1
                with measure_time(
                    f"formula_processing_{processed_formula_counter}",
                    tracker=time_tracker,
                ):
                    formula_text = _sanitize_latex_expression(
                        (element.text or "").strip()
                    )
                    formula_orig = (element.orig or "").strip()

                    needs_transcription = (
                        not formula_text and formula_transcriber is not None
                    )
                    if needs_transcription:
                        formula_image = element.get_image(conv_res.document)
                        if formula_image is not None:
                            formula_image_filename = (
                                formula_output_dir
                                / f"{doc_id}-formula-{formula_counter + 1}.png"
                            )
                            formula_image.save(formula_image_filename, "PNG")

                            try:
                                transcribed = formula_transcriber(
                                    formula_image_filename,
                                    formula_orig,
                                )
                                if transcribed:
                                    formula_text = _sanitize_latex_expression(
                                        transcribed
                                    )
                            except Exception:  # noqa: BLE001
                                pass

                    if not formula_text and not formula_orig:
                        continue

                    formula_counter += 1
                    print(
                        formula_text
                        if formula_text
                        else f"Original formula (not decoded): {formula_orig}"
                    )
                    if formula_text and pending_placeholders > len(
                        formula_placeholder_replacements
                    ):
                        formula_placeholder_replacements.append(formula_text)

        if formula_placeholder_replacements:
            full_markdown = _replace_formula_placeholders(
                full_markdown,
                formula_placeholder_replacements,
            )

        with measure_time("image_processing", tracker=time_tracker):
            picture_counter = 0
            for element, _level in conv_res.document.iterate_items():
                if not isinstance(element, PictureItem):
                    continue

                if element.get_image(conv_res.document) is None:
                    continue

                picture_counter += 1
                if picture_counter > len(markdown_image_paths):
                    break

                with measure_time(
                    f"image_processing_{picture_counter}",
                    tracker=time_tracker,
                ):
                    element_image_filename = (
                        doc_artifacts_dir / markdown_image_paths[picture_counter - 1]
                    )

                    caption = element.caption_text(conv_res.document).strip()
                    picture_description = ""
                    if image_describer is not None:
                        try:
                            picture_description = image_describer(
                                element_image_filename,
                                caption,
                            )
                        except Exception as err:  # noqa: BLE001
                            picture_description = ""
                            print(
                                f"Image description failed for {element_image_filename}: {err}"
                            )

                    markdown_image_alt_texts.append(
                        _build_markdown_image_alt_text(picture_description)
                    )

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
                        image_context_parts.append(
                            f"Visual description: {picture_description}"
                        )

                    print(
                        image_context_parts,
                    )

                    image_documents.append(
                        Document(
                            page_content="\n".join(image_context_parts),
                            metadata={
                                "source": file_path,
                                "doc_id": doc_id,
                                "type": "image",
                                "path": str(element_image_filename),
                                "page": page_no,
                                "caption": caption,
                            },
                        )
                    )

        if markdown_image_alt_texts:
            full_markdown = _replace_markdown_image_alt_texts(
                full_markdown,
                markdown_image_alt_texts,
            )

        # Final formula cleanup applies to both markdown export and chunking source.
        full_markdown = _sanitize_markdown_formulas(full_markdown)
        full_markdown = _normalize_markdown_image_paths(full_markdown)
        md_filename.write_text(full_markdown, encoding="utf-8")

        s3_artifact_keys: list[str] = []
        s3_markdown_key: str | None = None
        if upload_mode:
            from src.lib.aws import (
                artifact_markdown_s3_key,
                artifact_s3_prefix,
                upload_artifacts_directory_to_s3,
            )

            with measure_time("s3_upload", tracker=time_tracker):
                s3_artifact_keys = upload_artifacts_directory_to_s3(
                    doc_artifacts_dir,
                    doc_id,
                )
                s3_markdown_key = artifact_markdown_s3_key(doc_id)
                print(
                    f"Uploaded {len(s3_artifact_keys)} artifact files to S3 "
                    f"under {artifact_s3_prefix(doc_id)}/"
                )

            for image_doc in image_documents:
                image_metadata = dict(image_doc.metadata or {})
                image_metadata["s3_artifact_prefix"] = artifact_s3_prefix(doc_id)
                image_doc.metadata = image_metadata

        with measure_time("text_chunking", tracker=time_tracker):
            markdown_chunks = chunk_text(
                text=full_markdown,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for idx, chunk in enumerate(markdown_chunks, start=1):
                chunk_metadata: dict[str, object] = {
                    "source": file_path,
                    "doc_id": doc_id,
                    "type": "text_chunk",
                    "chunk_id": idx,
                    "chunk_total": len(markdown_chunks),
                    "markdown_path": str(md_filename),
                }
                if upload_mode and s3_markdown_key:
                    chunk_metadata["s3_artifact_prefix"] = artifact_s3_prefix(doc_id)
                    chunk_metadata["markdown_s3_key"] = s3_markdown_key

                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=chunk_metadata,
                    )
                )

        documents.extend(image_documents)

    print("Extraction time breakdown:")
    for key, value in time_tracker.items():
        print(f"{key}: {value:.2f} seconds")

    return documents
