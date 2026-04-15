from pathlib import Path
from typing import Any


class Pix2TexFormulaTranscriber:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._init_error: Exception | None = None

    def _ensure_model(self) -> None:
        if self._model is not None or self._init_error is not None:
            return

        try:
            from pix2tex.cli import LatexOCR

            self._model = LatexOCR()
        except Exception as err:
            self._init_error = err

    def transcribe_formula_latex(self, image_path: Path, formula_hint: str = "") -> str:
        _ = formula_hint
        self._ensure_model()
        if self._model is None:
            return ""

        try:
            from PIL import Image

            with Image.open(image_path) as img:
                latex_text = self._model(img.convert("RGB"))
        except Exception:
            return ""

        if not latex_text:
            return ""

        cleaned = latex_text.strip().strip("`").replace("$", "").strip()
        if not cleaned or cleaned.upper() == "NOT_AVAILABLE":
            return ""

        return cleaned
