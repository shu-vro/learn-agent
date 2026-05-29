from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from src.config.constants import DEFAULT_OCR_LIB
from src.db.models.preferences import Preferences

ThemeChoice = Literal["system", "light", "dark"]
EquationOcrLib = Literal["local", "llm"]


class IngestionPreferences(BaseModel):
    use_vision_model: bool = True
    use_image_descriptions: bool = True
    use_formula_transcription: bool = True
    equation_ocr_lib: EquationOcrLib = DEFAULT_OCR_LIB  # type: ignore[assignment]

    @field_validator("equation_ocr_lib", mode="before")
    @classmethod
    def normalize_ocr_lib(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"local", "llm"}:
            raise ValueError("equation_ocr_lib must be either 'local' or 'llm'.")
        return normalized


class UserPreferencesPublic(BaseModel):
    theme: ThemeChoice = "system"
    ingestion: IngestionPreferences

    @classmethod
    def from_model(cls, prefs: Preferences) -> "UserPreferencesPublic":
        return cls(
            theme=prefs.theme,  # type: ignore[arg-type]
            ingestion=IngestionPreferences(
                use_vision_model=bool(prefs.use_vision_model),
                use_image_descriptions=bool(prefs.use_image_descriptions),
                use_formula_transcription=bool(prefs.use_formula_transcription),
                equation_ocr_lib=prefs.equation_ocr_lib,  # type: ignore[arg-type]
            ),
        )


class IngestionPreferencesUpdate(BaseModel):
    use_vision_model: bool | None = None
    use_image_descriptions: bool | None = None
    use_formula_transcription: bool | None = None
    equation_ocr_lib: EquationOcrLib | None = None


class UserPreferencesUpdate(BaseModel):
    theme: ThemeChoice | None = None
    ingestion: IngestionPreferencesUpdate | None = None


def resolve_ingestion_flags(
    base: IngestionPreferences,
    *,
    use_vision_model: bool | None = None,
    use_image_descriptions: bool | None = None,
    use_formula_transcription: bool | None = None,
    equation_ocr_lib: str | None = None,
) -> IngestionPreferences:
    vision = use_vision_model if use_vision_model is not None else base.use_vision_model
    images = (
        use_image_descriptions
        if use_image_descriptions is not None
        else base.use_image_descriptions
    )
    formulas = (
        use_formula_transcription
        if use_formula_transcription is not None
        else base.use_formula_transcription
    )
    ocr = equation_ocr_lib if equation_ocr_lib is not None else base.equation_ocr_lib

    if not vision:
        images = False
        formulas = False

    return IngestionPreferences(
        use_vision_model=vision,
        use_image_descriptions=images,
        use_formula_transcription=formulas,
        equation_ocr_lib=ocr,  # type: ignore[arg-type]
    )


def apply_preferences_update(
    prefs: Preferences, payload: UserPreferencesUpdate
) -> None:
    if payload.theme is not None:
        prefs.theme = payload.theme
    if payload.ingestion is not None:
        ing = payload.ingestion
        if ing.use_vision_model is not None:
            prefs.use_vision_model = ing.use_vision_model
        if ing.use_image_descriptions is not None:
            prefs.use_image_descriptions = ing.use_image_descriptions
        if ing.use_formula_transcription is not None:
            prefs.use_formula_transcription = ing.use_formula_transcription
        if ing.equation_ocr_lib is not None:
            prefs.equation_ocr_lib = ing.equation_ocr_lib
        if prefs.use_vision_model is False:
            prefs.use_image_descriptions = False
            prefs.use_formula_transcription = False
