from fastapi import APIRouter
from pydantic import BaseModel

from src.config.constants import DEFAULT_OCR_LIB
from src.config.voice_config import DEFAULT_VOICE_ID, VOICES
from src.utils.api.BaseResponse import BaseResponse

router = APIRouter(prefix="/config", tags=["config"])


class IngestionConfigData(BaseModel):
    equation_ocr_options: list[str]
    default_equation_ocr_lib: str


IngestionConfigResponse = BaseResponse[IngestionConfigData]


@router.get("/ingestion", response_model=IngestionConfigResponse)
async def get_ingestion_config() -> IngestionConfigResponse:
    return IngestionConfigResponse.ok(
        data=IngestionConfigData(
            equation_ocr_options=["local", "llm"],
            default_equation_ocr_lib=DEFAULT_OCR_LIB,
        )
    )


class VoiceOption(BaseModel):
    id: str
    label: str


class VoiceConfigData(BaseModel):
    voices: list[VoiceOption]
    default_voice_id: str


VoiceConfigResponse = BaseResponse[VoiceConfigData]


@router.get("/voices", response_model=VoiceConfigResponse)
async def get_voice_config() -> VoiceConfigResponse:
    return VoiceConfigResponse.ok(
        data=VoiceConfigData(
            voices=[VoiceOption(**voice) for voice in VOICES],
            default_voice_id=DEFAULT_VOICE_ID,
        )
    )


__all__ = ["router"]
