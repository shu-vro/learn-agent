from fastapi import APIRouter
from pydantic import BaseModel

from src.config.constants import DEFAULT_OCR_LIB
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


__all__ = ["router"]
