from fastapi import APIRouter
from pydantic import BaseModel

from src.utils.api.BaseResponse import BaseResponse
from src.config.model_config import MODEL_CONFIGS, ReasoningEffort


router = APIRouter(prefix="/models", tags=["models"])


class ModelPicker(BaseModel):
    model: str
    provider: str
    context_window: int
    reasoning_effort: ReasoningEffort | None


class ModelPickers(BaseModel):
    models: list[ModelPicker]
    reasoning_efforts: list[ReasoningEffort]


ModelPickerResponse = BaseResponse[ModelPickers]


@router.get("/", response_model=ModelPickerResponse)
async def get_model_picker_presets():
    models = []
    reasoning_efforts = []
    for model in MODEL_CONFIGS.values():
        models.append(
            ModelPicker(
                model=model.model_name,
                provider=model.provider,
                context_window=model.context_window,
                reasoning_effort=model.reasoning_effort,
            )
        )

    reasoning_efforts = list(ReasoningEffort)
    return ModelPickerResponse.ok(
        data=ModelPickers(models=models, reasoning_efforts=reasoning_efforts)
    )
