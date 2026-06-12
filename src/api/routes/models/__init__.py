from fastapi import APIRouter
from pydantic import BaseModel

from src.utils.api.BaseResponse import BaseResponse
from src.config.model_config import MODEL_CONFIGS, ReasoningEffort


router = APIRouter(prefix="/models", tags=["models"])


class ModelPicker(BaseModel):
    id: str
    model: str
    provider: str
    context_window: int
    reasoning_effort: str | None


class ModelPickers(BaseModel):
    models: list[ModelPicker]
    reasoning_efforts: list[str | None]


ModelPickerResponse = BaseResponse[ModelPickers]


@router.get("/", response_model=ModelPickerResponse)
async def get_model_picker_presets():
    models = []
    for model_id, model in MODEL_CONFIGS.items():
        default_effort = model.reasoning_effort
        effort_value = (
            default_effort.value
            if default_effort is not None and default_effort != ReasoningEffort.NONE
            else None
        )
        models.append(
            ModelPicker(
                id=model_id,
                model=model.model_name,
                provider=model.provider,
                context_window=model.context_window,
                reasoning_effort=effort_value,
            )
        )

    reasoning_efforts = [
        effort.value for effort in ReasoningEffort if effort != ReasoningEffort.NONE
    ]
    reasoning_efforts.append(None)
    return ModelPickerResponse.ok(
        data=ModelPickers(models=models, reasoning_efforts=reasoning_efforts)
    )
