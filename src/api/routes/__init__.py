from fastapi import APIRouter, Depends
from src.api.routes.auth import router as auth_router
from src.api.routes.chat import router as chat_router
from src.api.routes.projects import router as projects_router
from src.api.routes.config import router as config_router
from src.api.routes.models import router as models_router
from src.utils.api.BaseResponse import BaseResponse
from src.utils.api.jwt import try_get_current_user
from pydantic import BaseModel
from src.db import engine
from sqlalchemy import text
from src.lib.redis_lib import redis_client
from fakeredis import FakeRedis

# main router
router = APIRouter()


class HealthData(BaseModel):
    db: bool
    redis: bool


HealthCheckResponse = BaseResponse[HealthData]


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    try:
        async with engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return HealthCheckResponse.error(
            status_code=500, data=HealthData(db=False, redis=True)
        )
    try:
        if isinstance(redis_client, FakeRedis):
            return HealthCheckResponse.error(
                status_code=500, data=HealthData(redis=False, db=True)
            )
    except Exception:
        return HealthCheckResponse.error(
            status_code=500, data=HealthData(redis=True, db=True)
        )
    return HealthCheckResponse.ok(data=HealthData(db=True, redis=True))


router.include_router(auth_router, prefix="/v1")
router.include_router(config_router, prefix="/v1")
router.include_router(
    chat_router, prefix="/v1", dependencies=[Depends(try_get_current_user)]
)
router.include_router(
    projects_router, prefix="/v1", dependencies=[Depends(try_get_current_user)]
)
router.include_router(
    models_router, prefix="/v1", dependencies=[Depends(try_get_current_user)]
)

__all__ = ["router"]
