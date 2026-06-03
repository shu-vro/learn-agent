from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.utils.api.BaseResponse import BaseResponse


def _error_response(*, status_code: int, data=None) -> JSONResponse:
    body = BaseResponse.error(status_code=status_code, data=data)
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return _error_response(status_code=exc.status_code, data=None or exc.detail)


async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error_response(status_code=422, data=exc.errors())


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
