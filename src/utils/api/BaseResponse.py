from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

Status = Literal["success", "error"]


class BaseResponse(BaseModel, Generic[T]):
    """Uniform API envelope: {status, data, status_code}.

    Subscribe with the payload type so OpenAPI/IDEs see the shape, e.g.
    ``BaseResponse[UserPublic]``. Use ``BaseResponse[None]`` when there is
    no payload.
    """

    status: Status = Field(description="'success' or 'error'")
    data: Optional[T] = Field(default=None, description="Payload data")
    status_code: int = Field(default=200, description="HTTP status code")

    @classmethod
    def ok(
        cls,
        data: Optional[T] = None,
        status_code: int = 200,
    ) -> "BaseResponse[T]":
        return cls(status="success", data=data, status_code=status_code)

    @classmethod
    def error(
        cls,
        *,
        status_code: int,
        data: Optional[T] = None,
    ) -> "BaseResponse[T]":
        return cls(status="error", data=data, status_code=status_code)
