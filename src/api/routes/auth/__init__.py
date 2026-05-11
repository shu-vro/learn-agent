from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_session
from src.db.models.user import User
from src.utils.api.BaseResponse import BaseResponse
from src.utils.api.jwt import (
    clear_auth_cookie,
    create_access_token,
    get_current_user,
    set_auth_cookie,
)
from src.utils.argon2_utils import verify_password


router = APIRouter(prefix="/auth", tags=["auth"])


class LoginPayload(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class RegisterPayload(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=255)


class UserPublic(BaseModel):
    id: str
    name: str
    email: EmailStr

    @classmethod
    def from_model(cls, user: User) -> "UserPublic":
        return cls(id=user.id, name=user.name, email=user.email)


UserResponse = BaseResponse[UserPublic]
EmptyResponse = BaseResponse[None]


def _issue_cookie(response: Response, user: User) -> None:
    token = create_access_token({"sub": user.id, "email": user.email})
    set_auth_cookie(response, token)


@router.post("/login", response_model=UserResponse)
async def login(
    payload: LoginPayload,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> UserResponse:
    user = await User.get_by_email(session, payload.email)
    if not user or not verify_password(user.password, payload.password):
        # Same message for both branches to avoid user enumeration.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    _issue_cookie(response, user)
    return UserResponse.ok(data=UserPublic.from_model(user))


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterPayload,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> UserResponse:
    if await User.get_by_email(session, payload.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists",
        )

    user = await User.create(
        session,
        name=payload.name,
        email=payload.email,
        password=payload.password,
    )

    _issue_cookie(response, user)
    return UserResponse.ok(
        data=UserPublic.from_model(user),
        status_code=status.HTTP_201_CREATED,
    )


@router.post("/logout", response_model=EmptyResponse)
async def logout(response: Response) -> EmptyResponse:
    clear_auth_cookie(response)
    return EmptyResponse.ok()


@router.get("/profile", response_model=UserResponse)
async def profile(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.ok(data=UserPublic.from_model(current_user))


__all__ = ["router"]
