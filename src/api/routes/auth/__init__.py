from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_session
from src.db.models.preferences import Preferences
from src.db.models.user import User
from src.schemas.preferences import (
    UserPreferencesPublic,
    UserPreferencesUpdate,
    apply_preferences_update,
)
from src.utils.api.BaseResponse import BaseResponse
from src.utils.api.jwt import (
    clear_auth_cookie,
    create_access_token,
    get_current_user,
    set_auth_cookie,
)
from src.utils.argon2_utils import hash_password, verify_password


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


class UserProfile(BaseModel):
    id: str
    name: str
    email: EmailStr
    preferences: UserPreferencesPublic

    @classmethod
    async def from_user(cls, user: User, session: AsyncSession) -> "UserProfile":
        prefs = await Preferences.get_or_create(session, user.id)
        return cls(
            id=user.id,
            name=user.name,
            email=user.email,
            preferences=UserPreferencesPublic.from_model(prefs),
        )


class ProfileUpdatePayload(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    email: EmailStr | None = None


class PasswordUpdatePayload(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=255)


UserResponse = BaseResponse[UserPublic]
ProfileResponse = BaseResponse[UserProfile]
EmptyResponse = BaseResponse[None]
PreferencesResponse = BaseResponse[UserPreferencesPublic]


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


@router.get("/profile", response_model=ProfileResponse)
async def profile(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ProfileResponse:
    data = await UserProfile.from_user(current_user, session)
    return ProfileResponse.ok(data=data)


@router.patch("/profile", response_model=ProfileResponse)
async def update_profile(
    payload: ProfileUpdatePayload,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ProfileResponse:
    if payload.email is not None and payload.email != current_user.email:
        existing = await User.get_by_email(session, payload.email)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already exists",
            )
        current_user.email = payload.email
    if payload.name is not None:
        current_user.name = payload.name
    session.add(current_user)
    await session.commit()
    await session.refresh(current_user)
    data = await UserProfile.from_user(current_user, session)
    return ProfileResponse.ok(data=data)


@router.patch("/password", response_model=EmptyResponse)
async def update_password(
    payload: PasswordUpdatePayload,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> EmptyResponse:
    if not verify_password(current_user.password, payload.current_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )
    current_user.password = hash_password(payload.new_password)
    session.add(current_user)
    await session.commit()
    return EmptyResponse.ok()


@router.patch("/preferences", response_model=PreferencesResponse)
async def update_preferences(
    payload: UserPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> PreferencesResponse:
    prefs = await Preferences.get_or_create(session, current_user.id)
    apply_preferences_update(prefs, payload)
    session.add(prefs)
    await session.commit()
    await session.refresh(prefs)
    return PreferencesResponse.ok(data=UserPreferencesPublic.from_model(prefs))


__all__ = ["router"]
