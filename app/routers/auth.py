"""Authentication routes: register, login, whoami.

All logic lives in app/services/auth.py -- this module only validates
input (via the Pydantic request models), calls the service, and shapes
the HTTP response. See CLAUDE.md's routing rule.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_current_user
from app.models.user import TokenPair, UserCreate, UserLogin, UserPublic
from app.services import auth as auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", status_code=201, response_model=TokenPair)
async def register(payload: UserCreate) -> TokenPair:
    try:
        user = auth_service.create_user(payload.email, payload.password, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    token = auth_service.create_access_token(user.user_id)
    return TokenPair(access_token=token, user=user)


@router.post("/login", response_model=TokenPair)
async def login(payload: UserLogin) -> TokenPair:
    user = auth_service.authenticate(payload.email, payload.password)
    if user is None:
        # Deliberately generic: never reveal whether the email or the
        # password was the wrong one.
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = auth_service.create_access_token(user.user_id)
    return TokenPair(access_token=token, user=user)


@router.get("/me", response_model=UserPublic)
async def me(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return current_user
