"""FastAPI dependencies shared across routers.

get_current_user is HTTP-layer glue (raises HTTPException, parses the
Bearer header) -- kept separate from app/services/auth.py, whose functions
are framework-agnostic and know nothing about HTTP.
"""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models.user import UserPublic
from app.services import auth as auth_service

# auto_error=False: HTTPBearer's default raises its own 403 on a missing
# header. We want 401 for that case (and for a malformed one -- wrong
# scheme, no token after "Bearer" -- HTTPBearer also just returns None for
# those when auto_error is off), so it's handled explicitly below instead.
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserPublic:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_id = auth_service.decode_token(credentials.credentials)
    except auth_service.ExpiredTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except auth_service.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = auth_service.get_user_by_id(user_id)
    if user is None:
        # Token is well-formed and unexpired but names a user that no
        # longer exists. Same generic message as any other bad-credentials
        # case -- nothing useful to distinguish for the client.
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return user
