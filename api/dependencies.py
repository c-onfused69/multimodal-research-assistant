"""FastAPI dependencies (auth, rate limiting)."""
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class UserSession:
    def __init__(self, user_id: str, groups: list[str]):
        self.user_id = user_id
        self.groups = groups


async def get_current_user(api_key: str = Security(api_key_header)) -> UserSession:
    # Hardcoded dummy auth for demonstration
    if api_key == "secret-admin-key":
        return UserSession(user_id="admin", groups=["public", "internal", "exec"])
    elif api_key == "secret-user-key":
        return UserSession(user_id="user1", groups=["public"])
    else:
        # Fallback to public if no key (or you can raise 401)
        return UserSession(user_id="anonymous", groups=["public"])


async def check_rate_limit(user: UserSession = Depends(get_current_user)):
    # Dummy rate limit
    if user.user_id == "anonymous":
        # e.g., check redis for IP limits
        pass
    return user
