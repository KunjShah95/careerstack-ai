"""Pydantic schemas for authentication.

UserPublic never includes the password hash -- that's the entire reason it
exists as a model distinct from the stored user record.
"""

from pydantic import BaseModel, EmailStr, Field

# bcrypt hashes at most 72 bytes of the password and raises ValueError
# beyond that rather than silently truncating (a deliberate bcrypt 4.x+
# safety change -- silent truncation would let two different long
# passwords collide on the same hash). Capping it here turns an over-long
# password into an ordinary 422 validation error instead of an unhandled
# exception out of hash_password().
PASSWORD_MAX_LENGTH = 72


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    email: EmailStr
    password: str = Field(min_length=8, max_length=PASSWORD_MAX_LENGTH)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    user_id: str
    email: str
    name: str
    created_at: str


class TokenPair(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
