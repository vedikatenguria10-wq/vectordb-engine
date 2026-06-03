"""Authentication routes: register, login, and current user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import (
    create_access_token,
    get_current_user,
    get_document_store,
    hash_password,
    verify_password,
)
from db.document_store import DocumentStore
from db.schema import User

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    """Registration payload with username and password."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    """Login payload with username and password."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class RegisterResponse(BaseModel):
    """Response after successful registration."""

    message: str
    id: str


class LoginResponse(BaseModel):
    """OAuth2-style token response."""

    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    """Public view of the authenticated user."""

    id: str
    username: str


@router.post("/register", response_model=RegisterResponse)
def register(
    body: RegisterRequest,
    store: DocumentStore = Depends(get_document_store),
) -> RegisterResponse:
    """Register a new user if the username is not already taken."""
    if store.get_user_by_username(body.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
    )
    store.insert_user(user)
    return RegisterResponse(message="User registered successfully", id=user.id)


@router.post("/login", response_model=LoginResponse)
def login(
    body: LoginRequest,
    store: DocumentStore = Depends(get_document_store),
) -> LoginResponse:
    """Authenticate and return a JWT access token."""
    user = store.get_user_by_username(body.username)
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token({"sub": user.username})
    return LoginResponse(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=MeResponse)
def me(current_user: User = Depends(get_current_user)) -> MeResponse:
    """Return the currently authenticated user's id and username."""
    return MeResponse(id=current_user.id, username=current_user.username)
