"""Tests for authentication: register, login, protected routes, and
per-user analysis ownership.

Each test registers its own throwaway user with a fresh, unique email, so
tests never collide with each other's leftover ./data/users/ state across
runs (this project persists to disk, not an in-memory test DB).
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

import app.services.analyze as analyze_module
import app.services.auth as auth_service
from app.config import settings
from app.main import app
from app.models.resume import ContactInfo, ParsedResume
from app.services.analyze import run_analysis
from app.store import save_json

client = TestClient(app)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex}@example.com"


def _register(email: str, password: str = "testpass123", name: str = "Test User") -> dict:
    response = client.post("/api/auth/register", json={"name": name, "email": email, "password": password})
    assert response.status_code == 201, response.text
    return response.json()


def test_register_login_and_access_protected_route():
    email = _unique_email()
    register_body = _register(email)
    assert register_body["token_type"] == "bearer"
    assert register_body["user"]["email"] == email
    assert "user_id" in register_body["user"]

    login_response = client.post("/api/auth/login", json={"email": email, "password": "testpass123"})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    me_response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["email"] == email


def test_duplicate_email_returns_409():
    email = _unique_email()
    _register(email)

    response = client.post(
        "/api/auth/register", json={"name": "Test User", "email": email, "password": "anotherpass123"}
    )
    assert response.status_code == 409


def test_wrong_password_returns_401_with_generic_message():
    email = _unique_email()
    _register(email)

    response = client.post("/api/auth/login", json={"email": email, "password": "wrong-password"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_unknown_email_returns_the_same_generic_401_message():
    """Never reveal whether the email itself is registered -- login for a
    nonexistent account must look identical to a wrong password.
    """
    response = client.post("/api/auth/login", json={"email": _unique_email(), "password": "whatever123"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_no_token_returns_401():
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_expired_token_returns_401():
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    expired_token = jwt.encode(
        {"sub": "some-user-id", "iat": past - timedelta(minutes=5), "exp": past},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_invalid_token_returns_401():
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def _stub_resume() -> ParsedResume:
    return ParsedResume(
        contact=ContactInfo(name="Someone"),
        skills=["Python"],
        sections_found=["skills"],
        total_experience_months=12,
    )


def test_user_a_cannot_fetch_user_bs_analysis(monkeypatch):
    """Deliberately 404, not 403 -- see main.py's get_analysis. Network-
    free: get_or_parse/get_or_mine stubbed the same way test_ats.py and
    test_analyze_degraded.py already do.
    """
    monkeypatch.setattr(analyze_module, "get_or_parse", lambda file_bytes, text: _stub_resume())

    async def _mining_succeeded(role, location):
        return {
            "role": "backend developer",
            "location": "India",
            "postings_sampled": 40,
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "skill_frequencies": {"python": {"frequency": 0.5, "count": 20}},
            "requirement_sentences": [],
            "median_experience_years": 3.0,
            "source_ids": [],
            "location_fallback": None,
        }

    monkeypatch.setattr(analyze_module, "get_or_mine", _mining_succeeded)

    file_bytes = (FIXTURES_DIR / "demo_after.pdf").read_bytes()
    user_b_analysis = asyncio.run(
        run_analysis(file_bytes, "demo_after.pdf", "backend developer", "India", "user-b-id")
    )

    user_a_token = _register(_unique_email())["access_token"]

    response = client.get(
        f"/api/analyze/{user_b_analysis['analysis_id']}",
        headers={"Authorization": f"Bearer {user_a_token}"},
    )
    assert response.status_code == 404


def test_password_hash_never_appears_in_any_response_body():
    email = _unique_email()
    register_body = _register(email)
    assert "password_hash" not in str(register_body)
    assert "password" not in register_body["user"]

    login_response = client.post("/api/auth/login", json={"email": email, "password": "testpass123"})
    login_body = login_response.json()
    assert "password_hash" not in str(login_body)
    assert "password" not in login_body["user"]

    token = login_body["access_token"]
    me_body = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert "password_hash" not in str(me_body)
    assert "password" not in me_body


def test_legacy_user_record_without_a_name_falls_back_to_email_local_part():
    """Existing users in ./data/users predate the name field. Reading one
    must not 500 -- it should fall back to something reasonable instead.
    """
    user_id = uuid.uuid4().hex
    email = f"legacy-{uuid.uuid4().hex}@example.com"
    save_json(
        auth_service.COLLECTION,
        auth_service._email_key(email),
        {
            "user_id": user_id,
            "email": email,
            "password_hash": auth_service.hash_password("whatever123"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            # deliberately no "name" key here
        },
    )

    user = auth_service.get_user_by_id(user_id)

    assert user is not None
    assert user.name == email.split("@", 1)[0]
