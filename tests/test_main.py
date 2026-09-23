"""Tests for the FastAPI routes in main.py.

Covers only the parts of /api/analyze that don't need a live Groq or
Adzuna call (input validation, needs_ocr), plus the two simple GET
routes -- consistent with the rest of this suite staying network-free.
The full pipeline (parse_resume + get_or_mine + scoring) was verified
manually against a real fixture; see the conversation, not this file.

/api/analyze and /api/analyze/{id} require auth now -- see test_auth.py
for the auth flow itself; here each test just registers its own throwaway
user (a fresh, unique email per test, so tests never collide with each
other's leftover ./data/users/ state across runs) to get past the
Depends(get_current_user) gate.
"""

import uuid

import fitz
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _auth_headers() -> dict:
    email = f"test-{uuid.uuid4().hex}@example.com"
    response = client.post(
        "/api/auth/register", json={"name": "Test User", "email": email, "password": "testpass123"}
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["demo_mode"], bool)


def test_get_missing_analysis_returns_404():
    response = client.get("/api/analyze/does-not-exist", headers=_auth_headers())
    assert response.status_code == 404
    assert "detail" in response.json()


def test_analyze_requires_auth():
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.pdf", b"%PDF-1.4", "application/pdf")},
        data={"role": "backend developer", "location": "India"},
    )
    assert response.status_code == 401


def test_analyze_rejects_unsupported_extension():
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.txt", b"hello", "text/plain")},
        data={"role": "backend developer", "location": "India"},
        headers=_auth_headers(),
    )
    assert response.status_code == 400
    assert ".txt" in response.json()["detail"]


def test_analyze_rejects_oversized_file():
    oversized = b"a" * (5 * 1024 * 1024 + 1)
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.pdf", oversized, "application/pdf")},
        data={"role": "backend developer", "location": "India"},
        headers=_auth_headers(),
    )
    assert response.status_code == 400
    assert "5MB" in response.json()["detail"]


def test_analyze_rejects_blank_role_or_location():
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.pdf", b"%PDF-1.4", "application/pdf")},
        data={"role": "  ", "location": "India"},
        headers=_auth_headers(),
    )
    assert response.status_code == 400
    assert "required" in response.json()["detail"].lower()


def test_analyze_returns_422_for_a_blank_page_pdf():
    """No selectable text at all -- needs_ocr should fire with the exact
    plain-language message specified for this case.
    """
    doc = fitz.open()
    doc.new_page()
    blank_pdf_bytes = doc.tobytes()
    doc.close()

    response = client.post(
        "/api/analyze",
        files={"file": ("blank.pdf", blank_pdf_bytes, "application/pdf")},
        data={"role": "backend developer", "location": "India"},
        headers=_auth_headers(),
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "couldn't find any selectable text" in detail
    assert "export a text-based PDF" in detail
