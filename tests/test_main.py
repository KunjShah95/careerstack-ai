"""Tests for the FastAPI routes in main.py.

Covers only the parts of /api/analyze that don't need a live Groq or
Adzuna call (input validation, needs_ocr), plus the two simple GET
routes -- consistent with the rest of this suite staying network-free.
The full pipeline (parse_resume + get_or_mine + scoring) was verified
manually against a real fixture; see the conversation, not this file.
"""

import fitz
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["demo_mode"], bool)


def test_get_missing_analysis_returns_404():
    response = client.get("/api/analyze/does-not-exist")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_analyze_rejects_unsupported_extension():
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.txt", b"hello", "text/plain")},
        data={"role": "backend developer", "location": "India"},
    )
    assert response.status_code == 400
    assert ".txt" in response.json()["detail"]


def test_analyze_rejects_oversized_file():
    oversized = b"a" * (5 * 1024 * 1024 + 1)
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.pdf", oversized, "application/pdf")},
        data={"role": "backend developer", "location": "India"},
    )
    assert response.status_code == 400
    assert "5MB" in response.json()["detail"]


def test_analyze_rejects_blank_role_or_location():
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.pdf", b"%PDF-1.4", "application/pdf")},
        data={"role": "  ", "location": "India"},
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
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "couldn't find any selectable text" in detail
    assert "export a text-based PDF" in detail
