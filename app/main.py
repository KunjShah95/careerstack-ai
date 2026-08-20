"""FastAPI app: routes only, zero logic. Every route validates its input,
calls a service, and returns the result -- see CLAUDE.md's routing rule.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.services.analyze import NeedsOcrError, RoleProfileUnavailableError, run_analysis
from app.services.extraction.text_extract import ExtractionError
from app.store import load_json

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

ANALYSES_COLLECTION = "analyses"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.demo_mode:
        print("=" * 64)
        print("  DEMO_MODE IS ON")
        print("  Role profile mining will never hit the network -- only")
        print("  cached profiles are served. Run scripts/prep_demo.py first")
        print("  if the profiles you need aren't cached yet.")
        print("=" * 64)
    yield


app = FastAPI(title="CareerStack AI — Resume Analysis", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safety net for every route: never let a raw traceback reach the
    client. Explicitly-raised HTTPExceptions (with their own plain-
    language detail messages) bypass this and are handled by FastAPI's
    default HTTPException handling instead.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong while processing your request. Please try again."},
    )


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "demo_mode": settings.demo_mode}


@app.get("/api/analyze/{analysis_id}")
async def get_analysis(analysis_id: str) -> dict:
    analysis = load_json(ANALYSES_COLLECTION, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="No analysis found with that ID.")
    return analysis


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    role: str = Form(...),
    location: str = Form(...),
) -> dict:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{extension or filename}'. "
                "Upload a PDF or Word document (.pdf, .docx, .doc)."
            ),
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File is too large. Please upload a resume under 5MB.",
        )

    role = role.strip()
    location = location.strip()
    if not role or not location:
        raise HTTPException(status_code=400, detail="Both role and location are required.")

    try:
        return await run_analysis(file_bytes, filename, role, location)
    except NeedsOcrError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ExtractionError as exc:
        logger.warning("Extraction failed for %s: %s", filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RoleProfileUnavailableError as exc:
        # run_analysis already tried get_or_mine and the most-recent-
        # cached-profile-for-any-role fallback; there is truly nothing to
        # score against.
        logger.warning("No role profile available at all for role=%r location=%r: %s", role, location, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        # DEMO_MODE with no cached parse for this exact file -- see
        # parse_cache.py's get_or_parse. The only remaining source of a
        # bare RuntimeError reaching here; get_or_mine's own RuntimeError
        # is already absorbed inside run_analysis's fallback handling.
        logger.warning("Resume parse cache miss in DEMO_MODE for %s: %s", filename, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# Registered last: a mount on "/" matches every path not already claimed
# by a route above it, so the API routes must be declared first or this
# would shadow them. check_dir=False since static/index.html (build order
# step 10) doesn't exist yet -- this only 404s on missing files, it
# doesn't fail at import time.
app.mount("/", StaticFiles(directory="static", html=True, check_dir=False), name="static")
