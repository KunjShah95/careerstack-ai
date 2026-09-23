"""Job discovery routes: validate input, call the service, return a
response. Zero logic -- the pipeline lives in app/services/discovery.py,
per CLAUDE.md's routing rule.

Every route here is authenticated, and every lookup is scoped to the
calling user. A run that belongs to someone else 404s rather than 403s:
a 403 would confirm the run id is real.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.dependencies import get_current_user
from app.models.job import DiscoveryRunRequest
from app.models.user import UserPublic
from app.services import discovery as discovery_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["discovery"])

RUN_NOT_FOUND = "No discovery run found with that ID."
JOB_NOT_FOUND = "No job with that fingerprint in this run."


def _public_run(run: dict) -> dict:
    """The client-facing shape of a run -- user_id stays server-side."""
    return {
        "run_id": run["run_id"],
        "analysis_id": run["analysis_id"],
        "status": run["status"],
        "target": run["target"],
        "stats": run["stats"],
        "widened": run["widened"],
        "queries": run.get("queries", []),
        "jobs": run.get("jobs", []),
        "error": run.get("error"),
        "created_at": run.get("created_at"),
        "completed_at": run.get("completed_at"),
    }


@router.post("/run")
async def start_run(
    payload: DiscoveryRunRequest,
    background_tasks: BackgroundTasks,
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """Create a pending run and kick the pipeline off in the background.

    Returns immediately with the run_id; the client polls
    GET /api/discovery/{run_id} for progress. Ownership and resume
    availability are validated synchronously so a doomed run fails here
    with a real status code rather than a moment later in the background.
    """
    try:
        run = discovery_service.create_run(payload.analysis_id, current_user.user_id)
    except discovery_service.AnalysisNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except discovery_service.ResumeUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background_tasks.add_task(discovery_service.execute_run, run["run_id"])
    return {"run_id": run["run_id"], "status": run["status"]}


@router.get("/{run_id}")
async def get_run(
    run_id: str, current_user: UserPublic = Depends(get_current_user)
) -> dict:
    run = discovery_service.load_run(run_id, current_user.user_id)
    if run is None:
        raise HTTPException(status_code=404, detail=RUN_NOT_FOUND)
    return _public_run(run)


@router.get("/{run_id}/jobs/{job_fingerprint}")
async def get_run_job(
    run_id: str,
    job_fingerprint: str,
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """One job, re-scored against this posting's own description.

    The score is recomputed here rather than read back from the stored run
    -- see rescore_job_against_posting.
    """
    run = discovery_service.load_run(run_id, current_user.user_id)
    if run is None:
        raise HTTPException(status_code=404, detail=RUN_NOT_FOUND)

    try:
        job = discovery_service.rescore_job_against_posting(run, job_fingerprint)
    except discovery_service.ResumeUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if job is None:
        raise HTTPException(status_code=404, detail=JOB_NOT_FOUND)
    return job
