from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Dict

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.llm import llm_enabled
from app.pipeline import run_search

app = FastAPI(title="Housing Search")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

JOBS: Dict[str, Dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


def _init_job() -> str:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "progress": 0.0,
            "message": "Starting search",
            "result": None,
            "error": None,
            "updated_at": time.time(),
        }
    return job_id


def _update_job(job_id: str, **updates: object) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id].update(updates)
        JOBS[job_id]["updated_at"] = time.time()


def _get_job(job_id: str) -> Dict[str, object] | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "llm_available": llm_enabled(),
            "example_prompt": (
                "2.5 to 3.5 rooms, surrounded by trees or with wide sweeping views, quiet, "
                "away from the street but close to public transport, village-like feel, "
                "lots of natural daylight, large balcony, roof terrace if possible, "
                "in a green and leafy suburb near Kreis 7 in Zürich and within 20 minutes walk to the lake"
            ),
            "min_price_chf": 1500,
            "max_price_chf": 3000,
        },
    )


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    prompt: str = Form(...),
    max_listings: int = Form(10),
    min_price_chf: str | None = Form(None),
    max_price_chf: str | None = Form(None),
    openai_api_key: str | None = Form(None),
) -> HTMLResponse:
    min_price = _parse_optional_int(min_price_chf)
    max_price = _parse_optional_int(max_price_chf)
    api_key = openai_api_key.strip() if openai_api_key else None
    result = run_search(
        prompt,
        use_llm=True,
        max_listings=max_listings,
        min_price_chf=min_price,
        max_price_chf=max_price,
        llm_api_key=api_key,
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "llm_available": llm_enabled(api_key),
            "example_prompt": prompt,
            "min_price_chf": min_price,
            "max_price_chf": max_price,
        },
    )


@app.post("/search/start")
async def start_search(
    prompt: str = Form(...),
    max_listings: int = Form(10),
    min_price_chf: str | None = Form(None),
    max_price_chf: str | None = Form(None),
    openai_api_key: str | None = Form(None),
) -> JSONResponse:
    job_id = _init_job()
    min_price = _parse_optional_int(min_price_chf)
    max_price = _parse_optional_int(max_price_chf)
    api_key = openai_api_key.strip() if openai_api_key else None

    def progress_callback(message: str, progress: float) -> None:
        _update_job(job_id, message=message, progress=progress)

    def run_job() -> None:
        try:
            result = run_search(
                prompt,
                use_llm=True,
                max_listings=max_listings,
                min_price_chf=min_price,
                max_price_chf=max_price,
                llm_api_key=api_key,
                progress_callback=progress_callback,
            )
            _update_job(job_id, status="done", progress=1.0, message="Done", result=result)
        except Exception as exc:
            _update_job(job_id, status="error", error=str(exc), message="Search failed")

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
    return JSONResponse({"job_id": job_id})


@app.get("/search/progress")
async def search_progress(job_id: str) -> StreamingResponse:
    async def event_stream():
        last_progress = None
        while True:
            job = _get_job(job_id)
            if job is None:
                payload = {"status": "error", "message": "Unknown job"}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                break
            status = job.get("status")
            progress = job.get("progress", 0.0)
            message = job.get("message", "")
            if progress != last_progress:
                payload = {"status": status, "progress": progress, "message": message}
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
                last_progress = progress
            if status in {"done", "error"}:
                payload = {"status": status, "message": job.get("error") or message}
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/search/result/{job_id}", response_class=HTMLResponse)
async def search_result(request: Request, job_id: str) -> HTMLResponse:
    job = _get_job(job_id)
    if not job:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "llm_available": llm_enabled(),
                "example_prompt": "",
                "min_price_chf": 1500,
                "max_price_chf": 3000,
            },
        )
    result = job.get("result")
    if job.get("status") != "done" or result is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "llm_available": llm_enabled(),
                "example_prompt": "",
                "min_price_chf": 1500,
                "max_price_chf": 3000,
            },
        )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "llm_available": llm_enabled(),
            "example_prompt": result.filters.raw_prompt,
            "min_price_chf": result.filters.min_price_chf,
            "max_price_chf": result.filters.max_price_chf,
        },
    )
