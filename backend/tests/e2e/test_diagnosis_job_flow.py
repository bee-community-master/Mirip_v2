"""End-to-end flow for upload -> diagnosis job -> worker -> result."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient

from mirip_backend.worker.inference.service import GpuInferenceService
from mirip_backend.worker.poller import JobPoller
from mirip_backend.worker.result_writer import DiagnosisResultWriter


async def test_diagnosis_job_flow(
    client: AsyncClient,
    app: FastAPI,
    auth_headers: dict[str, str],
) -> None:
    upload_response = await client.post(
        "/v1/uploads",
        headers=auth_headers,
        json={
            "filename": "portfolio.png",
            "content_type": "image/png",
            "size_bytes": 1024,
            "category": "diagnosis",
        },
    )
    assert upload_response.status_code == 201
    upload_payload = upload_response.json()
    upload_id = upload_payload["upload"]["id"]

    job_response = await client.post(
        "/v1/diagnosis/jobs",
        headers=auth_headers,
        json={
            "upload_ids": [upload_id],
            "job_type": "evaluate",
            "department": "visual_design",
            "include_feedback": True,
            "language": "ko",
        },
    )
    assert job_response.status_code == 201
    job_id = job_response.json()["id"]

    container = app.state.container
    poller = JobPoller(
        worker_id="test-worker",
        queue=container.job_queue,
        inference_service=GpuInferenceService(mode="stub", model_uri=None),
        result_writer=DiagnosisResultWriter(container.diagnosis_result_repository),
    )
    await poller.process_once()

    status_response = await client.get(f"/v1/diagnosis/jobs/{job_id}", headers=auth_headers)
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["job"]["status"] == "succeeded"
    assert status_payload["result"]["job_id"] == job_id

    history_response = await client.get("/v1/diagnosis/history", headers=auth_headers)
    assert history_response.status_code == 200
    history_payload = history_response.json()
    assert history_payload["total"] == 1
    assert history_payload["items"][0]["job_id"] == job_id
