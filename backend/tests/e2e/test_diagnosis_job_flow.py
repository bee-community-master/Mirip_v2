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
) -> None:
    upload_response = await client.post(
        "/v1/uploads",
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

    complete_response = await client.post(f"/v1/uploads/{upload_id}/complete")
    assert complete_response.status_code == 200

    job_response = await client.post(
        "/v1/diagnosis/jobs",
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

    status_response = await client.get(f"/v1/diagnosis/jobs/{job_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["job"]["status"] == "succeeded"
    assert status_payload["result"]["job_id"] == job_id

    credential_response = await client.post(
        "/v1/credentials",
        json={
            "result_id": status_payload["result"]["id"],
            "title": "Mirip Visual Diagnosis",
            "visibility": "public",
        },
    )
    assert credential_response.status_code == 200

    portfolio_item_response = await client.post(
        "/v1/profiles/me/portfolio-items",
        json={
            "title": "Portfolio Piece",
            "description": "Generated in e2e flow",
            "asset_upload_id": upload_id,
            "visibility": "public",
        },
    )
    assert portfolio_item_response.status_code == 201
    portfolio_item_id = portfolio_item_response.json()["id"]

    profile_response = await client.put(
        "/v1/profiles/me",
        json={
            "handle": "mirip-e2e",
            "display_name": "Mirip E2E User",
            "bio": "E2E validation profile",
            "visibility": "public",
            "portfolio_item_ids": [portfolio_item_id],
        },
    )
    assert profile_response.status_code == 200

    history_response = await client.get("/v1/diagnosis/history")
    assert history_response.status_code == 200
    history_payload = history_response.json()
    assert history_payload["total"] == 1
    assert history_payload["items"][0]["job_id"] == job_id

    public_profile_response = await client.get("/v1/profiles/mirip-e2e")
    assert public_profile_response.status_code == 200
    assert public_profile_response.json()["portfolio_items"][0]["id"] == portfolio_item_id
