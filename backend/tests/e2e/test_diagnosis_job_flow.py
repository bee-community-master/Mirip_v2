"""End-to-end flow for upload -> stub diagnosis rejection."""

from __future__ import annotations

from httpx import AsyncClient


async def test_diagnosis_job_flow(client: AsyncClient) -> None:
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
    assert job_response.status_code == 503
    assert job_response.json()["code"] == "DEPENDENCY_ERROR"
    assert job_response.json()["message"] == "Diagnosis model is not ready yet"
