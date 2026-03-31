"""Tests for Compute Engine launch payload generation."""

from __future__ import annotations

from datetime import UTC, datetime

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.compute.service import ComputeEngineSpotVmLauncher
from mirip_backend.infrastructure.config.settings import ComputeSettings
from mirip_backend.shared.enums import JobStatus


def test_compute_launcher_builds_metadata_driven_instance_resource() -> None:
    launcher = ComputeEngineSpotVmLauncher(
        ComputeSettings(
            enabled=True,
            project_id="mirip-v2",
            zone="asia-northeast3-b",
            instance_template="global/instanceTemplates/mirip-worker",
            service_account="worker@mirip-v2.iam.gserviceaccount.com",
            subnetwork="regions/asia-northeast3/subnetworks/default",
        )
    )
    job = DiagnosisJob(
        id="job-demo",
        user_id="user-1",
        upload_ids=["upl-1"],
        job_type="evaluate",
        department="visual_design",
        include_feedback=True,
        theme=None,
        language="ko",
        status=JobStatus.QUEUED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    resource = launcher.build_instance_resource(
        instance_name="mirip-diagnosis-job-demo",
        job=job,
        model_uri="gs://mirip-v2-assets/models/vitl-bundle",
        worker_mode="cpu_onnx",
    )

    metadata = {item["key"]: item["value"] for item in resource["metadata"]["items"]}
    assert metadata["run_once"] == "true"
    assert metadata["target_job_id"] == "job-demo"
    assert metadata["worker_mode"] == "cpu_onnx"
    assert metadata["model_uri"] == "gs://mirip-v2-assets/models/vitl-bundle"
    assert metadata["instance_name"] == "mirip-diagnosis-job-demo"
    assert metadata["zone"] == "asia-northeast3-b"
    assert resource["labels"]["mirip-managed"] == "true"
    assert "service_accounts" not in resource
    assert "network_interfaces" not in resource
