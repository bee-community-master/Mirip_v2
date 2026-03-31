"""Compute Engine Spot VM launch helpers."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.config.settings import ComputeSettings
from mirip_backend.shared.exceptions import DependencyError

INSTANCE_NAME_PREFIX = "mirip-diagnosis"
logger = structlog.get_logger(__name__)


@dataclass(slots=True, frozen=True)
class DiagnosisVmLaunchResult:
    instance_name: str
    zone: str
    launch_state: str
    model_bundle_uri: str
    target_job_id: str

    def metadata_patch(self) -> dict[str, str]:
        return {
            "vm_instance_name": self.instance_name,
            "zone": self.zone,
            "launch_state": self.launch_state,
            "model_bundle_uri": self.model_bundle_uri,
            "target_job_id": self.target_job_id,
        }


class DiagnosisVmLauncher(Protocol):
    async def launch_for_job(
        self,
        *,
        job: DiagnosisJob,
        model_uri: str,
        worker_mode: str,
    ) -> DiagnosisVmLaunchResult: ...

    async def delete_instance(self, *, instance_name: str, zone: str) -> None: ...


def _sanitize_instance_name(job_id: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", job_id.lower()).strip("-")
    suffix = slug[-32:] if slug else "job"
    return f"{INSTANCE_NAME_PREFIX}-{suffix}"[:63].rstrip("-")


@dataclass(slots=True)
class ComputeEngineSpotVmLauncher:
    settings: ComputeSettings

    def build_instance_resource(
        self,
        *,
        instance_name: str,
        job: DiagnosisJob,
        model_uri: str,
        worker_mode: str,
    ) -> dict[str, Any]:
        metadata_items = [
            {"key": "run_once", "value": "true"},
            {"key": "target_job_id", "value": job.id},
            {"key": "worker_mode", "value": worker_mode},
            {"key": "model_uri", "value": model_uri},
            {"key": "instance_name", "value": instance_name},
            {"key": "zone", "value": self.settings.zone or ""},
        ]
        resource: dict[str, Any] = {
            "name": instance_name,
            "labels": {
                "mirip-managed": "true",
                "mirip-job-id": (
                    re.sub(r"[^a-z0-9-]+", "-", job.id.lower())[:63].strip("-") or "job"
                ),
            },
            "metadata": {"items": metadata_items},
        }
        return resource

    async def launch_for_job(
        self,
        *,
        job: DiagnosisJob,
        model_uri: str,
        worker_mode: str,
    ) -> DiagnosisVmLaunchResult:
        if not self.settings.enabled:
            raise DependencyError("Compute Engine launcher is disabled")
        if (
            not self.settings.project_id
            or not self.settings.zone
            or not self.settings.instance_template
        ):
            raise DependencyError("Compute Engine settings are incomplete")

        instance_name = _sanitize_instance_name(job.id)

        def _launch() -> None:
            from google.cloud import compute_v1

            client = compute_v1.InstancesClient()
            request = compute_v1.InsertInstanceRequest(
                project=self.settings.project_id,
                zone=self.settings.zone,
                source_instance_template=self.settings.instance_template,
                instance_resource=compute_v1.Instance(
                    **self.build_instance_resource(
                        instance_name=instance_name,
                        job=job,
                        model_uri=model_uri,
                        worker_mode=worker_mode,
                    )
                ),
            )
            operation = client.insert(request=request)
            operation.result(timeout=self.settings.operation_timeout_seconds)  # type: ignore[no-untyped-call]

        try:
            await asyncio.to_thread(_launch)
        except Exception as exc:
            logger.exception(
                "diagnosis_vm_launch_failed",
                project_id=self.settings.project_id,
                zone=self.settings.zone,
                instance_template=self.settings.instance_template,
                instance_name=instance_name,
                job_id=job.id,
            )
            raise DependencyError("Failed to launch Spot VM for diagnosis job") from exc

        return DiagnosisVmLaunchResult(
            instance_name=instance_name,
            zone=self.settings.zone,
            launch_state="launched",
            model_bundle_uri=model_uri,
            target_job_id=job.id,
        )

    async def delete_instance(self, *, instance_name: str, zone: str) -> None:
        if not self.settings.enabled or not self.settings.project_id:
            return

        def _delete() -> None:
            from google.cloud import compute_v1

            client = compute_v1.InstancesClient()
            request = compute_v1.DeleteInstanceRequest(
                instance=instance_name,
                project=self.settings.project_id,
                zone=zone,
            )
            operation = client.delete(request=request)
            operation.result(timeout=self.settings.operation_timeout_seconds)  # type: ignore[no-untyped-call]

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:
            logger.exception(
                "diagnosis_vm_delete_failed",
                project_id=self.settings.project_id,
                zone=zone,
                instance_name=instance_name,
            )
            raise DependencyError("Failed to delete Spot VM instance") from exc
