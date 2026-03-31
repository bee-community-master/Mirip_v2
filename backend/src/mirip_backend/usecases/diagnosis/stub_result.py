"""Stub diagnosis output helpers for pre-model deployments."""

from __future__ import annotations

import hashlib
from random import Random

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.worker.inference.types import InferenceOutput

_UNIVERSITY_OPTIONS = {
    "visual_design": [
        ("홍익대학교", "시각디자인과"),
        ("국민대학교", "시각디자인학과"),
        ("건국대학교", "커뮤니케이션디자인학과"),
    ],
    "industrial_design": [
        ("국민대학교", "공업디자인학과"),
        ("KAIST", "산업디자인학과"),
        ("건국대학교", "산업디자인학과"),
    ],
    "fine_art": [
        ("서울대학교", "서양화과"),
        ("홍익대학교", "회화과"),
        ("중앙대학교", "서양화과"),
    ],
    "craft": [
        ("홍익대학교", "도예유리과"),
        ("국민대학교", "공예학과"),
        ("서울대학교", "공예과"),
    ],
}


def _make_rng(job: DiagnosisJob) -> Random:
    digest = hashlib.sha256(
        f"{job.id}:{job.department}:{','.join(job.upload_ids)}".encode()
    ).hexdigest()
    return Random(int(digest[:16], 16))


def build_stub_inference_output(job: DiagnosisJob) -> InferenceOutput:
    """Return a plausible looking diagnosis result without model execution."""

    rng = _make_rng(job)
    base = rng.uniform(62.0, 91.0)
    scores = {
        "composition": round(max(45.0, min(99.0, base + rng.uniform(-6.0, 4.5))), 1),
        "technique": round(max(45.0, min(99.0, base + rng.uniform(-5.0, 6.0))), 1),
        "creativity": round(max(45.0, min(99.0, base + rng.uniform(-4.0, 7.5))), 1),
        "completeness": round(max(45.0, min(99.0, base + rng.uniform(-7.0, 5.0))), 1),
    }
    average = sum(scores.values()) / len(scores)
    if average >= 86:
        tier = "S"
    elif average >= 78:
        tier = "A"
    elif average >= 70:
        tier = "B"
    else:
        tier = "C"

    universities = _UNIVERSITY_OPTIONS.get(job.department, _UNIVERSITY_OPTIONS["visual_design"])
    probabilities = []
    top_probability = round(rng.uniform(0.62, 0.89), 3)
    remaining = max(0.04, 1.0 - top_probability)
    second_probability = round(max(0.05, remaining * rng.uniform(0.52, 0.7)), 3)
    third_probability = round(max(0.05, 1.0 - top_probability - second_probability), 3)
    for index, (university, department) in enumerate(universities):
        probability = [top_probability, second_probability, third_probability][index]
        probabilities.append(
            {
                "university": university,
                "department": department,
                "probability": probability,
            }
        )

    feedback: dict[str, object] | None = None
    if job.include_feedback:
        strongest_axis = max(scores, key=scores.get)
        weakest_axis = min(scores, key=scores.get)
        labels = {
            "composition": "구성",
            "technique": "기술",
            "creativity": "창의성",
            "completeness": "완성도",
        }
        feedback = {
            "strengths": [
                f"{labels[strongest_axis]}이 현재 기준에서 가장 안정적입니다.",
                "전체 인상에서 입시 포트폴리오 톤이 잘 유지되고 있습니다.",
            ],
            "improvements": [
                f"{labels[weakest_axis]} 보완 여지가 큽니다.",
                "아이디어 전개를 한 단계 더 밀어주면 상위권 대학 적합도가 올라갑니다.",
            ],
            "overall": f"{job.department} 기준으로 {tier} 티어 예측입니다.",
        }

    summary = (
        f"{len(job.upload_ids)}개 작품 비교 분석이 완료되었습니다."
        if job.job_type == "compare"
        else "단일 작품 진단이 완료되었습니다."
    )
    return InferenceOutput(
        tier=tier,
        scores=scores,
        probabilities=probabilities,
        feedback=feedback,
        summary=summary,
    )
