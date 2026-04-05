#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import plistlib
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(SCRIPT_DIR))

TRAIN_ROOT = "train"
TRAINING_DIR = f"{TRAIN_ROOT}/training"
TRAIN_DATA_DIR = f"{TRAIN_ROOT}/data"
TRAIN_METADATA_DIR = f"{TRAIN_DATA_DIR}/metadata"
TRAIN_RAW_IMAGES_DIR = f"{TRAIN_DATA_DIR}/raw_images"
TRAINING_DATA_DIR = f"{TRAINING_DIR}/data"
TRAIN_CONFIG_PATH = f"{TRAIN_ROOT}/configs/vast_rtx_pro_4500_blackwell_32gb_ondemand.toml"
TRAIN_OUTPUT_MODELS_DIR = f"{TRAIN_ROOT}/output_models"
OUTPUT_MODELS_REL_DIR = "output_models"
TRAIN_REPORTS_DIR = f"{TRAIN_OUTPUT_MODELS_DIR}/logs"
TRAIN_CHECKPOINTS_DIR = f"{TRAIN_OUTPUT_MODELS_DIR}/checkpoints"
TRAIN_ANCHORS_DIR = f"{TRAIN_OUTPUT_MODELS_DIR}/anchors"
REPORTS_REL_DIR = f"{OUTPUT_MODELS_REL_DIR}/logs"
CHECKPOINTS_REL_DIR = f"{OUTPUT_MODELS_REL_DIR}/checkpoints"
ANCHORS_REL_DIR = f"{OUTPUT_MODELS_REL_DIR}/anchors"
TRAIN_MODEL_NAME = "PIA-SPACE-LAB/dinov3-vit7b16-pretrain-lvd1689m"
TRAIN_MODEL_SLUG = "dinov3_vit7b16"
TRAIN_DEFAULT_INPUT_SIZE = 256
TRAIN_UNFREEZE_MAX_INPUT_SIZE = 144
TRAIN_FEATURE_POOL = "cls_mean_patch_concat"
TRAIN_EFFECTIVE_BATCH_SIZE = 64
TRAIN_BATCH_SIZE_CANDIDATES = "8,6,4,2"
TRAIN_OOM_FALLBACK_BATCH_SIZE_CANDIDATES = (8, 6, 4, 2, 1)
TRAIN_NUM_WORKERS = 8
TRAIN_PREFETCH_FACTOR = 4
TRAIN_BUILD_PAIRS_TRAIN_RATIO = 0.8
TRAIN_BUILD_PAIRS_VAL_RATIO = 0.1
TRAIN_BUILD_PAIRS_TRAIN_TARGET = 40_000
TRAIN_BUILD_PAIRS_VAL_TARGET = 5_000
TRAIN_BUILD_PAIRS_MAX_APPEARANCES = 48
TRAIN_DISTANCE1_RATIO = 0.6
TRAIN_DISTANCE2_RATIO = 0.3
TRAIN_DISTANCE3_RATIO = 0.1
TRAIN_TIER_PAIR_MIN_AS = 4_000
TRAIN_TIER_PAIR_MIN_BC = 4_000
TRAIN_TIER_PAIR_MIN_AC = 3_000
TRAIN_TIER_PAIR_MIN_CS = 3_000
TRAIN_TIER_PAIR_CAP_AB = 18_000
VAL_TIER_PAIR_MIN_AS = 400
VAL_TIER_PAIR_MIN_BC = 400
VAL_TIER_PAIR_MIN_AC = 300
VAL_TIER_PAIR_MIN_CS = 300
VAL_TIER_PAIR_CAP_AB = 2_250
TRAIN_ANCHOR_EVAL_N_PER_TIER = 24
TRAIN_ANCHOR_EVAL_BOOTSTRAP_SEEDS = "42,43,44"
TRAIN_ANCHOR_EVAL_MIN_IMPROVEMENT = 0.005
READINESS_REPORT_PATH = f"{REPORTS_REL_DIR}/readiness_report.json"
SNAPSHOT_REPORT_PATH = f"{REPORTS_REL_DIR}/snapshot_report.json"
PREPARED_READINESS_REPORT_PATH = f"{REPORTS_REL_DIR}/prepared_readiness_report.json"
TRAIN_SNAPSHOT_MANIFEST = "training/data/snapshot_manifest.csv"
TRAIN_PREPARED_PAIRS_VAL = "training/data/pairs_val.csv"
TRAIN_PREPARED_METADATA_TRAIN = "training/data/metadata_train.csv"
TRAIN_PREPARED_METADATA_VAL = "training/data/metadata_val.csv"
TRAIN_FULL_TRAIN_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full_train.json"
TRAIN_FULL_TRAIN_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_full_train.json"
TRAIN_FULL_CANDIDATE_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full_candidate.json"
TRAIN_FULL_CANDIDATE_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_full_candidate.json"
TRAIN_FULL_CANDIDATE_ANCHORS = f"{ANCHORS_REL_DIR}/{TRAIN_MODEL_SLUG}_candidate_anchors.pt"
TRAIN_FULL_CANDIDATE_ANCHORS_FILE = f"{TRAIN_ANCHORS_DIR}/{TRAIN_MODEL_SLUG}_candidate_anchors.pt"
TRAIN_FULL_FINAL_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full.json"
TRAIN_FULL_FINAL_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_full.json"
TRAIN_FULL_FINAL_ANCHORS = f"{ANCHORS_REL_DIR}/{TRAIN_MODEL_SLUG}_anchors.pt"
TRAIN_FULL_FINAL_ANCHORS_FILE = f"{TRAIN_ANCHORS_DIR}/{TRAIN_MODEL_SLUG}_anchors.pt"
TRAIN_FULL_REGISTRY = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_postprocess_registry.json"
TRAIN_FULL_REGISTRY_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_postprocess_registry.json"
TRAIN_FULL_FINAL_ANCHOR_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full_anchors.json"
TRAIN_FULL_FINAL_ANCHOR_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_full_anchors.json"
TRAIN_ABLATION_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_summary.json"
TRAIN_ABLATION_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_ablation_summary.json"
TRAIN_FROZEN_ABLATION_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_frozen_ablation_summary.json"
TRAIN_FROZEN_ABLATION_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_frozen_ablation_summary.json"
TRAIN_UNFREEZE_ABLATION_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_unfreeze_ablation_summary.json"
TRAIN_UNFREEZE_ABLATION_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_unfreeze_ablation_summary.json"
TRAIN_OVERALL_WINNER_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_overall_winner.json"
TRAIN_OVERALL_WINNER_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_overall_winner.json"
TRAIN_BASELINE_CHECKPOINT = f"{CHECKPOINTS_REL_DIR}/{TRAIN_MODEL_SLUG}/full/checkpoint_epoch_0010.pt"
TRAIN_BASELINE_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_epoch10_robust_baseline.json"
TRAIN_BASELINE_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_epoch10_robust_baseline.json"
TRAIN_BASELINE_ANCHORS = f"{ANCHORS_REL_DIR}/{TRAIN_MODEL_SLUG}_epoch10_robust_baseline_anchors.pt"
TRAIN_BASELINE_ANCHORS_FILE = f"{TRAIN_ANCHORS_DIR}/{TRAIN_MODEL_SLUG}_epoch10_robust_baseline_anchors.pt"
TRAIN_BATCH_PROBE_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_batch_probe.json"
TRAIN_BATCH_PROBE_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_batch_probe.json"
TRAIN_ANCHOR_ENRICHMENT_REPORT = f"{REPORTS_REL_DIR}/anchor_group_enrichment_report.json"
TRAIN_ANCHOR_ENRICHMENT_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/anchor_group_enrichment_report.json"
TRAIN_ARCHIVE_DIR = f"{TRAIN_OUTPUT_MODELS_DIR}/archive"
ENV_PATH = ROOT / ".env"
LAUNCH_AGENT_LABEL = "com.mirip.vast-checkpoint-sync"
LAUNCH_AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
SYNC_LOG_DIR = ROOT / OUTPUT_MODELS_REL_DIR / "logs" / "vast_sync"
SYNC_LOCK_PATH = SYNC_LOG_DIR / "sync-prune.lock"
LAUNCH_AGENT_LOG_DIR = Path.home() / "Library" / "Logs" / LAUNCH_AGENT_LABEL
CHECKPOINT_EPOCH_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
LAUNCH_AGENT_START_INTERVAL = 900
LEGACY_CHECKPOINTS_REL_DIR = "checkpoints"
REMOTE_CHECKPOINT_ROOTS = (
    f"{CHECKPOINTS_REL_DIR}/{TRAIN_MODEL_SLUG}/full",
    f"{LEGACY_CHECKPOINTS_REL_DIR}/{TRAIN_MODEL_SLUG}/full",
)
FROZEN_ABLATION_VARIANTS: tuple[dict[str, object], ...] = (
    {
        "name": "F1",
        "input_size": 256,
        "head_type": "linear",
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
    },
    {
        "name": "F2",
        "input_size": 256,
        "head_type": "linear",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
    },
    {
        "name": "F3",
        "input_size": 256,
        "head_type": "mlp_small",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
    },
    {
        "name": "F4",
        "input_size": 320,
        "head_type": "linear",
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
    },
)
UNFREEZE_ABLATION_VARIANTS: tuple[dict[str, object], ...] = (
    {
        "name": "U1",
        "unfreeze_last_n_layers": 1,
        "backbone_learning_rate_scale": 0.05,
    },
    {
        "name": "U2",
        "unfreeze_last_n_layers": 1,
        "backbone_learning_rate_scale": 0.02,
    },
)

from vast_ai_control import (  # noqa: E402
    VastClient,
    get_connection_info,
    get_workspace_config,
    load_toml,
    normalize_path,
    require_env,
    run_local_command,
)


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _resolve_train_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def sync_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[sync-prune {timestamp}] {message}", flush=True)


def try_acquire_sync_prune_lock():
    SYNC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = SYNC_LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def release_sync_prune_lock(handle) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    handle.close()


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def resolve_instance_id(instance_id: int | None) -> int:
    if instance_id is not None:
        return instance_id
    value = os.getenv("VAST_INSTANCE_ID")
    if not value:
        raise SystemExit("Missing Vast instance id. Pass --instance-id or set VAST_INSTANCE_ID in train/.env.")
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid VAST_INSTANCE_ID value: {value!r}") from exc


def load_postprocess_registry(path: str | Path) -> dict[str, object]:
    registry_path = _resolve_train_path(path)
    if not registry_path.exists():
        raise SystemExit(f"Postprocess registry not found: {registry_path}")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_best_checkpoint_after_compare")
    if not selected:
        raise SystemExit(f"Postprocess registry is incomplete: {registry_path}")
    allowed_prefixes = (f"{CHECKPOINTS_REL_DIR}/", f"{LEGACY_CHECKPOINTS_REL_DIR}/")
    if not str(selected).startswith(allowed_prefixes):
        raise SystemExit(
            f"Selected checkpoint must stay under {TRAIN_CHECKPOINTS_DIR} or {TRAIN_ROOT}/{LEGACY_CHECKPOINTS_REL_DIR}: {selected}"
        )
    return payload


def resolve_retained_checkpoints(registry: dict[str, object]) -> list[str]:
    retained = registry.get("retained_checkpoints")
    checkpoints: list[str] = []
    if isinstance(retained, list):
        checkpoints.extend(str(item) for item in retained if item)
    selected = registry.get("selected_best_checkpoint_after_compare")
    current_candidate = registry.get("current_candidate_checkpoint")
    for candidate in (selected, current_candidate):
        if candidate:
            checkpoints.append(str(candidate))

    deduped: list[str] = []
    for checkpoint in checkpoints:
        if checkpoint not in deduped:
            deduped.append(checkpoint)
    if not deduped:
        raise SystemExit("Postprocess registry did not contain any retained checkpoints.")
    return deduped


def checkpoint_epoch_value(path: str | Path | None) -> int | None:
    if path in (None, ""):
        return None
    match = CHECKPOINT_EPOCH_PATTERN.search(str(path))
    if not match:
        return None
    return int(match.group(1))


def newest_local_checkpoint_epoch(checkpoints_root: Path) -> int | None:
    epochs = [
        checkpoint_epoch_value(path.name)
        for path in checkpoints_root.glob("**/checkpoint_epoch_*.pt")
    ]
    epochs = [epoch for epoch in epochs if epoch is not None]
    return max(epochs) if epochs else None


def registry_is_stale_for_local_checkpoints(registry: dict[str, object], checkpoints_root: Path) -> bool:
    newest_local_epoch = newest_local_checkpoint_epoch(checkpoints_root)
    candidate_epoch = checkpoint_epoch_value(registry.get("current_candidate_checkpoint"))
    if newest_local_epoch is None or candidate_epoch is None:
        return False
    return newest_local_epoch > candidate_epoch


def _ssh_command(host: str, port: int, remote_command: str) -> list[str]:
    privkey_path = normalize_path(os.getenv("VAST_SSH_PRIVKEY_PATH"))
    command = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
    ]
    if privkey_path:
        command.extend(["-i", str(privkey_path)])
    command.extend([f"root@{host}", remote_command])
    return command


def remote_path_exists(host: str, port: int, remote_root: str, relative_path: str) -> bool:
    remote_command = f"test -f {shlex.quote(f'{remote_root}/{TRAIN_ROOT}/{relative_path}')}"
    result = subprocess.run(_ssh_command(host, port, remote_command), capture_output=True, text=True)
    return result.returncode == 0


def clear_local_sync_cache(paths: list[str]) -> None:
    for relative_path in paths:
        _resolve_train_path(relative_path).unlink(missing_ok=True)


def _remote_python(remote_root: str) -> str:
    return f"{remote_root}/.venv/bin/python"


def _bash_command(parts: list[str]) -> str:
    script = "\n".join(parts)
    return f"bash -lc {shlex.quote(script)}"


def _json_value_command(python_bin: str, json_path: str, expression: str, error_message: str) -> str:
    script = "\n".join(
        [
            "import json",
            "from pathlib import Path",
            f"payload = json.loads(Path({json_path!r}).read_text(encoding='utf-8'))",
            f"value = {expression}",
            "if not value:",
            f"    raise SystemExit({error_message!r})",
            "print(value)",
        ]
    )
    return f"$({python_bin} -c {shlex.quote(script)})"


def _json_env_assignment(
    variable_name: str,
    python_bin: str,
    json_path: str,
    expression: str,
    error_message: str,
) -> str:
    value_command = _json_value_command(python_bin, json_path, expression, error_message)
    return f'{variable_name}="{value_command}"'


def _bootstrap_parts(remote_root: str) -> list[str]:
    return [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_root)}",
        "python3 -m venv --system-site-packages .venv",
        ". .venv/bin/activate",
        "python -m pip install --upgrade pip",
        f"python -m pip install -r {TRAINING_DIR}/requirements.txt",
        "mkdir -p "
        + " ".join(
            [
                TRAIN_OUTPUT_MODELS_DIR,
                f"{TRAIN_CHECKPOINTS_DIR}/dinov3_vitl16",
                f"{TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}",
                TRAIN_REPORTS_DIR,
                TRAIN_ANCHORS_DIR,
                TRAINING_DATA_DIR,
                TRAIN_METADATA_DIR,
                TRAIN_RAW_IMAGES_DIR,
            ]
        ),
    ]


def _build_bootstrap_command(remote_root: str) -> str:
    return _bash_command(_bootstrap_parts(remote_root))


def _validate_upload_parts(remote_root: str) -> list[str]:
    python_bin = _remote_python(remote_root)
    return [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_root)}",
        f"{python_bin} {TRAINING_DIR}/validate_training_readiness.py --mode prepared --metadata-dir data/metadata --image-root data --manifest {TRAIN_SNAPSHOT_MANIFEST} --prepared-dir training/data --baseline-readiness-report {READINESS_REPORT_PATH} --baseline-snapshot-report {SNAPSHOT_REPORT_PATH} --train-ratio {TRAIN_BUILD_PAIRS_TRAIN_RATIO} --val-ratio {TRAIN_BUILD_PAIRS_VAL_RATIO} --train-pairs-target {TRAIN_BUILD_PAIRS_TRAIN_TARGET} --val-pairs-target {TRAIN_BUILD_PAIRS_VAL_TARGET} --same-dept-ratio 0.5 --min-score-gap 5.0 --max-appearances {TRAIN_BUILD_PAIRS_MAX_APPEARANCES} --distance1-ratio {TRAIN_DISTANCE1_RATIO} --distance2-ratio {TRAIN_DISTANCE2_RATIO} --distance3-ratio {TRAIN_DISTANCE3_RATIO} {_tier_pair_quota_args()} --report {PREPARED_READINESS_REPORT_PATH}",
    ]


def _build_validate_upload_command(remote_root: str) -> str:
    return _bash_command(_validate_upload_parts(remote_root))


def _prepare_training_data_parts(python_bin: str) -> list[str]:
    return [
        f"{python_bin} {TRAINING_DIR}/enrich_anchor_metadata.py --metadata-dir data/metadata --report {TRAIN_ANCHOR_ENRICHMENT_REPORT} --min-group-size 15 --apply",
        f"{python_bin} {TRAINING_DIR}/prepare_snapshot.py --metadata-dir data/metadata --image-root data --output-manifest {TRAIN_SNAPSHOT_MANIFEST} --report {SNAPSHOT_REPORT_PATH} --min-group-size 15",
        f"{python_bin} {TRAINING_DIR}/validate_training_readiness.py --mode raw --metadata-dir data/metadata --image-root data --train-ratio {TRAIN_BUILD_PAIRS_TRAIN_RATIO} --val-ratio {TRAIN_BUILD_PAIRS_VAL_RATIO} --train-pairs-target {TRAIN_BUILD_PAIRS_TRAIN_TARGET} --val-pairs-target {TRAIN_BUILD_PAIRS_VAL_TARGET} --same-dept-ratio 0.5 --min-score-gap 5.0 --max-appearances {TRAIN_BUILD_PAIRS_MAX_APPEARANCES} --distance1-ratio {TRAIN_DISTANCE1_RATIO} --distance2-ratio {TRAIN_DISTANCE2_RATIO} --distance3-ratio {TRAIN_DISTANCE3_RATIO} {_tier_pair_quota_args()} --report {READINESS_REPORT_PATH}",
        f"{python_bin} {TRAINING_DIR}/build_pairs.py --manifest {TRAIN_SNAPSHOT_MANIFEST} --output-dir training/data --train-ratio {TRAIN_BUILD_PAIRS_TRAIN_RATIO} --val-ratio {TRAIN_BUILD_PAIRS_VAL_RATIO} --train-pairs-target {TRAIN_BUILD_PAIRS_TRAIN_TARGET} --val-pairs-target {TRAIN_BUILD_PAIRS_VAL_TARGET} --same-dept-ratio 0.5 --min-score-gap 5.0 --max-appearances {TRAIN_BUILD_PAIRS_MAX_APPEARANCES} --distance1-ratio {TRAIN_DISTANCE1_RATIO} --distance2-ratio {TRAIN_DISTANCE2_RATIO} --distance3-ratio {TRAIN_DISTANCE3_RATIO} {_tier_pair_quota_args()} --allow-shortfall",
        f"{python_bin} {TRAINING_DIR}/validate_training_readiness.py --mode prepared --metadata-dir data/metadata --image-root data --manifest {TRAIN_SNAPSHOT_MANIFEST} --prepared-dir training/data --baseline-readiness-report {READINESS_REPORT_PATH} --baseline-snapshot-report {SNAPSHOT_REPORT_PATH} --train-ratio {TRAIN_BUILD_PAIRS_TRAIN_RATIO} --val-ratio {TRAIN_BUILD_PAIRS_VAL_RATIO} --train-pairs-target {TRAIN_BUILD_PAIRS_TRAIN_TARGET} --val-pairs-target {TRAIN_BUILD_PAIRS_VAL_TARGET} --same-dept-ratio 0.5 --min-score-gap 5.0 --max-appearances {TRAIN_BUILD_PAIRS_MAX_APPEARANCES} --distance1-ratio {TRAIN_DISTANCE1_RATIO} --distance2-ratio {TRAIN_DISTANCE2_RATIO} --distance3-ratio {TRAIN_DISTANCE3_RATIO} {_tier_pair_quota_args()} --report {PREPARED_READINESS_REPORT_PATH}",
    ]


def _tier_pair_quota_args() -> str:
    return " ".join(
        [
            f"--train-tier-pair-min-a-s {TRAIN_TIER_PAIR_MIN_AS}",
            f"--train-tier-pair-min-b-c {TRAIN_TIER_PAIR_MIN_BC}",
            f"--train-tier-pair-min-a-c {TRAIN_TIER_PAIR_MIN_AC}",
            f"--train-tier-pair-min-c-s {TRAIN_TIER_PAIR_MIN_CS}",
            f"--train-tier-pair-cap-a-b {TRAIN_TIER_PAIR_CAP_AB}",
            f"--val-tier-pair-min-a-s {VAL_TIER_PAIR_MIN_AS}",
            f"--val-tier-pair-min-b-c {VAL_TIER_PAIR_MIN_BC}",
            f"--val-tier-pair-min-a-c {VAL_TIER_PAIR_MIN_AC}",
            f"--val-tier-pair-min-c-s {VAL_TIER_PAIR_MIN_CS}",
            f"--val-tier-pair-cap-a-b {VAL_TIER_PAIR_CAP_AB}",
        ]
    )


def _anchor_eval_args() -> str:
    return " ".join(
        [
            f"--anchor-eval-n-per-tier {TRAIN_ANCHOR_EVAL_N_PER_TIER}",
            f"--anchor-eval-bootstrap-seeds {TRAIN_ANCHOR_EVAL_BOOTSTRAP_SEEDS}",
            f"--anchor-eval-min-improvement {TRAIN_ANCHOR_EVAL_MIN_IMPROVEMENT}",
            "--anchor-eval-group-balanced",
        ]
    )


def _remote_train_file(path: str) -> str:
    return f"{TRAIN_ROOT}/{path}"


def _batch_probe_parts(remote_root: str, *, head_type: str, input_size: str | int, report_file: str) -> list[str]:
    python_bin = _remote_python(remote_root)
    return [
        f"mkdir -p {TRAIN_REPORTS_DIR}",
        f"{python_bin} {TRAINING_DIR}/probe_dinov3_batch_size.py --pairs-train training/data/pairs_train.csv --image-root data --model-name {shlex.quote(TRAIN_MODEL_NAME)} --input-size {input_size} --feature-pool {TRAIN_FEATURE_POOL} --head-type {head_type} --dropout 0.1 --margin 0.3 --backbone-dtype auto --precision bf16 --batch-size-candidates {TRAIN_BATCH_SIZE_CANDIDATES} > {report_file}",
        _json_env_assignment(
            "MICRO_BATCH",
            python_bin,
            report_file,
            "payload.get('selected_batch_size')",
            "batch probe did not select a batch size",
        ),
        f'GRAD_ACCUM=$((({TRAIN_EFFECTIVE_BATCH_SIZE} + MICRO_BATCH - 1) / MICRO_BATCH))',
    ]


def _oom_retry_shell_parts() -> list[str]:
    candidates = " ".join(str(candidate) for candidate in TRAIN_OOM_FALLBACK_BATCH_SIZE_CANDIDATES)
    oom_pattern = "CUDA out of memory|OutOfMemoryError|torch\\.OutOfMemoryError"
    return [
        'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
        "next_micro_batch() {",
        '  local current="$1"',
        f'  for candidate in {candidates}; do',
        '    if [ "$candidate" -lt "$current" ]; then',
        '      printf "%s" "$candidate"',
        "      return 0",
        "    fi",
        "  done",
        "  return 0",
        "}",
        "resolve_stage_progress_args() {",
        '  local checkpoint_dir_rel="$1"',
        '  local checkpoint_dir_file="$2"',
        '  local initial_progress_args="$3"',
        '  local latest_checkpoint=""',
        '  if [ -d "$checkpoint_dir_file" ]; then',
        '    latest_checkpoint=$(find "$checkpoint_dir_file" -maxdepth 1 -type f -name "checkpoint_epoch_*.pt" | sort | tail -n 1)',
        "  fi",
        '  if [ -n "$latest_checkpoint" ]; then',
        '    printf -- "--resume-from %s --resume-next-epoch" "${checkpoint_dir_rel}/$(basename "$latest_checkpoint")"',
        "    return 0",
        "  fi",
        '  if [ -n "$initial_progress_args" ]; then',
        '    eval "printf \'%s\' \\"$initial_progress_args\\""',
        "  fi",
        "}",
        "run_training_with_oom_retry() {",
        '  local stage_label="$1"',
        '  local checkpoint_dir_rel="$2"',
        '  local checkpoint_dir_file="$3"',
        '  local initial_progress_args="$4"',
        '  local command_template="$5"',
        "  local attempt=1",
        '  local current_micro_batch="${MICRO_BATCH}"',
        "  while true; do",
        '    MICRO_BATCH="$current_micro_batch"',
        f'    GRAD_ACCUM=$((({TRAIN_EFFECTIVE_BATCH_SIZE} + MICRO_BATCH - 1) / MICRO_BATCH))',
        '    PROGRESS_ARGS="$(resolve_stage_progress_args "$checkpoint_dir_rel" "$checkpoint_dir_file" "$initial_progress_args")"',
        f'    ATTEMPT_LOG="{TRAIN_REPORTS_DIR}/${{stage_label}}_attempt_${{attempt}}_mb${{MICRO_BATCH}}.log"',
        '    echo "[oom-retry] stage=$stage_label attempt=$attempt micro_batch=$MICRO_BATCH grad_accum=$GRAD_ACCUM progress_args=${PROGRESS_ARGS:-<none>}"',
        "    set +e",
        '    eval "$command_template" > >(tee "$ATTEMPT_LOG") 2> >(tee -a "$ATTEMPT_LOG" >&2)',
        "    local status=$?",
        "    set -e",
        '    if [ "$status" -eq 0 ]; then',
        "      return 0",
        "    fi",
        f'    if ! grep -Eiq "{oom_pattern}" "$ATTEMPT_LOG"; then',
        '      echo "[oom-retry] stage=$stage_label failed with non-OOM status=$status"',
        '      return "$status"',
        "    fi",
        '    local next_batch=""',
        '    next_batch="$(next_micro_batch "$MICRO_BATCH")"',
        '    if [ -z "$next_batch" ]; then',
        '      echo "[oom-retry] stage=$stage_label exhausted fallback candidates after OOM at micro_batch=$MICRO_BATCH"',
        '      return "$status"',
        "    fi",
        '    echo "[oom-retry] stage=$stage_label OOM at micro_batch=$MICRO_BATCH; retrying with micro_batch=$next_batch"',
        '    current_micro_batch="$next_batch"',
        "    attempt=$((attempt + 1))",
        "  done",
        "}",
    ]


def _build_oom_retry_train_parts(
    *,
    stage_label: str,
    checkpoint_dir: str,
    checkpoint_dir_file: str,
    initial_progress_args: str,
    train_command: str,
) -> list[str]:
    return [
        f'TRAIN_COMMAND_{stage_label}=$(cat <<\'EOF_{stage_label}\'\n{train_command}\nEOF_{stage_label}\n)',
        f'run_training_with_oom_retry "{stage_label}" "{checkpoint_dir}" "{checkpoint_dir_file}" {shlex.quote(initial_progress_args)} "$TRAIN_COMMAND_{stage_label}"',
    ]


def _build_pairs_legacy_aligned_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}",
            *_prepare_training_data_parts(python_bin),
        ]
    )


def _build_re_evaluate_baseline_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    required_prepared_paths = (
        TRAIN_PREPARED_PAIRS_VAL,
        TRAIN_PREPARED_METADATA_TRAIN,
        TRAIN_PREPARED_METADATA_VAL,
    )
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}",
            f"if [ ! -x {shlex.quote(python_bin)} ]; then",
            *[f"  {command}" for command in _bootstrap_parts(remote_root)],
            "fi",
            "if "
            + " || ".join(f"[ ! -f {_remote_train_file(path)} ]" for path in required_prepared_paths)
            + "; then",
            *[f"  {command}" for command in _prepare_training_data_parts(python_bin)],
            "fi",
            f"mkdir -p {TRAIN_REPORTS_DIR} {TRAIN_ANCHORS_DIR}",
            f"{python_bin} {TRAINING_DIR}/reevaluate_checkpoint.py --checkpoint {TRAIN_BASELINE_CHECKPOINT} --pairs-val training/data/pairs_val.csv --metadata-train training/data/metadata_train.csv --metadata-eval training/data/metadata_val.csv --image-root data --anchors-output {TRAIN_BASELINE_ANCHORS} --output {TRAIN_BASELINE_REPORT} --batch-size 8 --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --device cuda --precision bf16",
        ]
    )


def _build_variant_keep_best_only_parts(
    python_bin: str,
    *,
    checkpoint_dir_file: str,
    registry_report_file: str,
) -> list[str]:
    selected_checkpoint = _json_value_command(
        python_bin,
        registry_report_file,
        "payload.get('selected_best_checkpoint_after_compare')",
        "selected best checkpoint missing from variant registry",
    )
    return [
        f'SELECTED_VARIANT_CHECKPOINT_REL="{selected_checkpoint}"',
        f'SELECTED_VARIANT_CHECKPOINT="{TRAIN_ROOT}/${{SELECTED_VARIANT_CHECKPOINT_REL}}"',
        f'find {shlex.quote(checkpoint_dir_file)} -maxdepth 1 -type f -name "checkpoint_epoch_*.pt" ! -path "$SELECTED_VARIANT_CHECKPOINT" -delete',
        f'ln -sfn "$(basename "$SELECTED_VARIANT_CHECKPOINT")" {shlex.quote(f"{checkpoint_dir_file}/best_model.pt")}',
    ]


def _build_prune_non_winner_variant_checkpoints_parts(
    python_bin: str,
    *,
    summary_report_file: str,
    summary_label: str,
    variants: tuple[dict[str, object], ...],
    variants_root_dir_file: str,
) -> list[str]:
    winner_name_var = f"{summary_label.upper()}_WINNER_NAME"
    parts = [
        _json_env_assignment(
            winner_name_var,
            python_bin,
            summary_report_file,
            "payload.get('winner_name')",
            f"{summary_label} winner name missing",
        )
    ]
    for variant in variants:
        variant_name = str(variant["name"])
        parts.append(
            f'if [ "${winner_name_var}" != "{variant_name}" ]; then rm -rf {shlex.quote(f"{variants_root_dir_file}/{variant_name}")}; fi'
        )
    return parts


def _build_frozen_ablation_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    parts = [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_root)}",
        f"mkdir -p {TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/ablation {TRAIN_REPORTS_DIR} {TRAIN_ANCHORS_DIR}",
        *_oom_retry_shell_parts(),
    ]
    for variant in FROZEN_ABLATION_VARIANTS:
        name = str(variant["name"])
        input_size = int(variant["input_size"])
        head_type = str(variant["head_type"])
        learning_rate = float(variant["learning_rate"])
        weight_decay = float(variant["weight_decay"])
        checkpoint_dir = f"{CHECKPOINTS_REL_DIR}/{TRAIN_MODEL_SLUG}/ablation/{name}"
        checkpoint_dir_file = f"{TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/ablation/{name}"
        candidate_report = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_candidate.json"
        candidate_report_file = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_candidate.json"
        registry_report = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_registry.json"
        registry_report_file = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_registry.json"
        anchors_path = f"{ANCHORS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_candidate_anchors.pt"
        anchors_path_file = f"{TRAIN_ANCHORS_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_candidate_anchors.pt"
        train_report = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_train.json"
        train_report_file = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_train.json"
        probe_report_file = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_batch_probe_{name}.json"
        parts.extend(
            [
                f"rm -rf {checkpoint_dir_file}",
                f"rm -f {candidate_report_file} {registry_report_file} {anchors_path_file} {train_report_file} {probe_report_file}",
                *_batch_probe_parts(
                    remote_root,
                    head_type=head_type,
                    input_size=input_size,
                    report_file=probe_report_file,
                ),
                *_build_oom_retry_train_parts(
                    stage_label=f"frozen_{name}",
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_dir_file=checkpoint_dir_file,
                    initial_progress_args="",
                    train_command=f"{python_bin} {TRAINING_DIR}/train_dinov3.py --pairs-train training/data/pairs_train.csv --pairs-val training/data/pairs_val.csv --image-root data --output-dir {checkpoint_dir} --model-name {shlex.quote(TRAIN_MODEL_NAME)} --backbone-dtype auto --epochs 6 --warmup-epochs 1 --batch-size \"$MICRO_BATCH\" --gradient-accumulation-steps \"$GRAD_ACCUM\" --learning-rate {learning_rate} --weight-decay {weight_decay} --backbone-learning-rate-scale 0.1 --dropout 0.1 --margin 0.3 --input-size {input_size} --feature-pool {TRAIN_FEATURE_POOL} --head-type {head_type} --freeze-backbone --unfreeze-last-n-layers 0 --patience 2 --restart-from-best-patience 0 --early-stopping-metric anchor_tier_accuracy {_anchor_eval_args()} --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --precision bf16 $PROGRESS_ARGS --report {train_report} --postprocess-metadata-train training/data/metadata_train.csv --postprocess-metadata-eval training/data/metadata_val.csv --postprocess-anchors-output {anchors_path} --postprocess-report {candidate_report} --postprocess-registry {registry_report}",
                ),
                *_build_variant_keep_best_only_parts(
                    python_bin,
                    checkpoint_dir_file=checkpoint_dir_file,
                    registry_report_file=registry_report_file,
                ),
            ]
        )
    return _bash_command(parts)


def _build_select_ablation_winner_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    candidate_args = " ".join(
        f"--candidate {variant['name']}={REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{variant['name']}_registry.json"
        for variant in FROZEN_ABLATION_VARIANTS
    )
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}",
            f"{python_bin} {TRAINING_DIR}/select_ablation_winner.py {candidate_args} --min-improvement {TRAIN_ANCHOR_EVAL_MIN_IMPROVEMENT} --output {TRAIN_FROZEN_ABLATION_REPORT}",
        ]
    )


def _build_unfreeze_ablation_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    winner_metric = _json_value_command(
        python_bin,
        TRAIN_FROZEN_ABLATION_REPORT_FILE,
        "payload.get('winner_metrics', {}).get('anchor_tier_accuracy_mean') or payload.get('winner_metrics', {}).get('anchor_tier_accuracy')",
        "winner metric missing from frozen ablation report",
    )
    parts = [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_root)}",
        *_oom_retry_shell_parts(),
        f'FROZEN_WINNER_METRIC="{winner_metric}"',
        "python3 - \"$FROZEN_WINNER_METRIC\" <<'PY'\nimport json, sys\nfrom pathlib import Path\nvalue = float(sys.argv[1])\nout = Path('" + TRAIN_UNFREEZE_ABLATION_REPORT_FILE + "')\nout.parent.mkdir(parents=True, exist_ok=True)\nif value >= 0.56:\n    out.write_text(json.dumps({'skipped': True, 'reason': 'frozen_winner_meets_threshold', 'threshold': 0.56}, indent=2, ensure_ascii=False), encoding='utf-8')\nPY",
        'if python3 - "$FROZEN_WINNER_METRIC" <<\'PY\'\nimport sys\nsys.exit(0 if float(sys.argv[1]) >= 0.56 else 1)\nPY\nthen exit 0; fi',
        _json_env_assignment(
            "FROZEN_WINNER_CHECKPOINT",
            python_bin,
            TRAIN_FROZEN_ABLATION_REPORT_FILE,
            "payload.get('winner_checkpoint')",
            "frozen winner checkpoint missing",
        ),
        _json_env_assignment(
            "FROZEN_WINNER_HEAD_TYPE",
            python_bin,
            TRAIN_FROZEN_ABLATION_REPORT_FILE,
            "payload.get('winner_config', {}).get('head_type')",
            "frozen winner head type missing",
        ),
        _json_env_assignment(
            "FROZEN_WINNER_INPUT_SIZE",
            python_bin,
            TRAIN_FROZEN_ABLATION_REPORT_FILE,
            "payload.get('winner_config', {}).get('input_size')",
            "frozen winner input size missing",
        ),
        _json_env_assignment(
            "FROZEN_WINNER_LR",
            python_bin,
            TRAIN_FROZEN_ABLATION_REPORT_FILE,
            "payload.get('winner_config', {}).get('learning_rate')",
            "frozen winner lr missing",
        ),
        _json_env_assignment(
            "FROZEN_WINNER_WEIGHT_DECAY",
            python_bin,
            TRAIN_FROZEN_ABLATION_REPORT_FILE,
            "payload.get('winner_config', {}).get('weight_decay')",
            "frozen winner weight decay missing",
        ),
        'UNFREEZE_INPUT_SIZE="$(python3 - "$FROZEN_WINNER_INPUT_SIZE" <<\'PY\'\nimport sys\nprint(min(int(float(sys.argv[1])), ' + str(TRAIN_UNFREEZE_MAX_INPUT_SIZE) + '))\nPY\n)"',
        *_build_prune_non_winner_variant_checkpoints_parts(
            python_bin,
            summary_report_file=TRAIN_FROZEN_ABLATION_REPORT_FILE,
            summary_label="frozen",
            variants=FROZEN_ABLATION_VARIANTS,
            variants_root_dir_file=f"{TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/ablation",
        ),
        'HALF_WINNER_LR="$(python3 - "$FROZEN_WINNER_LR" <<\'PY\'\nimport sys\nprint(float(sys.argv[1]) / 2.0)\nPY\n)"',
        f"mkdir -p {TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/ablation {TRAIN_REPORTS_DIR} {TRAIN_ANCHORS_DIR}",
    ]
    for variant in UNFREEZE_ABLATION_VARIANTS:
        name = str(variant["name"])
        unfreeze_last_n_layers = int(variant["unfreeze_last_n_layers"])
        backbone_learning_rate_scale = float(variant["backbone_learning_rate_scale"])
        checkpoint_dir = f"{CHECKPOINTS_REL_DIR}/{TRAIN_MODEL_SLUG}/ablation/{name}"
        checkpoint_dir_file = f"{TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/ablation/{name}"
        candidate_report = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_candidate.json"
        candidate_report_file = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_candidate.json"
        registry_report = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_registry.json"
        registry_report_file = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_registry.json"
        anchors_path = f"{ANCHORS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_candidate_anchors.pt"
        anchors_path_file = f"{TRAIN_ANCHORS_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_candidate_anchors.pt"
        train_report = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_train.json"
        train_report_file = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_ablation_{name}_train.json"
        probe_report_file = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_batch_probe_{name}.json"
        stage_label = f"unfreeze_{name}"
        train_command = f"{python_bin} {TRAINING_DIR}/train_dinov3.py --pairs-train training/data/pairs_train.csv --pairs-val training/data/pairs_val.csv --image-root data --output-dir {checkpoint_dir} --model-name {shlex.quote(TRAIN_MODEL_NAME)} --backbone-dtype auto --epochs 4 --warmup-epochs 1 --batch-size \"$MICRO_BATCH\" --gradient-accumulation-steps \"$GRAD_ACCUM\" --learning-rate \"$HALF_WINNER_LR\" --weight-decay \"$FROZEN_WINNER_WEIGHT_DECAY\" --backbone-learning-rate-scale {backbone_learning_rate_scale} --dropout 0.1 --margin 0.3 --input-size \"$UNFREEZE_INPUT_SIZE\" --feature-pool {TRAIN_FEATURE_POOL} --head-type \"$FROZEN_WINNER_HEAD_TYPE\" --no-freeze-backbone --unfreeze-last-n-layers {unfreeze_last_n_layers} --patience 2 --restart-from-best-patience 0 --early-stopping-metric anchor_tier_accuracy {_anchor_eval_args()} --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --precision bf16 $PROGRESS_ARGS --report {train_report} --postprocess-metadata-train training/data/metadata_train.csv --postprocess-metadata-eval training/data/metadata_val.csv --postprocess-anchors-output {anchors_path} --postprocess-report {candidate_report} --postprocess-registry {registry_report}"
        skip_on_failure_command = f"""if ! run_training_with_oom_retry "{stage_label}" "{checkpoint_dir}" "{checkpoint_dir_file}" {shlex.quote("--initialize-from $FROZEN_WINNER_CHECKPOINT")} "$TRAIN_COMMAND_{stage_label}"; then
python3 - "{name}" "{TRAIN_UNFREEZE_ABLATION_REPORT_FILE}" "$UNFREEZE_INPUT_SIZE" "{unfreeze_last_n_layers}" <<'PY'
import json, sys
from pathlib import Path
variant_name, output_path, input_size, unfreeze_last_n_layers = sys.argv[1:5]
out = Path(output_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({{
    "skipped": True,
    "reason": "oom_minimum_unfreeze_config",
    "failed_variant": variant_name,
    "attempted_input_size": int(input_size),
    "attempted_unfreeze_last_n_layers": int(unfreeze_last_n_layers),
}}, indent=2, ensure_ascii=False), encoding="utf-8")
PY
exit 0
fi"""
        parts.extend(
            [
                f"rm -rf {checkpoint_dir_file}",
                f"rm -f {candidate_report_file} {registry_report_file} {anchors_path_file} {train_report_file} {probe_report_file}",
                *_batch_probe_parts(
                    remote_root,
                    head_type="$FROZEN_WINNER_HEAD_TYPE",
                    input_size="$UNFREEZE_INPUT_SIZE",
                    report_file=probe_report_file,
                ),
                f'TRAIN_COMMAND_{stage_label}=$(cat <<\'EOF_{stage_label}\'\n{train_command}\nEOF_{stage_label}\n)',
                skip_on_failure_command,
                *_build_variant_keep_best_only_parts(
                    python_bin,
                    checkpoint_dir_file=checkpoint_dir_file,
                    registry_report_file=registry_report_file,
                ),
            ]
        )
    candidate_args = " ".join(
        f"--candidate {variant['name']}={REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_ablation_{variant['name']}_registry.json"
        for variant in UNFREEZE_ABLATION_VARIANTS
    )
    parts.append(
        f"{python_bin} {TRAINING_DIR}/select_ablation_winner.py {candidate_args} --min-improvement {TRAIN_ANCHOR_EVAL_MIN_IMPROVEMENT} --output {TRAIN_UNFREEZE_ABLATION_REPORT}"
    )
    return _bash_command(parts)


def _build_select_overall_winner_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}",
            f"{python_bin} {TRAINING_DIR}/select_overall_winner.py --summary frozen={TRAIN_FROZEN_ABLATION_REPORT} --summary unfreeze={TRAIN_UNFREEZE_ABLATION_REPORT} --min-improvement {TRAIN_ANCHOR_EVAL_MIN_IMPROVEMENT} --output {TRAIN_OVERALL_WINNER_REPORT}",
        ]
    )


def _build_archive_reset_parts() -> list[str]:
    archive_candidates = [
        f"{TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/full",
        TRAIN_FULL_TRAIN_REPORT_FILE,
        TRAIN_FULL_CANDIDATE_REPORT_FILE,
        TRAIN_FULL_REGISTRY_FILE,
        TRAIN_FULL_CANDIDATE_ANCHORS_FILE,
        TRAIN_FULL_FINAL_REPORT_FILE,
        TRAIN_FULL_FINAL_ANCHORS_FILE,
        TRAIN_FULL_FINAL_ANCHOR_REPORT_FILE,
    ]
    parts = [
        "RUN_ARCHIVE_ID=$(date +%Y%m%d_%H%M%S)",
        f"FULL_ARCHIVE_DIR={TRAIN_ARCHIVE_DIR}/{TRAIN_MODEL_SLUG}_full_${{RUN_ARCHIVE_ID}}",
        "NEED_ARCHIVE=0",
    ]
    for candidate in archive_candidates:
        parts.append(f'if [ -e {shlex.quote(candidate)} ]; then NEED_ARCHIVE=1; fi')
    parts.extend(
        [
            'if [ "$NEED_ARCHIVE" -eq 1 ]; then mkdir -p "$FULL_ARCHIVE_DIR/checkpoints" "$FULL_ARCHIVE_DIR/logs" "$FULL_ARCHIVE_DIR/anchors"; fi',
            f'if [ -d {TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/full ]; then mv {TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/full "$FULL_ARCHIVE_DIR/checkpoints/full"; fi',
            f'for FILE in {TRAIN_FULL_TRAIN_REPORT_FILE} {TRAIN_FULL_CANDIDATE_REPORT_FILE} {TRAIN_FULL_REGISTRY_FILE} {TRAIN_FULL_FINAL_REPORT_FILE} {TRAIN_FULL_FINAL_ANCHOR_REPORT_FILE}; do if [ -f \"$FILE\" ]; then mv \"$FILE\" \"$FULL_ARCHIVE_DIR/logs/\"; fi; done',
            f'for FILE in {TRAIN_FULL_CANDIDATE_ANCHORS_FILE} {TRAIN_FULL_FINAL_ANCHORS_FILE}; do if [ -f \"$FILE\" ]; then mv \"$FILE\" \"$FULL_ARCHIVE_DIR/anchors/\"; fi; done',
            f"mkdir -p {TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/full",
            f"rm -f {TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/full/*.pt",
        ]
    )
    return parts


def _build_full_fresh_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    winner_head_type = _json_value_command(
        python_bin,
        TRAIN_OVERALL_WINNER_REPORT_FILE,
        "payload.get('winner_config', {}).get('head_type')",
        "winner head type missing from ablation report",
    )
    winner_input_size = _json_value_command(
        python_bin,
        TRAIN_OVERALL_WINNER_REPORT_FILE,
        "payload.get('winner_config', {}).get('input_size')",
        "winner input size missing from overall winner report",
    )
    winner_learning_rate = _json_value_command(
        python_bin,
        TRAIN_OVERALL_WINNER_REPORT_FILE,
        "payload.get('winner_config', {}).get('learning_rate')",
        "winner learning rate missing from ablation report",
    )
    winner_weight_decay = _json_value_command(
        python_bin,
        TRAIN_OVERALL_WINNER_REPORT_FILE,
        "payload.get('winner_config', {}).get('weight_decay')",
        "winner weight decay missing from ablation report",
    )
    winner_backbone_lr_scale = _json_value_command(
        python_bin,
        TRAIN_OVERALL_WINNER_REPORT_FILE,
        "payload.get('winner_config', {}).get('backbone_learning_rate_scale', 0.1)",
        "winner backbone lr scale missing from overall winner report",
    )
    winner_freeze_backbone = _json_value_command(
        python_bin,
        TRAIN_OVERALL_WINNER_REPORT_FILE,
        "payload.get('winner_config', {}).get('freeze_backbone', True)",
        "winner freeze_backbone missing from overall winner report",
    )
    winner_unfreeze_last_n_layers = _json_value_command(
        python_bin,
        TRAIN_OVERALL_WINNER_REPORT_FILE,
        "payload.get('winner_config', {}).get('unfreeze_last_n_layers', 0)",
        "winner unfreeze depth missing from overall winner report",
    )
    selected_checkpoint_value = _json_value_command(
        python_bin,
        TRAIN_FULL_REGISTRY_FILE,
        "payload.get('selected_best_checkpoint_after_compare')",
        "selected best checkpoint missing from postprocess registry",
    )
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}",
            *_oom_retry_shell_parts(),
            f'WINNER_HEAD_TYPE="{winner_head_type}"',
            f'WINNER_INPUT_SIZE="{winner_input_size}"',
            f'WINNER_LR="{winner_learning_rate}"',
            f'WINNER_WEIGHT_DECAY="{winner_weight_decay}"',
            f'WINNER_BACKBONE_LR_SCALE="{winner_backbone_lr_scale}"',
            f'WINNER_FREEZE_BACKBONE="{winner_freeze_backbone}"',
            f'WINNER_UNFREEZE_LAST_N_LAYERS="{winner_unfreeze_last_n_layers}"',
            'if [ "$WINNER_FREEZE_BACKBONE" = "True" ] || [ "$WINNER_FREEZE_BACKBONE" = "true" ]; then FREEZE_FLAG="--freeze-backbone"; else FREEZE_FLAG="--no-freeze-backbone"; fi',
            * _batch_probe_parts(
                remote_root,
                head_type="$WINNER_HEAD_TYPE",
                input_size="$WINNER_INPUT_SIZE",
                report_file=TRAIN_BATCH_PROBE_REPORT_FILE,
            ),
            * _build_archive_reset_parts(),
            *_build_oom_retry_train_parts(
                stage_label="full_fresh",
                checkpoint_dir=f"{CHECKPOINTS_REL_DIR}/{TRAIN_MODEL_SLUG}/full",
                checkpoint_dir_file=f"{TRAIN_CHECKPOINTS_DIR}/{TRAIN_MODEL_SLUG}/full",
                initial_progress_args="",
                train_command=f"{python_bin} {TRAINING_DIR}/train_dinov3.py --pairs-train training/data/pairs_train.csv --pairs-val training/data/pairs_val.csv --image-root data --output-dir {CHECKPOINTS_REL_DIR}/{TRAIN_MODEL_SLUG}/full --model-name {shlex.quote(TRAIN_MODEL_NAME)} --backbone-dtype auto --epochs 24 --warmup-epochs 2 --batch-size \"$MICRO_BATCH\" --gradient-accumulation-steps \"$GRAD_ACCUM\" --learning-rate \"$WINNER_LR\" --weight-decay \"$WINNER_WEIGHT_DECAY\" --backbone-learning-rate-scale \"$WINNER_BACKBONE_LR_SCALE\" --dropout 0.1 --margin 0.3 --input-size \"$WINNER_INPUT_SIZE\" --feature-pool {TRAIN_FEATURE_POOL} --head-type \"$WINNER_HEAD_TYPE\" $FREEZE_FLAG --unfreeze-last-n-layers \"$WINNER_UNFREEZE_LAST_N_LAYERS\" --patience 6 --restart-from-best-patience 3 --early-stopping-metric anchor_tier_accuracy {_anchor_eval_args()} --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --precision bf16 $PROGRESS_ARGS --report {TRAIN_FULL_TRAIN_REPORT} --postprocess-metadata-train training/data/metadata_train.csv --postprocess-metadata-eval training/data/metadata_val.csv --postprocess-anchors-output {TRAIN_FULL_CANDIDATE_ANCHORS} --postprocess-report {TRAIN_FULL_CANDIDATE_REPORT} --postprocess-registry {TRAIN_FULL_REGISTRY}",
            ),
            f'SELECTED_CHECKPOINT="{selected_checkpoint_value}"',
            f"{python_bin} {TRAINING_DIR}/build_anchors_dinov3.py --checkpoint \"$SELECTED_CHECKPOINT\" --metadata training/data/metadata_train.csv --image-root data --output {TRAIN_FULL_FINAL_ANCHORS} --report {TRAIN_FULL_FINAL_ANCHOR_REPORT}",
            f"{python_bin} {TRAINING_DIR}/evaluate_dinov3.py --checkpoint \"$SELECTED_CHECKPOINT\" --pairs-val training/data/pairs_val.csv --image-root data --anchors {TRAIN_FULL_FINAL_ANCHORS} --metadata-eval training/data/metadata_val.csv --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --output {TRAIN_FULL_FINAL_REPORT}",
        ]
    )


def build_stage_command(stage: str, remote_root: str) -> str:
    if stage == "bootstrap":
        return _build_bootstrap_command(remote_root)
    if stage == "validate-upload":
        return _build_validate_upload_command(remote_root)
    if stage == "build-pairs-legacy-aligned":
        return _build_pairs_legacy_aligned_command(remote_root)
    if stage == "re-evaluate-baseline":
        return _build_re_evaluate_baseline_command(remote_root)
    if stage == "frozen-ablation":
        return _build_frozen_ablation_command(remote_root)
    if stage == "select-ablation-winner":
        return _build_select_ablation_winner_command(remote_root)
    if stage == "unfreeze-ablation":
        return _build_unfreeze_ablation_command(remote_root)
    if stage == "select-overall-winner":
        return _build_select_overall_winner_command(remote_root)
    if stage == "full-fresh":
        return _build_full_fresh_command(remote_root)
    raise SystemExit(f"Unsupported stage: {stage}")


def execute_stage(client: VastClient, instance_id: int, command: str, follow: bool, follow_delay: int) -> int:
    result = client.execute(instance_id, command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if follow and result.get("result_url"):
        time.sleep(follow_delay)
        print(client.fetch_text(result["result_url"]))
    return 0


def pull_artifacts(config_path: str, instance_id: int) -> int:
    api_key = require_env("VAST_API_KEY")
    client = VastClient(api_key)
    host, port = get_connection_info(client, instance_id)
    config_absolute = str(_resolve_repo_path(config_path))
    workspace_cfg = get_workspace_config(config_absolute)
    remote_root = workspace_cfg.get("remote_root", "/workspace/mirip_v2")
    rsync_prefix = build_rsync_prefix(port)
    artifact_dirs = (
        (TRAIN_CHECKPOINTS_DIR, ROOT / CHECKPOINTS_REL_DIR),
        (TRAIN_REPORTS_DIR, ROOT / REPORTS_REL_DIR),
        (TRAIN_ANCHORS_DIR, ROOT / ANCHORS_REL_DIR),
    )
    for remote_dir, local_dir in artifact_dirs:
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = list(rsync_prefix)
        remote = f"root@{host}:{remote_root}/{remote_dir}/"
        local = str(local_dir)
        cmd.extend([remote, local])
        result = run_local_command(cmd)
        if result != 0:
            return result
    return 0


def build_rsync_prefix(port: int, *extra_args: str) -> list[str]:
    privkey_path = normalize_path(os.getenv("VAST_SSH_PRIVKEY_PATH"))
    ssh_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no"
    prefix = [
        "rsync",
        "-az",
        "-e",
        ssh_cmd if not privkey_path else f"{ssh_cmd} -i {shlex.quote(str(privkey_path))}",
    ]
    prefix.extend(extra_args)
    return prefix


def checkpoint_local_path(checkpoint_relative: str) -> Path:
    checkpoint_path = Path(checkpoint_relative)
    parts = checkpoint_path.parts
    if parts[:2] == (OUTPUT_MODELS_REL_DIR, "checkpoints"):
        suffix = Path(*parts[2:])
    elif parts[:1] == (LEGACY_CHECKPOINTS_REL_DIR,):
        suffix = Path(*parts[1:])
    else:
        raise SystemExit(f"Unsupported checkpoint path in registry: {checkpoint_relative}")
    return ROOT / CHECKPOINTS_REL_DIR / suffix


def pull_sync_prune_artifacts(
    *,
    host: str,
    port: int,
    remote_root: str,
    include_anchors: bool = False,
    retained_checkpoints: list[str] | None = None,
    selected_checkpoint: str | None = None,
) -> int:
    rsync_prefix = build_rsync_prefix(port)
    artifact_dirs = [(TRAIN_REPORTS_DIR, ROOT / REPORTS_REL_DIR)]
    if include_anchors:
        artifact_dirs.append((TRAIN_ANCHORS_DIR, ROOT / ANCHORS_REL_DIR))
    for remote_dir, local_dir in artifact_dirs:
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = list(rsync_prefix)
        cmd.extend([f"root@{host}:{remote_root}/{remote_dir}/", str(local_dir)])
        result = run_local_command(cmd)
        if result != 0:
            return result

    if retained_checkpoints:
        for checkpoint_relative in retained_checkpoints:
            local_path = checkpoint_local_path(checkpoint_relative)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = build_rsync_prefix(port, "--partial", "--size-only")
            remote_path = f"root@{host}:{remote_root}/{TRAIN_ROOT}/{checkpoint_relative}"
            cmd.extend([remote_path, str(local_path)])
            result = run_local_command(cmd)
            if result != 0:
                return result

    if selected_checkpoint:
        selected_local_path = checkpoint_local_path(selected_checkpoint)
        best_link = selected_local_path.parent / "best_model.pt"
        best_link.unlink(missing_ok=True)

    return 0


def build_remote_prune_command(
    remote_root: str,
    selected_checkpoint: str,
    retained_checkpoints: list[str],
    registry_candidate_checkpoint: str | None = None,
) -> str:
    retained = [checkpoint.strip() for checkpoint in retained_checkpoints if checkpoint.strip()]
    selected = selected_checkpoint.strip()
    selected_dir = str(Path(selected).parent).replace("\\", "/")
    command_parts = [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_root)}/{TRAIN_ROOT}",
        f'SELECTED_CHECKPOINT={shlex.quote(selected)}',
        'if [ ! -f "$SELECTED_CHECKPOINT" ]; then echo "Selected checkpoint missing: $SELECTED_CHECKPOINT" >&2; exit 1; fi',
    ]
    for index, checkpoint in enumerate(retained):
        command_parts.append(f'KEEP_{index}={shlex.quote(checkpoint)}')
    candidate_epoch = checkpoint_epoch_value(registry_candidate_checkpoint)
    if candidate_epoch is not None:
        remote_epoch_find_commands = " ; ".join(
            f'find {shlex.quote(root)} -type f -name "checkpoint_epoch_*.pt" 2>/dev/null'
            for root in REMOTE_CHECKPOINT_ROOTS
        )
        command_parts.extend(
            [
                f"REGISTRY_CANDIDATE_EPOCH={candidate_epoch}",
                "LATEST_REMOTE_EPOCH=$(" +
                "{ " +
                remote_epoch_find_commands +
                "; } | sed -E 's|.*checkpoint_epoch_0*([0-9]+)\\.pt|\\1|' | sort -n | tail -n 1)",
                'if [ -n "$LATEST_REMOTE_EPOCH" ] && [ "$LATEST_REMOTE_EPOCH" -gt "$REGISTRY_CANDIDATE_EPOCH" ]; then echo "Registry is stale on remote: candidate epoch $REGISTRY_CANDIDATE_EPOCH, newest checkpoint epoch $LATEST_REMOTE_EPOCH" >&2; exit 1; fi',
            ]
        )
    command_parts.extend(
        [
            "for CHECKPOINT_ROOT in " + " ".join(shlex.quote(root) for root in REMOTE_CHECKPOINT_ROOTS) + "; do",
            '  [ -d "$CHECKPOINT_ROOT" ] || continue',
            '  find "$CHECKPOINT_ROOT" \\( -type f -o -type l \\) -name "*.pt" -print0 | while IFS= read -r -d "" CHECKPOINT_PATH; do',
            '    SHOULD_KEEP=0',
            "    for KEEP_PATH in " + " ".join(f'"$KEEP_{index}"' for index in range(len(retained))) + "; do",
            '      [ -n "$KEEP_PATH" ] || continue',
            '      if [ "$CHECKPOINT_PATH" = "$KEEP_PATH" ]; then SHOULD_KEEP=1; break; fi',
            "    done",
            '    if [ "$SHOULD_KEEP" -eq 0 ]; then rm -f "$CHECKPOINT_PATH"; fi',
            "  done",
            "done",
            f'BEST_LINK={shlex.quote(f"{selected_dir}/best_model.pt")}',
            'if [ "$SELECTED_CHECKPOINT" != "$BEST_LINK" ]; then rm -f "$BEST_LINK"; ln -sfn "$(basename "$SELECTED_CHECKPOINT")" "$BEST_LINK"; fi',
            'echo "retained=' + ",".join(retained) + '"',
        ]
    )
    return _bash_command(command_parts)


def execute_remote_command_over_ssh(client: VastClient, instance_id: int, remote_command: str) -> int:
    host, port = get_connection_info(client, instance_id)
    return run_local_command(_ssh_command(host, port, remote_command))


def sync_prune(config_path: str, instance_id: int) -> int:
    lock_handle = try_acquire_sync_prune_lock()
    if lock_handle is None:
        sync_log("another sync-prune is already running; skipped")
        return 0

    try:
        sync_log(f"starting config={config_path} instance_id={instance_id}")
        config_absolute = str(_resolve_repo_path(config_path))
        api_key = require_env("VAST_API_KEY")
        client = VastClient(api_key)
        host, port = get_connection_info(client, instance_id)
        config = load_toml(config_absolute)
        remote_root = config.get("workspace", {}).get("remote_root", "/workspace/mirip_v2")

        sync_log("pulling reports")
        pull_result = pull_sync_prune_artifacts(host=host, port=port, remote_root=remote_root)
        if pull_result != 0:
            sync_log(f"initial artifact pull failed exit_code={pull_result}")
            return pull_result

        if not remote_path_exists(host, port, remote_root, TRAIN_FULL_REGISTRY):
            clear_local_sync_cache(
                [
                    TRAIN_FULL_REGISTRY,
                    TRAIN_FULL_CANDIDATE_REPORT,
                    TRAIN_FULL_TRAIN_REPORT,
                ]
            )
            sync_log("remote registry missing; cleared stale local registry cache and skipped prune")
            return 0

        try:
            registry = load_postprocess_registry(TRAIN_FULL_REGISTRY)
        except SystemExit as exc:
            sync_log(f"registry unavailable; skipped prune reason={exc}")
            return 0
        checkpoints_root = _resolve_train_path(CHECKPOINTS_REL_DIR) / TRAIN_MODEL_SLUG
        if registry_is_stale_for_local_checkpoints(registry, checkpoints_root):
            sync_log("registry older than newest local checkpoint; skipped prune")
            return 0

        try:
            retained_checkpoints = resolve_retained_checkpoints(registry)
        except SystemExit as exc:
            sync_log(f"registry retained checkpoints unavailable; skipped prune reason={exc}")
            return 0
        selected_checkpoint = str(registry["selected_best_checkpoint_after_compare"])
        current_candidate_checkpoint = registry.get("current_candidate_checkpoint")

        if not remote_path_exists(host, port, remote_root, selected_checkpoint):
            sync_log(f"selected checkpoint missing remotely; skipped prune checkpoint={selected_checkpoint}")
            return 0
        if current_candidate_checkpoint and not remote_path_exists(host, port, remote_root, str(current_candidate_checkpoint)):
            sync_log(
                "registry candidate checkpoint missing remotely; likely stale cache after restart; skipped prune "
                f"checkpoint={current_candidate_checkpoint}"
            )
            return 0

        command = build_remote_prune_command(
            remote_root,
            selected_checkpoint,
            retained_checkpoints,
            str(current_candidate_checkpoint) if current_candidate_checkpoint else None,
        )
        sync_log(
            "executing remote prune "
            f"selected={selected_checkpoint} retained={','.join(retained_checkpoints)}"
        )
        result = execute_remote_command_over_ssh(client, instance_id, command)
        sync_log(f"finished exit_code={result}")
        return result
    finally:
        release_sync_prune_lock(lock_handle)


def build_launch_agent_payload(config_path: str) -> dict[str, object]:
    config_absolute = str(_resolve_repo_path(config_path))
    LAUNCH_AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LAUNCH_AGENT_LOG_DIR / "launchd.stdout.log"
    stderr_path = LAUNCH_AGENT_LOG_DIR / "launchd.stderr.log"
    return {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [
            sys.executable,
            str(SCRIPT_DIR / "vast_ai_training_runner.py"),
            "sync-prune",
            "--config",
            config_absolute,
        ],
        "WorkingDirectory": str(REPO_ROOT),
        "RunAtLoad": True,
        "StartInterval": LAUNCH_AGENT_START_INTERVAL,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
    }


def install_launch_agent(config_path: str) -> int:
    LAUNCH_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LAUNCH_AGENT_PATH.open("wb") as handle:
        plistlib.dump(build_launch_agent_payload(config_path), handle)

    if LAUNCH_AGENT_PATH.exists():
        run_local_command(["launchctl", "unload", str(LAUNCH_AGENT_PATH)])
    load_result = run_local_command(["launchctl", "load", str(LAUNCH_AGENT_PATH)])
    if load_result != 0:
        return load_result
    print(str(LAUNCH_AGENT_PATH))
    return 0


def uninstall_launch_agent() -> int:
    if LAUNCH_AGENT_PATH.exists():
        unload_result = run_local_command(["launchctl", "unload", str(LAUNCH_AGENT_PATH)])
        if unload_result != 0:
            return unload_result
        LAUNCH_AGENT_PATH.unlink()
    print(str(LAUNCH_AGENT_PATH))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mirip_v2 training stages on Vast.ai instances.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=TRAIN_CONFIG_PATH)
    common.add_argument(
        "--stage",
        choices=[
            "bootstrap",
            "validate-upload",
            "build-pairs-legacy-aligned",
            "re-evaluate-baseline",
            "frozen-ablation",
            "select-ablation-winner",
            "unfreeze-ablation",
            "select-overall-winner",
            "full-fresh",
        ],
        required=True,
    )

    subparsers.add_parser("print-command", parents=[common], help="Print the remote command for a stage")

    execute_cmd = subparsers.add_parser("execute-stage", parents=[common], help="Execute a stage on an existing instance")
    execute_cmd.add_argument("--instance-id", type=int)
    execute_cmd.add_argument("--follow", action="store_true")
    execute_cmd.add_argument("--follow-delay", type=int, default=5)

    pull_cmd = subparsers.add_parser("pull-artifacts", help="Download output_model checkpoints/logs/anchors from an instance")
    pull_cmd.add_argument("--config", default=TRAIN_CONFIG_PATH)
    pull_cmd.add_argument("--instance-id", type=int)

    sync_cmd = subparsers.add_parser("sync-prune", help="Pull artifacts, then prune remote checkpoints using the postprocess registry.")
    sync_cmd.add_argument("--config", default=TRAIN_CONFIG_PATH)
    sync_cmd.add_argument("--instance-id", type=int)

    install_cmd = subparsers.add_parser("install-launch-agent", help="Install a 15-minute launchd sync/prune job.")
    install_cmd.add_argument("--config", default=TRAIN_CONFIG_PATH)

    subparsers.add_parser("uninstall-launch-agent", help="Remove the launchd sync/prune job.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file()

    if args.command == "pull-artifacts":
        return pull_artifacts(args.config, resolve_instance_id(args.instance_id))
    if args.command == "sync-prune":
        return sync_prune(args.config, resolve_instance_id(args.instance_id))
    if args.command == "install-launch-agent":
        return install_launch_agent(args.config)
    if args.command == "uninstall-launch-agent":
        return uninstall_launch_agent()

    config = load_toml(str(_resolve_repo_path(args.config)))
    remote_root = config.get("workspace", {}).get("remote_root", "/workspace/mirip_v2")
    command = build_stage_command(args.stage, remote_root)

    if args.command == "print-command":
        print(command)
        return 0

    api_key = require_env("VAST_API_KEY")
    client = VastClient(api_key)
    return execute_stage(
        client,
        resolve_instance_id(args.instance_id),
        command,
        args.follow,
        args.follow_delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
