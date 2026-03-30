#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

TRAIN_ROOT = "train"
TRAINING_DIR = f"{TRAIN_ROOT}/training"
TRAIN_DATA_DIR = f"{TRAIN_ROOT}/data"
TRAIN_METADATA_DIR = f"{TRAIN_DATA_DIR}/metadata"
TRAIN_RAW_IMAGES_DIR = f"{TRAIN_DATA_DIR}/raw_images"
TRAINING_DATA_DIR = f"{TRAINING_DIR}/data"
TRAIN_CONFIG_PATH = f"{TRAIN_ROOT}/configs/vast_rtx_pro_4500_blackwell_32gb_ondemand.toml"
TRAIN_REPORTS_DIR = f"{TRAIN_ROOT}/reports"
TRAIN_CHECKPOINTS_DIR = f"{TRAIN_ROOT}/checkpoints"
TRAIN_ANCHORS_DIR = f"{TRAIN_ROOT}/anchors"
TRAIN_NUM_WORKERS = 8
TRAIN_PREFETCH_FACTOR = 4

from vast_ai_control import (  # noqa: E402
    VastClient,
    get_connection_info,
    get_workspace_config,
    load_toml,
    normalize_path,
    require_env,
    run_local_command,
)


def _remote_python(remote_root: str) -> str:
    return f"{remote_root}/.venv/bin/python"


def _bash_command(parts: list[str]) -> str:
    script = "; ".join(parts)
    return f"bash -lc {shlex.quote(script)}"


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
                f"{TRAIN_CHECKPOINTS_DIR}/dinov3_vitl16",
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
        f"{python_bin} {TRAINING_DIR}/validate_training_readiness.py --mode prepared --metadata-dir data/metadata --image-root data --manifest training/data/snapshot_manifest.csv --prepared-dir training/data --baseline-readiness-report reports/readiness_report.json --baseline-snapshot-report reports/snapshot_report.json --total-pairs 50000 --max-appearances 30 --report reports/prepared_readiness_report.json",
    ]


def _build_validate_upload_command(remote_root: str) -> str:
    return _bash_command(_validate_upload_parts(remote_root))


def _build_smoke_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    checkpoint_dir = "checkpoints/dinov3_vitl16/smoke"
    best_checkpoint = f"{checkpoint_dir}/best_model.pt"
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}",
            f"{python_bin} {TRAINING_DIR}/train_dinov3.py --pairs-train training/data/pairs_train.csv --pairs-val training/data/pairs_val.csv --image-root data --output-dir {checkpoint_dir} --epochs 1 --batch-size 8 --gradient-accumulation-steps 8 --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --report reports/dinov3_vitl16_smoke_train.json",
            f"{python_bin} {TRAINING_DIR}/build_anchors_dinov3.py --checkpoint {best_checkpoint} --metadata training/data/metadata_train.csv --image-root data --output anchors/anchors.pt --report reports/dinov3_vitl16_smoke_anchors.json",
            f"{python_bin} {TRAINING_DIR}/evaluate_dinov3.py --checkpoint {best_checkpoint} --pairs-val training/data/pairs_val.csv --image-root data --anchors anchors/anchors.pt --metadata-eval training/data/metadata_val.csv --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --output reports/dinov3_vitl16_smoke.json",
        ]
    )


def _build_full_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    checkpoint_dir = "checkpoints/dinov3_vitl16/full"
    best_checkpoint = f"{checkpoint_dir}/best_model.pt"
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}",
            f"{python_bin} {TRAINING_DIR}/train_dinov3.py --pairs-train training/data/pairs_train.csv --pairs-val training/data/pairs_val.csv --image-root data --output-dir {checkpoint_dir} --epochs 50 --batch-size 8 --gradient-accumulation-steps 8 --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --report reports/dinov3_vitl16_full_train.json",
            f"{python_bin} {TRAINING_DIR}/build_anchors_dinov3.py --checkpoint {best_checkpoint} --metadata training/data/metadata_train.csv --image-root data --output anchors/anchors.pt --report reports/dinov3_vitl16_full_anchors.json",
            f"{python_bin} {TRAINING_DIR}/evaluate_dinov3.py --checkpoint {best_checkpoint} --pairs-val training/data/pairs_val.csv --image-root data --anchors anchors/anchors.pt --metadata-eval training/data/metadata_val.csv --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --output reports/dinov3_vitl16_full.json",
        ]
    )


def build_stage_command(stage: str, remote_root: str) -> str:
    if stage == "bootstrap":
        return _build_bootstrap_command(remote_root)
    if stage == "validate-upload":
        return _build_validate_upload_command(remote_root)
    if stage == "smoke":
        return _build_smoke_command(remote_root)
    if stage == "full":
        return _build_full_command(remote_root)
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
    workspace_cfg = get_workspace_config(config_path)
    remote_root = workspace_cfg.get("remote_root", "/workspace/mirip_v2")
    privkey_path = normalize_path(os.getenv("VAST_SSH_PRIVKEY_PATH"))

    ssh_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no"
    rsync_prefix = [
        "rsync",
        "-az",
        "-e",
        ssh_cmd if not privkey_path else f"{ssh_cmd} -i {shlex.quote(str(privkey_path))}",
    ]
    for artifact_dir in ("checkpoints", "reports", "anchors"):
        cmd = list(rsync_prefix)
        remote = f"root@{host}:{remote_root}/{TRAIN_ROOT}/{artifact_dir}/"
        local = str(ROOT / artifact_dir)
        cmd.extend([remote, local])
        result = run_local_command(cmd)
        if result != 0:
            return result
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mirip_v2 training stages on Vast.ai instances.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=TRAIN_CONFIG_PATH)
    common.add_argument("--stage", choices=["bootstrap", "validate-upload", "smoke", "full"], required=True)

    print_cmd = subparsers.add_parser("print-command", parents=[common], help="Print the remote command for a stage")

    execute_cmd = subparsers.add_parser("execute-stage", parents=[common], help="Execute a stage on an existing instance")
    execute_cmd.add_argument("--instance-id", type=int, required=True)
    execute_cmd.add_argument("--follow", action="store_true")
    execute_cmd.add_argument("--follow-delay", type=int, default=5)

    pull_cmd = subparsers.add_parser("pull-artifacts", help="Download checkpoints/reports/anchors from an instance")
    pull_cmd.add_argument("--config", default=TRAIN_CONFIG_PATH)
    pull_cmd.add_argument("--instance-id", type=int, required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "pull-artifacts":
        return pull_artifacts(args.config, args.instance_id)

    config = load_toml(args.config)
    remote_root = config.get("workspace", {}).get("remote_root", "/workspace/mirip_v2")
    command = build_stage_command(args.stage, remote_root)

    if args.command == "print-command":
        print(command)
        return 0

    api_key = require_env("VAST_API_KEY")
    client = VastClient(api_key)
    return execute_stage(client, args.instance_id, command, args.follow, args.follow_delay)


if __name__ == "__main__":
    raise SystemExit(main())
