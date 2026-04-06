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
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(SCRIPT_DIR))

from vast_ai_control import (  # noqa: E402
    VastClient,
    get_connection_info,
    get_workspace_config,
    load_toml,
    normalize_path,
    require_env,
    run_local_command,
)

TRAIN_ROOT = "train"
DISTILL_DIR = f"{TRAIN_ROOT}/distillation"
DISTILL_CONFIG_PATH = f"{TRAIN_ROOT}/configs/vast_dinov3_vitl_distill_5090.toml"
TRAIN_REPORTS_DIR = f"{TRAIN_ROOT}/reports"
TRAIN_CHECKPOINTS_DIR = f"{TRAIN_ROOT}/checkpoints"
ENV_PATH = ROOT / ".env"


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


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
        os.environ[key] = value.strip().strip('"').strip("'")


def resolve_instance_id(instance_id: int | None) -> int:
    if instance_id is not None:
        return instance_id
    value = os.getenv("VAST_INSTANCE_ID")
    if not value:
        raise SystemExit("Missing Vast instance id. Pass --instance-id or set VAST_INSTANCE_ID in .env.")
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid VAST_INSTANCE_ID value: {value!r}") from exc


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
        f"python -m pip install -r {DISTILL_DIR}/requirements.txt",
        "mkdir -p "
        + " ".join(
            [
                f"{TRAIN_CHECKPOINTS_DIR}/distill_vitl",
                f"{TRAIN_REPORTS_DIR}/distillation",
            ]
        ),
    ]


def _validate_data_parts(remote_root: str) -> list[str]:
    python_bin = _remote_python(remote_root)
    return [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_root)}/{DISTILL_DIR}",
        f"{python_bin} train.py --config configs/vitl_distill.yaml --validate-data --report reports/distillation/validate_data.json",
    ]


def _smoke_distill_parts(remote_root: str) -> list[str]:
    python_bin = _remote_python(remote_root)
    return [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_root)}/{DISTILL_DIR}",
        f"{python_bin} train.py --config configs/vitl_distill.yaml --smoke",
    ]


def _full_distill_parts(remote_root: str) -> list[str]:
    python_bin = _remote_python(remote_root)
    return [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_root)}/{DISTILL_DIR}",
        f"{python_bin} train.py --config configs/vitl_distill.yaml",
    ]


def build_stage_command(stage: str, remote_root: str) -> str:
    if stage == "bootstrap":
        return _bash_command(_bootstrap_parts(remote_root))
    if stage == "validate-data":
        return _bash_command(_validate_data_parts(remote_root))
    if stage == "smoke-distill":
        return _bash_command(_smoke_distill_parts(remote_root))
    if stage == "full-distill":
        return _bash_command(_full_distill_parts(remote_root))
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
    privkey_path = normalize_path(os.getenv("VAST_SSH_PRIVKEY_PATH"))

    ssh_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no"
    rsync_prefix = [
        "rsync",
        "-az",
        "-e",
        ssh_cmd if not privkey_path else f"{ssh_cmd} -i {shlex.quote(str(privkey_path))}",
    ]
    for artifact_dir in ("checkpoints", "reports"):
        cmd = list(rsync_prefix)
        remote = f"root@{host}:{remote_root}/{TRAIN_ROOT}/{artifact_dir}/"
        local = str(ROOT / artifact_dir)
        cmd.extend([remote, local])
        result = run_local_command(cmd)
        if result != 0:
            return result
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mirip_v2 distillation stages on Vast.ai instances.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DISTILL_CONFIG_PATH)
    common.add_argument(
        "--stage",
        choices=["bootstrap", "validate-data", "smoke-distill", "full-distill"],
        required=True,
    )

    subparsers.add_parser("print-command", parents=[common], help="Print the remote command for a stage.")

    execute_cmd = subparsers.add_parser("execute-stage", parents=[common], help="Execute a stage on an existing instance.")
    execute_cmd.add_argument("--instance-id", type=int)
    execute_cmd.add_argument("--follow", action="store_true")
    execute_cmd.add_argument("--follow-delay", type=int, default=5)

    pull_cmd = subparsers.add_parser("pull-artifacts", help="Download checkpoints/reports from an instance.")
    pull_cmd.add_argument("--config", default=DISTILL_CONFIG_PATH)
    pull_cmd.add_argument("--instance-id", type=int)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file()

    if args.command == "pull-artifacts":
        return pull_artifacts(args.config, resolve_instance_id(args.instance_id))

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
