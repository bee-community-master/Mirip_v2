#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import plistlib
import shlex
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
TRAIN_REPORTS_DIR = f"{TRAIN_ROOT}/reports"
TRAIN_CHECKPOINTS_DIR = f"{TRAIN_ROOT}/checkpoints"
TRAIN_ANCHORS_DIR = f"{TRAIN_ROOT}/anchors"
REPORTS_REL_DIR = "reports"
CHECKPOINTS_REL_DIR = "checkpoints"
ANCHORS_REL_DIR = "anchors"
TRAIN_MODEL_NAME = "PIA-SPACE-LAB/dinov3-vit7b16-pretrain-lvd1689m"
TRAIN_MODEL_SLUG = "dinov3_vit7b16"
TRAIN_BATCH_SIZE = 16
TRAIN_GRADIENT_ACCUMULATION_STEPS = 4
TRAIN_NUM_WORKERS = 8
TRAIN_PREFETCH_FACTOR = 4
TRAIN_FULL_TRAIN_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full_train.json"
TRAIN_FULL_TRAIN_REPORT_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_full_train.json"
TRAIN_FULL_CANDIDATE_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full_candidate.json"
TRAIN_FULL_CANDIDATE_ANCHORS = f"{ANCHORS_REL_DIR}/{TRAIN_MODEL_SLUG}_candidate_anchors.pt"
TRAIN_FULL_FINAL_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full.json"
TRAIN_FULL_FINAL_ANCHORS = f"{ANCHORS_REL_DIR}/{TRAIN_MODEL_SLUG}_anchors.pt"
TRAIN_FULL_REGISTRY = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_postprocess_registry.json"
TRAIN_FULL_REGISTRY_FILE = f"{TRAIN_REPORTS_DIR}/{TRAIN_MODEL_SLUG}_postprocess_registry.json"
TRAIN_FULL_CANDIDATE_ANCHOR_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full_candidate_anchors.json"
TRAIN_FULL_FINAL_ANCHOR_REPORT = f"{REPORTS_REL_DIR}/{TRAIN_MODEL_SLUG}_full_anchors.json"
ENV_PATH = ROOT / ".env"
LAUNCH_AGENT_LABEL = "com.mirip.vast-checkpoint-sync"
LAUNCH_AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
SYNC_LOG_DIR = ROOT / "reports" / "vast_sync"

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
    retained = payload.get("retained_checkpoints")
    if not selected or not isinstance(retained, list) or not retained:
        raise SystemExit(f"Postprocess registry is incomplete: {registry_path}")
    if not str(selected).startswith("checkpoints/"):
        raise SystemExit(f"Selected checkpoint must stay under train/checkpoints: {selected}")
    return payload


def _remote_python(remote_root: str) -> str:
    return f"{remote_root}/.venv/bin/python"


def _bash_command(parts: list[str]) -> str:
    script = "; ".join(parts)
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
        f"{python_bin} {TRAINING_DIR}/validate_training_readiness.py --mode prepared --metadata-dir data/metadata --image-root data --manifest training/data/snapshot_manifest.csv --prepared-dir training/data --baseline-readiness-report reports/readiness_report.json --baseline-snapshot-report reports/snapshot_report.json --total-pairs 50000 --max-appearances 30 --report reports/prepared_readiness_report.json",
    ]


def _build_validate_upload_command(remote_root: str) -> str:
    return _bash_command(_validate_upload_parts(remote_root))


def _build_smoke_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    checkpoint_dir = f"checkpoints/{TRAIN_MODEL_SLUG}/smoke"
    best_checkpoint = f"{checkpoint_dir}/best_model.pt"
    anchors_path = f"anchors/{TRAIN_MODEL_SLUG}_anchors.pt"
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}",
            f"{python_bin} {TRAINING_DIR}/train_dinov3.py --pairs-train training/data/pairs_train.csv --pairs-val training/data/pairs_val.csv --image-root data --output-dir {checkpoint_dir} --model-name {shlex.quote(TRAIN_MODEL_NAME)} --backbone-dtype auto --epochs 1 --batch-size {TRAIN_BATCH_SIZE} --gradient-accumulation-steps {TRAIN_GRADIENT_ACCUMULATION_STEPS} --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --report reports/{TRAIN_MODEL_SLUG}_smoke_train.json",
            f"{python_bin} {TRAINING_DIR}/build_anchors_dinov3.py --checkpoint {best_checkpoint} --metadata training/data/metadata_train.csv --image-root data --output {anchors_path} --report reports/{TRAIN_MODEL_SLUG}_smoke_anchors.json",
            f"{python_bin} {TRAINING_DIR}/evaluate_dinov3.py --checkpoint {best_checkpoint} --pairs-val training/data/pairs_val.csv --image-root data --anchors {anchors_path} --metadata-eval training/data/metadata_val.csv --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --output reports/{TRAIN_MODEL_SLUG}_smoke.json",
        ]
    )


def _build_full_command(remote_root: str) -> str:
    python_bin = _remote_python(remote_root)
    checkpoint_dir = f"checkpoints/{TRAIN_MODEL_SLUG}/full"
    smoke_checkpoint = f"checkpoints/{TRAIN_MODEL_SLUG}/smoke/best_model.pt"
    current_checkpoint_value = _json_value_command(
        python_bin,
        TRAIN_FULL_TRAIN_REPORT_FILE,
        "payload.get('paths', {}).get('latest_completed_checkpoint_relative') or payload.get('latest_completed_checkpoint')",
        "latest completed checkpoint missing from training summary",
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
            f"{python_bin} {TRAINING_DIR}/train_dinov3.py --pairs-train training/data/pairs_train.csv --pairs-val training/data/pairs_val.csv --image-root data --output-dir {checkpoint_dir} --model-name {shlex.quote(TRAIN_MODEL_NAME)} --backbone-dtype auto --resume-from {smoke_checkpoint} --epochs 50 --batch-size {TRAIN_BATCH_SIZE} --gradient-accumulation-steps {TRAIN_GRADIENT_ACCUMULATION_STEPS} --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --report {TRAIN_FULL_TRAIN_REPORT}",
            f'CURRENT_CHECKPOINT="{current_checkpoint_value}"',
            f"{python_bin} {TRAINING_DIR}/build_anchors_dinov3.py --checkpoint \"$CURRENT_CHECKPOINT\" --metadata training/data/metadata_train.csv --image-root data --output {TRAIN_FULL_CANDIDATE_ANCHORS} --report {TRAIN_FULL_CANDIDATE_ANCHOR_REPORT}",
            f"{python_bin} {TRAINING_DIR}/evaluate_dinov3.py --checkpoint \"$CURRENT_CHECKPOINT\" --pairs-val training/data/pairs_val.csv --image-root data --anchors {TRAIN_FULL_CANDIDATE_ANCHORS} --metadata-eval training/data/metadata_val.csv --num-workers {TRAIN_NUM_WORKERS} --prefetch-factor {TRAIN_PREFETCH_FACTOR} --output {TRAIN_FULL_CANDIDATE_REPORT}",
            f"{python_bin} {TRAINING_DIR}/select_postprocess_best.py --current-checkpoint \"$CURRENT_CHECKPOINT\" --current-report {TRAIN_FULL_CANDIDATE_REPORT} --output-registry {TRAIN_FULL_REGISTRY}",
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
    for artifact_dir in ("checkpoints", "reports", "anchors"):
        cmd = list(rsync_prefix)
        remote = f"root@{host}:{remote_root}/{TRAIN_ROOT}/{artifact_dir}/"
        local = str(ROOT / artifact_dir)
        cmd.extend([remote, local])
        result = run_local_command(cmd)
        if result != 0:
            return result
    return 0


def build_remote_prune_command(remote_root: str, retained_checkpoint: str) -> str:
    retained = retained_checkpoint.strip()
    selected_dir = str(Path(retained).parent).replace("\\", "/")
    checkpoint_root = f"checkpoints/{TRAIN_MODEL_SLUG}"
    return _bash_command(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_root)}/{TRAIN_ROOT}",
            f'SELECTED_CHECKPOINT={shlex.quote(retained)}',
            'if [ ! -f "$SELECTED_CHECKPOINT" ]; then echo "Selected checkpoint missing: $SELECTED_CHECKPOINT" >&2; exit 1; fi',
            f'find {shlex.quote(checkpoint_root)} \\( -type f -o -type l \\) -name "*.pt" ! -path "$SELECTED_CHECKPOINT" -delete',
            f'BEST_LINK={shlex.quote(f"{selected_dir}/best_model.pt")}',
            'if [ "$SELECTED_CHECKPOINT" != "$BEST_LINK" ]; then rm -f "$BEST_LINK"; ln -sfn "$(basename "$SELECTED_CHECKPOINT")" "$BEST_LINK"; fi',
            'echo "retained=$SELECTED_CHECKPOINT"',
        ]
    )


def sync_prune(config_path: str, instance_id: int) -> int:
    config_absolute = str(_resolve_repo_path(config_path))
    pull_result = pull_artifacts(config_absolute, instance_id)
    if pull_result != 0:
        return pull_result

    registry = load_postprocess_registry(TRAIN_FULL_REGISTRY)
    retained_checkpoint = str(registry["selected_best_checkpoint_after_compare"])

    config = load_toml(config_absolute)
    remote_root = config.get("workspace", {}).get("remote_root", "/workspace/mirip_v2")
    command = build_remote_prune_command(remote_root, retained_checkpoint)
    api_key = require_env("VAST_API_KEY")
    client = VastClient(api_key)
    return execute_stage(client, instance_id, command, follow=True, follow_delay=3)


def build_launch_agent_payload(config_path: str) -> dict[str, object]:
    config_absolute = str(_resolve_repo_path(config_path))
    SYNC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = SYNC_LOG_DIR / "launchd.stdout.log"
    stderr_path = SYNC_LOG_DIR / "launchd.stderr.log"
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
        "StartInterval": 3600,
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
    common.add_argument("--stage", choices=["bootstrap", "validate-upload", "smoke", "full"], required=True)

    subparsers.add_parser("print-command", parents=[common], help="Print the remote command for a stage")

    execute_cmd = subparsers.add_parser("execute-stage", parents=[common], help="Execute a stage on an existing instance")
    execute_cmd.add_argument("--instance-id", type=int)
    execute_cmd.add_argument("--follow", action="store_true")
    execute_cmd.add_argument("--follow-delay", type=int, default=5)

    pull_cmd = subparsers.add_parser("pull-artifacts", help="Download checkpoints/reports/anchors from an instance")
    pull_cmd.add_argument("--config", default=TRAIN_CONFIG_PATH)
    pull_cmd.add_argument("--instance-id", type=int)

    sync_cmd = subparsers.add_parser("sync-prune", help="Pull artifacts, then prune remote checkpoints using the postprocess registry.")
    sync_cmd.add_argument("--config", default=TRAIN_CONFIG_PATH)
    sync_cmd.add_argument("--instance-id", type=int)

    install_cmd = subparsers.add_parser("install-launch-agent", help="Install a 1-hour launchd sync/prune job.")
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
