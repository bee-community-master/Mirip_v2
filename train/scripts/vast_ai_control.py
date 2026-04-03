#!/usr/bin/env python3
"""
Minimal Vast.ai control helper for Mirip_v2.

This script intentionally focuses on infrastructure preparation:
- search offers
- create/show/wait/manage/destroy instances
- attach SSH key
- execute remote commands
- sync local preparation files with rsync
- open SSH sessions using the connection info returned by Vast.ai
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://console.vast.ai/api/v0"
REPO_ROOT = Path(__file__).resolve().parents[2]


class VastApiError(RuntimeError):
    """Raised when the Vast.ai API returns an error response."""


def _import_toml_module() -> Any:
    for module_name in ("tomllib", "tomli", "pip._vendor.tomli"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "No TOML parser is available. Install tomli for Python < 3.11 or use a Python build with tomllib."
    )


tomllib = _import_toml_module()


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def normalize_path(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(path).expanduser().resolve()


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


class VastClient:
    def __init__(self, api_key: str, api_base: str = API_BASE) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.api_base}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise VastApiError(f"{method} {url} failed: {exc.code} {raw}") from exc
        except urllib.error.URLError as exc:
            raise VastApiError(f"{method} {url} failed: {exc}") from exc

        if not raw:
            return {}
        return json.loads(raw)

    def search_offers(self, filters: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/bundles/", filters)

    def create_instance(self, offer_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/asks/{offer_id}/", payload)

    def show_instance(self, instance_id: int) -> dict[str, Any]:
        return self._request("GET", f"/instances/{instance_id}/")

    def manage_instance(
        self,
        instance_id: int,
        *,
        state: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if state is not None:
            payload["state"] = state
        if label is not None:
            payload["label"] = label
        return self._request("PUT", f"/instances/{instance_id}/", payload)

    def destroy_instance(self, instance_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/instances/{instance_id}/")

    def attach_ssh_key(self, instance_id: int, ssh_key: str) -> dict[str, Any]:
        return self._request("POST", f"/instances/{instance_id}/ssh/", {"ssh_key": ssh_key})

    def execute(self, instance_id: int, command: str) -> dict[str, Any]:
        return self._request("PUT", f"/instances/command/{instance_id}/", {"command": command})

    def fetch_text(self, url: str) -> str:
        request = urllib.request.Request(url=url, headers={"Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace")


def build_offer_filters(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("offer_filters", {})
    filters: dict[str, Any] = {
        "limit": section.get("limit", 5),
    }

    if "gpu_name" in section:
        filters["gpu_name"] = {"in": list(section["gpu_name"])}
    if "num_gpus" in section:
        filters["num_gpus"] = {"gte": int(section["num_gpus"])}
    if "gpu_ram_gb" in section:
        filters["gpu_ram"] = {"gte": int(section["gpu_ram_gb"]) * 1000}
    if "reliability" in section:
        filters["reliability"] = {"gte": float(section["reliability"])}
    if "verified" in section:
        filters["verified"] = {"eq": bool(section["verified"])}
    if "rentable" in section:
        filters["rentable"] = {"eq": bool(section["rentable"])}
    if "type" in section:
        filters["type"] = section["type"]
    return filters


def build_instance_payload(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("instance", {})
    payload: dict[str, Any] = {
        "image": section["image"],
        "disk": int(section.get("disk_gb", 60)),
        "runtype": section.get("runtype", "ssh_direct"),
        "label": section.get("label", "mirip-v2"),
        "cancel_unavail": bool(section.get("cancel_unavail", True)),
    }
    if "price" in section:
        payload["price"] = float(section["price"])
    if "onstart" in section:
        payload["onstart"] = section["onstart"].strip()

    env = dict(section.get("env", {}))
    for port in section.get("ports", []):
        env[f"-p {port}"] = "1"
    if env:
        payload["env"] = env

    return payload


def choose_offer(search_result: dict[str, Any], requested_offer_id: int | None) -> dict[str, Any]:
    offers = search_result.get("offers") or search_result.get("results") or []
    if not offers:
        raise SystemExit("No offers matched the current filter.")
    if requested_offer_id is not None:
        for offer in offers:
            if int(offer["id"]) == requested_offer_id:
                return offer
        raise SystemExit(f"Offer id {requested_offer_id} not found in current search result.")
    return offers[0]


def extract_instance(search_result: dict[str, Any]) -> dict[str, Any]:
    instance = search_result.get("instances")
    if instance:
        return instance
    return search_result


def wait_for_status(client: VastClient, instance_id: int, status: str, interval: int, timeout: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.show_instance(instance_id)
        instance = extract_instance(payload)
        actual = instance.get("actual_status") or instance.get("cur_state")
        if actual == status:
            return payload
        print(f"[wait] instance={instance_id} actual_status={actual!r} target={status!r}")
        time.sleep(interval)
    raise SystemExit(f"Timed out waiting for instance {instance_id} to reach status {status!r}")


def read_public_key(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def get_connection_info(client: VastClient, instance_id: int) -> tuple[str, int]:
    payload = client.show_instance(instance_id)
    instance = extract_instance(payload)
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    if not host or not port:
        raise SystemExit(
            "SSH host/port are not available yet. Wait for the instance to finish loading first."
        )
    return str(host), int(port)


def get_workspace_config(config_path: str | Path) -> dict[str, Any]:
    return load_toml(config_path).get("workspace", {})


def run_local_command(cmd: list[str], cwd: Path | None = None) -> int:
    print("$", shlex.join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return completed.returncode


def sync_workspace(args: argparse.Namespace, client: VastClient) -> int:
    workspace_cfg = get_workspace_config(args.config)
    sync_paths = workspace_cfg.get("sync_paths", [])
    if not sync_paths:
        raise SystemExit("No [workspace].sync_paths configured.")
    remote_root = workspace_cfg.get("remote_root", "/workspace/mirip_v2")
    host, port = get_connection_info(client, args.instance_id)

    ssh_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no"
    privkey_path = normalize_path(os.getenv("VAST_SSH_PRIVKEY_PATH"))
    if privkey_path:
        ssh_cmd += f" -i {shlex.quote(str(privkey_path))}"

    remote_target = f"root@{host}:{remote_root}/"
    cmd = [
        "rsync",
        "-az",
        "--relative",
        "-e",
        ssh_cmd,
    ]
    cmd.extend(sync_paths)
    cmd.append(remote_target)

    mkdir_cmd = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
    ]
    if privkey_path:
        mkdir_cmd.extend(["-i", str(privkey_path)])
    mkdir_cmd.extend([f"root@{host}", f"mkdir -p {shlex.quote(remote_root)}"])
    if run_local_command(mkdir_cmd, cwd=REPO_ROOT) != 0:
        return 1
    return run_local_command(cmd, cwd=REPO_ROOT)


def open_ssh(args: argparse.Namespace, client: VastClient) -> int:
    host, port = get_connection_info(client, args.instance_id)
    cmd = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
    ]
    privkey_path = normalize_path(os.getenv("VAST_SSH_PRIVKEY_PATH"))
    if privkey_path:
        cmd.extend(["-i", str(privkey_path)])
    cmd.append(f"{args.user}@{host}")
    return run_local_command(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirip_v2 Vast.ai preparation helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Search offers using a TOML config")
    search.add_argument("--config", required=True)

    create = subparsers.add_parser("create", help="Create an instance from the first matching offer")
    create.add_argument("--config", required=True)
    create.add_argument("--offer-id", type=int, default=None)
    create.add_argument("--attach-ssh", action="store_true")
    create.add_argument("--wait", action="store_true")

    show = subparsers.add_parser("show", help="Show instance details")
    show.add_argument("--instance-id", type=int, required=True)

    wait = subparsers.add_parser("wait", help="Wait for an instance status")
    wait.add_argument("--instance-id", type=int, required=True)
    wait.add_argument("--status", default="running")
    wait.add_argument("--interval", type=int, default=10)
    wait.add_argument("--timeout", type=int, default=900)

    manage = subparsers.add_parser("manage", help="Start/stop/label an instance")
    manage.add_argument("--instance-id", type=int, required=True)
    manage.add_argument("--state", choices=["running", "stopped"], default=None)
    manage.add_argument("--label", default=None)

    destroy = subparsers.add_parser("destroy", help="Destroy an instance")
    destroy.add_argument("--instance-id", type=int, required=True)

    attach = subparsers.add_parser("attach-ssh", help="Attach a local public key")
    attach.add_argument("--instance-id", type=int, required=True)
    attach.add_argument("--pubkey", default=None)

    execute = subparsers.add_parser("execute", help="Execute a remote command through the Vast API")
    execute.add_argument("--instance-id", type=int, required=True)
    execute.add_argument("--command-text", required=True)
    execute.add_argument("--follow", action="store_true")
    execute.add_argument("--follow-delay", type=int, default=5)

    ssh = subparsers.add_parser("ssh", help="Open an SSH session")
    ssh.add_argument("--instance-id", type=int, required=True)
    ssh.add_argument("--user", default="root")

    sync = subparsers.add_parser("sync-workspace", help="Upload configured prep files via rsync")
    sync.add_argument("--config", required=True)
    sync.add_argument("--instance-id", type=int, required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = require_env("VAST_API_KEY")
    client = VastClient(api_key)

    if args.command == "search":
        config = load_toml(args.config)
        result = client.search_offers(build_offer_filters(config))
        offers = result.get("offers", [])
        slim = []
        for offer in offers:
            slim.append(
                {
                    "id": offer.get("id"),
                    "gpu_name": offer.get("gpu_name"),
                    "num_gpus": offer.get("num_gpus"),
                    "gpu_ram": offer.get("gpu_ram"),
                    "reliability": offer.get("reliability"),
                    "verified": offer.get("verified"),
                    "dph_total": offer.get("dph_total"),
                    "machine_id": offer.get("machine_id"),
                }
            )
        print_json({"offers": slim})
        return 0

    if args.command == "create":
        config = load_toml(args.config)
        search_result = client.search_offers(build_offer_filters(config))
        chosen_offer = choose_offer(search_result, args.offer_id)
        create_result = client.create_instance(int(chosen_offer["id"]), build_instance_payload(config))
        print_json({"offer": chosen_offer, "create_result": create_result})

        instance_id = create_result.get("new_contract") or create_result.get("instance_id")
        if not instance_id:
            return 0

        if args.attach_ssh:
            workspace_cfg = config.get("workspace", {})
            pubkey = normalize_path(
                workspace_cfg.get("ssh_public_key_path") or os.getenv("VAST_SSH_PUBKEY_PATH")
            )
            if not pubkey or not pubkey.exists():
                raise SystemExit("Public key path not found. Set VAST_SSH_PUBKEY_PATH or workspace.ssh_public_key_path.")
            attach_result = client.attach_ssh_key(int(instance_id), read_public_key(pubkey))
            print_json({"attach_ssh_result": attach_result})

        if args.wait:
            payload = wait_for_status(client, int(instance_id), "running", interval=10, timeout=900)
            print_json({"instance": extract_instance(payload)})
        return 0

    if args.command == "show":
        print_json(client.show_instance(args.instance_id))
        return 0

    if args.command == "wait":
        payload = wait_for_status(client, args.instance_id, args.status, args.interval, args.timeout)
        print_json({"instance": extract_instance(payload)})
        return 0

    if args.command == "manage":
        print_json(client.manage_instance(args.instance_id, state=args.state, label=args.label))
        return 0

    if args.command == "destroy":
        print_json(client.destroy_instance(args.instance_id))
        return 0

    if args.command == "attach-ssh":
        pubkey_path = normalize_path(args.pubkey or os.getenv("VAST_SSH_PUBKEY_PATH"))
        if not pubkey_path or not pubkey_path.exists():
            raise SystemExit("Public key path not found. Pass --pubkey or set VAST_SSH_PUBKEY_PATH.")
        print_json(client.attach_ssh_key(args.instance_id, read_public_key(pubkey_path)))
        return 0

    if args.command == "execute":
        result = client.execute(args.instance_id, args.command_text)
        print_json(result)
        if args.follow and result.get("result_url"):
            time.sleep(args.follow_delay)
            print(client.fetch_text(result["result_url"]))
        return 0

    if args.command == "ssh":
        return open_ssh(args, client)

    if args.command == "sync-workspace":
        return sync_workspace(args, client)

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
