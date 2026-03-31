from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vast_ai_training_runner.py"
SPEC = importlib.util.spec_from_file_location("vast_ai_training_runner", SCRIPT_PATH)
assert SPEC and SPEC.loader
vast_ai_training_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vast_ai_training_runner)


class VastAiTrainingRunnerTests(unittest.TestCase):
    def test_load_env_file_sets_instance_id_without_overwriting_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("VAST_INSTANCE_ID=1234\nNEW_KEY=value\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"VAST_INSTANCE_ID": "9999"}, clear=False):
                vast_ai_training_runner.load_env_file(env_path)
                self.assertEqual(os.environ["VAST_INSTANCE_ID"], "9999")
                self.assertEqual(os.environ["NEW_KEY"], "value")

    def test_build_remote_prune_command_keeps_selected_and_latest_checkpoints(self) -> None:
        command = vast_ai_training_runner.build_remote_prune_command(
            "/workspace/mirip_v2",
            "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
            [
                "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
                "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
            ],
            "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0009.pt",
        )

        self.assertIn("checkpoint_epoch_0002.pt", command)
        self.assertIn("checkpoint_epoch_0005.pt", command)
        self.assertIn("output_models/checkpoints/dinov3_vit7b16", command)
        self.assertIn("checkpoints/dinov3_vit7b16", command)
        self.assertIn("best_model.pt", command)
        self.assertIn("REGISTRY_CANDIDATE_EPOCH=9", command)
        self.assertIn("LATEST_REMOTE_EPOCH", command)
        self.assertNotIn("then;", command)
        self.assertNotIn("do;", command)
        self.assertNotIn("{find", command)
        self.assertNotIn("2>/dev/null find", command)

    def test_build_launch_agent_payload_targets_sync_prune(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("", encoding="utf-8")
            with mock.patch.object(vast_ai_training_runner, "SYNC_LOG_DIR", Path(temp_dir) / "logs"):
                payload = vast_ai_training_runner.build_launch_agent_payload(str(config_path))

        self.assertEqual(payload["Label"], "com.mirip.vast-checkpoint-sync")
        self.assertIn("sync-prune", payload["ProgramArguments"])
        self.assertIn(str(config_path), payload["ProgramArguments"])
        self.assertEqual(payload["StartInterval"], 900)

    def test_sync_prune_lock_allows_only_one_active_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_dir = Path(temp_dir) / "logs"
            lock_path = lock_dir / "sync-prune.lock"
            with (
                mock.patch.object(vast_ai_training_runner, "SYNC_LOG_DIR", lock_dir),
                mock.patch.object(vast_ai_training_runner, "SYNC_LOCK_PATH", lock_path),
            ):
                first = vast_ai_training_runner.try_acquire_sync_prune_lock()
                self.assertIsNotNone(first)
                second = vast_ai_training_runner.try_acquire_sync_prune_lock()
                self.assertIsNone(second)
                vast_ai_training_runner.release_sync_prune_lock(first)
                third = vast_ai_training_runner.try_acquire_sync_prune_lock()
                self.assertIsNotNone(third)
                vast_ai_training_runner.release_sync_prune_lock(third)

    def test_execute_remote_command_over_ssh_uses_instance_connection_info(self) -> None:
        fake_client = object()
        expected_key_path = str(Path("/tmp/vast_key").resolve())
        with (
            mock.patch.object(vast_ai_training_runner, "get_connection_info", return_value=("ssh1.vast.ai", 31416)),
            mock.patch.object(vast_ai_training_runner, "run_local_command", return_value=0) as run_local_command,
            mock.patch.dict(os.environ, {"VAST_SSH_PRIVKEY_PATH": "/tmp/vast_key"}, clear=False),
        ):
            result = vast_ai_training_runner.execute_remote_command_over_ssh(
                fake_client,
                33831416,
                "bash -lc 'echo ok'",
            )

        self.assertEqual(result, 0)
        run_local_command.assert_called_once_with(
            [
                "ssh",
                "-p",
                "31416",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                expected_key_path,
                "root@ssh1.vast.ai",
                "bash -lc 'echo ok'",
            ]
        )

    def test_pull_sync_prune_artifacts_downloads_retained_legacy_checkpoint_into_output_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            train_root = repo_root / "train"
            train_root.mkdir(parents=True, exist_ok=True)
            best_link = train_root / "output_models" / "checkpoints" / "dinov3_vit7b16" / "full" / "best_model.pt"
            best_link.parent.mkdir(parents=True, exist_ok=True)
            best_link.write_text("stale-link-placeholder", encoding="utf-8")
            recorded_commands: list[list[str]] = []

            def _record_command(cmd: list[str], cwd: Path | None = None) -> int:
                recorded_commands.append(cmd)
                return 0

            with (
                mock.patch.object(vast_ai_training_runner, "ROOT", train_root),
                mock.patch.object(vast_ai_training_runner, "run_local_command", side_effect=_record_command),
                mock.patch.dict(os.environ, {"VAST_SSH_PRIVKEY_PATH": "/tmp/vast_key"}, clear=False),
            ):
                result = vast_ai_training_runner.pull_sync_prune_artifacts(
                    host="ssh1.vast.ai",
                    port=31416,
                    remote_root="/workspace/mirip_v2",
                    retained_checkpoints=["checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt"],
                    selected_checkpoint="checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
                )

        self.assertEqual(result, 0)
        self.assertEqual(len(recorded_commands), 3)
        self.assertFalse(best_link.exists())
        self.assertIn("--partial", recorded_commands[2])
        self.assertIn("--size-only", recorded_commands[2])
        self.assertIn(
            "root@ssh1.vast.ai:/workspace/mirip_v2/train/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
            recorded_commands[2],
        )
        self.assertIn(
            str(train_root / "output_models" / "checkpoints" / "dinov3_vit7b16" / "full" / "checkpoint_epoch_0002.pt"),
            recorded_commands[2],
        )

    def test_load_postprocess_registry_requires_selected_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            registry_path.write_text(json.dumps({"retained_checkpoints": []}), encoding="utf-8")

            with self.assertRaises(SystemExit):
                vast_ai_training_runner.load_postprocess_registry(registry_path)

    def test_resolve_retained_checkpoints_keeps_selected_and_current_candidate(self) -> None:
        retained = vast_ai_training_runner.resolve_retained_checkpoints(
            {
                "selected_best_checkpoint_after_compare": "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
                "current_candidate_checkpoint": "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
                "retained_checkpoints": ["checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt"],
            }
        )

        self.assertEqual(
            retained,
            [
                "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
                "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
            ],
        )

    def test_registry_is_stale_when_newer_checkpoint_exists_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoints_root = Path(temp_dir) / "dinov3_vit7b16" / "full"
            checkpoints_root.mkdir(parents=True, exist_ok=True)
            (checkpoints_root / "checkpoint_epoch_0008.pt").write_text("", encoding="utf-8")
            (checkpoints_root / "checkpoint_epoch_0009.pt").write_text("", encoding="utf-8")

            stale = vast_ai_training_runner.registry_is_stale_for_local_checkpoints(
                {"current_candidate_checkpoint": "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0008.pt"},
                checkpoints_root.parent,
            )

            self.assertTrue(stale)

    def test_sync_prune_downloads_only_selected_best_checkpoint_locally(self) -> None:
        registry = {
            "selected_best_checkpoint_after_compare": "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
            "current_candidate_checkpoint": "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
            "retained_checkpoints": [
                "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
                "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("", encoding="utf-8")
            fake_client = object()
            with (
                mock.patch.object(vast_ai_training_runner, "require_env", return_value="vast-api-key"),
                mock.patch.object(vast_ai_training_runner, "VastClient", return_value=fake_client),
                mock.patch.object(vast_ai_training_runner, "get_connection_info", return_value=("ssh1.vast.ai", 31416)),
                mock.patch.object(vast_ai_training_runner, "load_toml", return_value={"workspace": {"remote_root": "/workspace/mirip_v2"}}),
                mock.patch.object(vast_ai_training_runner, "load_postprocess_registry", return_value=registry),
                mock.patch.object(vast_ai_training_runner, "registry_is_stale_for_local_checkpoints", return_value=False),
                mock.patch.object(vast_ai_training_runner, "remote_path_exists", return_value=True),
                mock.patch.object(vast_ai_training_runner, "build_remote_prune_command", return_value="bash -lc 'echo ok'") as build_remote_prune_command,
                mock.patch.object(vast_ai_training_runner, "execute_remote_command_over_ssh", return_value=0) as execute_remote_command_over_ssh,
                mock.patch.object(vast_ai_training_runner, "pull_sync_prune_artifacts", side_effect=[0, 0]) as pull_sync_prune_artifacts,
            ):
                result = vast_ai_training_runner.sync_prune(str(config_path), 33831416)

        self.assertEqual(result, 0)
        self.assertEqual(pull_sync_prune_artifacts.call_count, 2)
        first_call = pull_sync_prune_artifacts.call_args_list[0]
        second_call = pull_sync_prune_artifacts.call_args_list[1]
        self.assertEqual(first_call.kwargs["host"], "ssh1.vast.ai")
        self.assertIsNone(first_call.kwargs.get("retained_checkpoints"))
        self.assertEqual(
            second_call.kwargs["retained_checkpoints"],
            ["checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt"],
        )
        build_remote_prune_command.assert_called_once_with(
            "/workspace/mirip_v2",
            "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
            [
                "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
                "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
            ],
            "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
        )
        execute_remote_command_over_ssh.assert_called_once_with(fake_client, 33831416, "bash -lc 'echo ok'")

    def test_sync_prune_skips_when_remote_registry_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("", encoding="utf-8")
            with (
                mock.patch.object(vast_ai_training_runner, "require_env", return_value="vast-api-key"),
                mock.patch.object(vast_ai_training_runner, "VastClient", return_value=object()),
                mock.patch.object(vast_ai_training_runner, "get_connection_info", return_value=("ssh1.vast.ai", 31416)),
                mock.patch.object(vast_ai_training_runner, "load_toml", return_value={"workspace": {"remote_root": "/workspace/mirip_v2"}}),
                mock.patch.object(vast_ai_training_runner, "pull_sync_prune_artifacts", return_value=0) as pull_sync_prune_artifacts,
                mock.patch.object(vast_ai_training_runner, "remote_path_exists", return_value=False),
                mock.patch.object(vast_ai_training_runner, "clear_local_sync_cache") as clear_local_sync_cache,
                mock.patch.object(vast_ai_training_runner, "load_postprocess_registry") as load_postprocess_registry,
            ):
                result = vast_ai_training_runner.sync_prune(str(config_path), 33831416)

        self.assertEqual(result, 0)
        pull_sync_prune_artifacts.assert_called_once()
        clear_local_sync_cache.assert_called_once()
        load_postprocess_registry.assert_not_called()

    def test_sync_prune_skips_when_another_run_holds_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("", encoding="utf-8")
            with (
                mock.patch.object(vast_ai_training_runner, "try_acquire_sync_prune_lock", return_value=None),
                mock.patch.object(vast_ai_training_runner, "require_env") as require_env,
            ):
                result = vast_ai_training_runner.sync_prune(str(config_path), 33831416)

        self.assertEqual(result, 0)
        require_env.assert_not_called()

    def test_sync_prune_skips_when_registry_candidate_checkpoint_is_missing_remotely(self) -> None:
        registry = {
            "selected_best_checkpoint_after_compare": "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0004.pt",
            "current_candidate_checkpoint": "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt",
            "retained_checkpoints": [
                "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0004.pt",
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("", encoding="utf-8")
            fake_client = object()
            remote_exists = {
                vast_ai_training_runner.TRAIN_FULL_REGISTRY: True,
                "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0004.pt": True,
                "output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0005.pt": False,
            }

            def _remote_path_exists(host: str, port: int, remote_root: str, relative_path: str) -> bool:
                return remote_exists.get(relative_path, False)

            with (
                mock.patch.object(vast_ai_training_runner, "require_env", return_value="vast-api-key"),
                mock.patch.object(vast_ai_training_runner, "VastClient", return_value=fake_client),
                mock.patch.object(vast_ai_training_runner, "get_connection_info", return_value=("ssh1.vast.ai", 31416)),
                mock.patch.object(vast_ai_training_runner, "load_toml", return_value={"workspace": {"remote_root": "/workspace/mirip_v2"}}),
                mock.patch.object(vast_ai_training_runner, "load_postprocess_registry", return_value=registry),
                mock.patch.object(vast_ai_training_runner, "registry_is_stale_for_local_checkpoints", return_value=False),
                mock.patch.object(vast_ai_training_runner, "remote_path_exists", side_effect=_remote_path_exists),
                mock.patch.object(vast_ai_training_runner, "pull_sync_prune_artifacts", return_value=0) as pull_sync_prune_artifacts,
                mock.patch.object(vast_ai_training_runner, "execute_remote_command_over_ssh") as execute_remote_command_over_ssh,
            ):
                result = vast_ai_training_runner.sync_prune(str(config_path), 33831416)

        self.assertEqual(result, 0)
        pull_sync_prune_artifacts.assert_called_once()
        execute_remote_command_over_ssh.assert_not_called()

    def test_full_stage_command_runs_postprocess_selection_before_final_eval(self) -> None:
        command = vast_ai_training_runner.build_stage_command("full", "/workspace/mirip_v2")

        self.assertIn("--learning-rate 3e-5", command)
        self.assertIn("--backbone-learning-rate-scale 0.2", command)
        self.assertIn("--unfreeze-last-n-layers 2", command)
        self.assertIn("--early-stopping-metric anchor_tier_accuracy", command)
        self.assertIn("--postprocess-registry output_models/logs/dinov3_vit7b16_postprocess_registry.json", command)
        self.assertIn("--postprocess-report output_models/logs/dinov3_vit7b16_full_candidate.json", command)
        self.assertIn("--postprocess-best-checkpoint output_models/checkpoints/dinov3_vit7b16/smoke/best_model.pt", command)
        self.assertIn("--postprocess-best-report output_models/logs/dinov3_vit7b16_smoke.json", command)
        self.assertIn("dinov3_vit7b16_postprocess_registry.json", command)
        self.assertIn("dinov3_vit7b16_full.json", command)


if __name__ == "__main__":
    unittest.main()
