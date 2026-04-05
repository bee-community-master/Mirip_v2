from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
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
    def test_re_evaluate_baseline_bootstraps_venv_when_missing(self) -> None:
        command = vast_ai_training_runner.build_stage_command("re-evaluate-baseline", "/workspace/mirip_v2")

        bootstrap_fragment = "if [ ! -x /workspace/mirip_v2/.venv/bin/python ]; then"
        create_venv_fragment = "python3 -m venv --system-site-packages .venv"
        reevaluate_fragment = "reevaluate_checkpoint.py --checkpoint output_models/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0010.pt"

        self.assertIn(bootstrap_fragment, command)
        self.assertIn(create_venv_fragment, command)
        self.assertIn(reevaluate_fragment, command)
        self.assertLess(command.index(bootstrap_fragment), command.index(reevaluate_fragment))

    def test_build_pairs_legacy_aligned_command_runs_enrichment_before_prepare_snapshot(self) -> None:
        command = vast_ai_training_runner.build_stage_command("build-pairs-legacy-aligned", "/workspace/mirip_v2")

        enrichment_fragment = "enrich_anchor_metadata.py --metadata-dir data/metadata --report output_models/logs/anchor_group_enrichment_report.json --min-group-size 15 --apply"
        prepare_fragment = "prepare_snapshot.py --metadata-dir data/metadata --image-root data --output-manifest training/data/snapshot_manifest.csv --report output_models/logs/snapshot_report.json --min-group-size 15"

        self.assertIn(enrichment_fragment, command)
        self.assertIn(prepare_fragment, command)
        self.assertLess(command.index(enrichment_fragment), command.index(prepare_fragment))

    def test_unfreeze_ablation_command_passes_winner_values_as_python_args(self) -> None:
        with (
            mock.patch.object(vast_ai_training_runner, "_batch_probe_parts", return_value=["echo batch-probe"]),
            mock.patch.object(vast_ai_training_runner, "_build_variant_keep_best_only_parts", return_value=["echo keep-best"]),
            mock.patch.object(
                vast_ai_training_runner,
                "_json_value_command",
                side_effect=[
                    "0.4920634920634921",
                    "output_models/checkpoints/dinov3_vit7b16/ablation/F2/checkpoint_epoch_0006.pt",
                    "linear",
                    "256",
                    "0.0003",
                    "0.0001",
                    "F2",
                ],
            ),
        ):
            command = vast_ai_training_runner.build_stage_command("unfreeze-ablation", "/workspace/mirip_v2")

        self.assertIn('python3 - "$FROZEN_WINNER_METRIC"', command)
        self.assertIn('python3 - "$FROZEN_WINNER_LR"', command)
        self.assertNotIn("os.environ['FROZEN_WINNER_METRIC']", command)
        self.assertNotIn('os.environ["FROZEN_WINNER_LR"]', command)

    def test_resolve_stage_progress_args_expands_initializer_variables(self) -> None:
        checkpoint_dir = Path(tempfile.mkdtemp(prefix="vast-runner-empty-checkpoints-"))
        script = "\n".join(
            [
                "set -euo pipefail",
                *vast_ai_training_runner._oom_retry_shell_parts(),
                'FROZEN_WINNER_CHECKPOINT="output_models/checkpoints/dinov3_vit7b16/ablation/F2/checkpoint_epoch_0006.pt"',
                "resolve_stage_progress_args "
                + " ".join(
                    [
                        shlex.quote("output_models/checkpoints/dinov3_vit7b16/ablation/U1"),
                        shlex.quote(str(checkpoint_dir)),
                        shlex.quote("--initialize-from $FROZEN_WINNER_CHECKPOINT"),
                    ]
                ),
            ]
        )

        completed = subprocess.run(
            ["bash", "-lc", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.stdout.strip(),
            "--initialize-from output_models/checkpoints/dinov3_vit7b16/ablation/F2/checkpoint_epoch_0006.pt",
        )

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
        self.assertIn("output_models/checkpoints/dinov3_vit7b16/full", command)
        self.assertIn("checkpoints/dinov3_vit7b16/full", command)
        self.assertNotIn("output_models/checkpoints/dinov3_vit7b16/ablation", command)
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
            with mock.patch.object(vast_ai_training_runner, "LAUNCH_AGENT_LOG_DIR", Path(temp_dir) / "launchd_logs"):
                payload = vast_ai_training_runner.build_launch_agent_payload(str(config_path))

        self.assertEqual(payload["Label"], "com.mirip.vast-checkpoint-sync")
        self.assertIn("sync-prune", payload["ProgramArguments"])
        self.assertIn(str(config_path), payload["ProgramArguments"])
        self.assertEqual(payload["StartInterval"], 900)
        self.assertIn("launchd_logs", payload["StandardOutPath"])
        self.assertIn("launchd_logs", payload["StandardErrorPath"])

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
        self.assertEqual(len(recorded_commands), 2)
        self.assertFalse(best_link.exists())
        self.assertIn("--partial", recorded_commands[1])
        self.assertIn("--size-only", recorded_commands[1])
        self.assertIn(
            "root@ssh1.vast.ai:/workspace/mirip_v2/train/checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0002.pt",
            recorded_commands[1],
        )
        self.assertIn(
            str(train_root / "output_models" / "checkpoints" / "dinov3_vit7b16" / "full" / "checkpoint_epoch_0002.pt"),
            recorded_commands[1],
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

    def test_sync_prune_prunes_remotely_without_downloading_checkpoints(self) -> None:
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
        pull_sync_prune_artifacts.assert_called_once()
        first_call = pull_sync_prune_artifacts.call_args_list[0]
        self.assertEqual(first_call.kwargs["host"], "ssh1.vast.ai")
        self.assertIsNone(first_call.kwargs.get("retained_checkpoints"))
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

    def test_build_pairs_stage_command_uses_legacy_aligned_targets(self) -> None:
        command = vast_ai_training_runner.build_stage_command("build-pairs-legacy-aligned", "/workspace/mirip_v2")

        self.assertIn("--train-ratio 0.8", command)
        self.assertIn("--val-ratio 0.1", command)
        self.assertIn("--train-pairs-target 40000", command)
        self.assertIn("--val-pairs-target 5000", command)
        self.assertIn("--max-appearances 48", command)
        self.assertIn("--distance1-ratio 0.6", command)
        self.assertIn("--train-tier-pair-min-a-s 4000", command)
        self.assertIn("--train-tier-pair-cap-a-b 18000", command)
        self.assertIn("--allow-shortfall", command)
        self.assertIn("prepare_snapshot.py", command)
        self.assertIn("build_pairs.py", command)

    def test_re_evaluate_baseline_stage_generates_epoch10_report(self) -> None:
        command = vast_ai_training_runner.build_stage_command("re-evaluate-baseline", "/workspace/mirip_v2")

        self.assertIn("reevaluate_checkpoint.py", command)
        self.assertIn("checkpoint_epoch_0010.pt", command)
        self.assertIn("epoch10_robust_baseline.json", command)
        self.assertIn("[ ! -f train/training/data/pairs_val.csv ]", command)
        self.assertIn("[ ! -f train/training/data/metadata_train.csv ]", command)
        self.assertIn("[ ! -f train/training/data/metadata_val.csv ]", command)
        self.assertIn("build_pairs.py", command)

    def test_frozen_ablation_stage_runs_probe_and_four_variants(self) -> None:
        command = vast_ai_training_runner.build_stage_command("frozen-ablation", "/workspace/mirip_v2")

        self.assertIn("probe_dinov3_batch_size.py", command)
        self.assertIn("run_training_with_oom_retry", command)
        self.assertIn('run_training_with_oom_retry "frozen_F1"', command)
        self.assertIn("--batch-size-candidates 8,6,4,2", command)
        self.assertIn("ablation/F1", command)
        self.assertIn("ablation/F2", command)
        self.assertIn("ablation/F3", command)
        self.assertIn("ablation/F4", command)
        self.assertIn("--head-type linear", command)
        self.assertIn("--head-type mlp_small", command)
        self.assertIn("--epochs 6", command)
        self.assertIn("--warmup-epochs 1", command)
        self.assertIn("--freeze-backbone", command)
        self.assertIn("--anchor-eval-n-per-tier 24", command)
        self.assertIn('find train/output_models/checkpoints/dinov3_vit7b16/ablation/F1 -maxdepth 1 -type f -name "checkpoint_epoch_*.pt"', command)
        self.assertIn('ln -sfn "$(basename "$SELECTED_VARIANT_CHECKPOINT")" train/output_models/checkpoints/dinov3_vit7b16/ablation/F4/best_model.pt', command)

    def test_select_ablation_winner_stage_reads_frozen_variant_registries(self) -> None:
        command = vast_ai_training_runner.build_stage_command("select-ablation-winner", "/workspace/mirip_v2")

        self.assertIn("select_ablation_winner.py", command)
        self.assertIn("--candidate F1=output_models/logs/dinov3_vit7b16_ablation_F1_registry.json", command)
        self.assertIn("--candidate F4=output_models/logs/dinov3_vit7b16_ablation_F4_registry.json", command)
        self.assertIn("dinov3_vit7b16_frozen_ablation_summary.json", command)
        self.assertIn("--min-improvement 0.005", command)

    def test_unfreeze_ablation_stage_uses_frozen_winner_as_initializer(self) -> None:
        command = vast_ai_training_runner.build_stage_command("unfreeze-ablation", "/workspace/mirip_v2")

        self.assertIn("frozen_ablation_summary.json", command)
        self.assertIn("select_ablation_winner.py", command)
        self.assertIn("run_training_with_oom_retry", command)
        self.assertIn('export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"', command)
        self.assertIn('run_training_with_oom_retry "unfreeze_U1"', command)
        self.assertIn('FROZEN_WINNER_NAME="$(', command)
        self.assertIn('if [ "$FROZEN_WINNER_NAME" != "F1" ]; then rm -rf train/output_models/checkpoints/dinov3_vit7b16/ablation/F1; fi', command)
        self.assertIn('if [ "$FROZEN_WINNER_NAME" != "F4" ]; then rm -rf train/output_models/checkpoints/dinov3_vit7b16/ablation/F4; fi', command)
        self.assertIn("--initialize-from $FROZEN_WINNER_CHECKPOINT", command)
        self.assertIn("--resume-from %s --resume-next-epoch", command)
        self.assertIn("--no-freeze-backbone", command)
        self.assertIn("--unfreeze-last-n-layers 1", command)
        self.assertIn("--unfreeze-last-n-layers 2", command)
        self.assertIn("ablation/U1", command)
        self.assertIn("ablation/U2", command)
        self.assertIn('find train/output_models/checkpoints/dinov3_vit7b16/ablation/U1 -maxdepth 1 -type f -name "checkpoint_epoch_*.pt"', command)

    def test_select_overall_winner_stage_reads_frozen_and_unfreeze_summaries(self) -> None:
        command = vast_ai_training_runner.build_stage_command("select-overall-winner", "/workspace/mirip_v2")

        self.assertIn("select_overall_winner.py", command)
        self.assertIn("--summary frozen=output_models/logs/dinov3_vit7b16_frozen_ablation_summary.json", command)
        self.assertIn("--summary unfreeze=output_models/logs/dinov3_vit7b16_unfreeze_ablation_summary.json", command)

    def test_full_fresh_stage_uses_overall_winner_without_resume(self) -> None:
        command = vast_ai_training_runner.build_stage_command("full-fresh", "/workspace/mirip_v2")

        self.assertIn("winner_config", command)
        self.assertIn("probe_dinov3_batch_size.py", command)
        self.assertIn("run_training_with_oom_retry", command)
        self.assertIn('run_training_with_oom_retry "full_fresh"', command)
        self.assertIn("output_models/archive", command)
        self.assertIn("overall_winner.json", command)
        self.assertIn("--epochs 24", command)
        self.assertIn("--warmup-epochs 2", command)
        self.assertIn("--restart-from-best-patience 3", command)
        self.assertIn("--feature-pool cls_mean_patch_concat", command)
        self.assertIn("$FREEZE_FLAG", command)
        self.assertIn("$PROGRESS_ARGS", command)
        self.assertIn("--postprocess-registry output_models/logs/dinov3_vit7b16_postprocess_registry.json", command)
        self.assertIn("dinov3_vit7b16_full.json", command)


if __name__ == "__main__":
    unittest.main()
