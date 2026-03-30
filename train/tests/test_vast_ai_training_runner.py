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

    def test_build_remote_prune_command_uses_single_retained_checkpoint(self) -> None:
        command = vast_ai_training_runner.build_remote_prune_command(
            "/workspace/mirip_v2",
            "checkpoints/dinov3_vit7b16/full/checkpoint_epoch_0009.pt",
        )

        self.assertIn("checkpoint_epoch_0009.pt", command)
        self.assertIn('find checkpoints/dinov3_vit7b16 \\( -type f -o -type l \\) -name "*.pt"', command)
        self.assertIn("best_model.pt", command)

    def test_build_launch_agent_payload_targets_sync_prune(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("", encoding="utf-8")
            with mock.patch.object(vast_ai_training_runner, "SYNC_LOG_DIR", Path(temp_dir) / "logs"):
                payload = vast_ai_training_runner.build_launch_agent_payload(str(config_path))

        self.assertEqual(payload["Label"], "com.mirip.vast-checkpoint-sync")
        self.assertIn("sync-prune", payload["ProgramArguments"])
        self.assertIn(str(config_path), payload["ProgramArguments"])
        self.assertEqual(payload["StartInterval"], 3600)

    def test_load_postprocess_registry_requires_selected_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            registry_path.write_text(json.dumps({"retained_checkpoints": []}), encoding="utf-8")

            with self.assertRaises(SystemExit):
                vast_ai_training_runner.load_postprocess_registry(registry_path)

    def test_full_stage_command_runs_postprocess_selection_before_final_eval(self) -> None:
        command = vast_ai_training_runner.build_stage_command("full", "/workspace/mirip_v2")

        self.assertIn("select_postprocess_best.py", command)
        self.assertIn("dinov3_vit7b16_postprocess_registry.json", command)
        self.assertIn("dinov3_vit7b16_full_candidate.json", command)
        self.assertIn("dinov3_vit7b16_full.json", command)


if __name__ == "__main__":
    unittest.main()
