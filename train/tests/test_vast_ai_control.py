from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vast_ai_control.py"
SPEC = importlib.util.spec_from_file_location("vast_ai_control", SCRIPT_PATH)
assert SPEC and SPEC.loader
vast_ai_control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vast_ai_control)


class VastAiControlTests(unittest.TestCase):
    def test_import_toml_module_falls_back_to_pip_vendor(self) -> None:
        sentinel = object()

        def fake_import(name: str):
            if name in {"tomllib", "tomli"}:
                raise ModuleNotFoundError(name)
            if name == "pip._vendor.tomli":
                return sentinel
            raise AssertionError(f"unexpected import: {name}")

        with mock.patch.object(vast_ai_control.importlib, "import_module", side_effect=fake_import):
            module = vast_ai_control._import_toml_module()

        self.assertIs(module, sentinel)

    def test_load_toml_reads_basic_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("[workspace]\nremote_root = \"/workspace/mirip_v2\"\n", encoding="utf-8")

            payload = vast_ai_control.load_toml(config_path)

        self.assertEqual(payload["workspace"]["remote_root"], "/workspace/mirip_v2")

