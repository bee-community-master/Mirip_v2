from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = TRAIN_ROOT / "training" / "enrich_anchor_metadata.py"
SPEC = importlib.util.spec_from_file_location("enrich_anchor_metadata", MODULE_PATH)
assert SPEC and SPEC.loader
enrich_anchor_metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(enrich_anchor_metadata)


def _payload(
    *,
    post_no: int,
    interview_clean: str,
    university: str = "",
    department_raw: str = "",
    normalized_dept: str = "",
    anchor_group: str = "",
    tier: str = "A",
) -> dict[str, object]:
    return {
        "post_no": post_no,
        "images": [f"{post_no}.jpg"],
        "tier": tier,
        "interview_clean": interview_clean,
        "interview_raw": interview_clean,
        "university": university,
        "department_raw": department_raw,
        "normalized_dept": normalized_dept,
        "anchor_group": anchor_group,
    }


class EnrichAnchorMetadataTests(unittest.TestCase):
    def test_extract_header_fields_parses_known_examples(self) -> None:
        self.assertEqual(
            enrich_anchor_metadata.extract_header_fields("2020학년도 (정시) 국민대 영상디자인 합격!!"),
            ("국민대", "영상디자인"),
        )
        self.assertEqual(
            enrich_anchor_metadata.extract_header_fields("2018학년도 (정시) 경희대 회화과 합격!!"),
            ("경희대", "회화과"),
        )
        self.assertEqual(
            enrich_anchor_metadata.extract_header_fields("2024학년도 (정시) 상명대(천안) 디자인학부 합격!!"),
            ("상명대(천안)", "디자인학부"),
        )

    def test_build_exact_mappings_drops_ambiguous_header_department_keys(self) -> None:
        payloads = [
            _payload(
                post_no=1,
                interview_clean="2024학년도 (정시) 국민대 제품디자인 합격!!",
                university="국민대",
                department_raw="제품디자인",
                normalized_dept="industrial_design",
                anchor_group="국민대_industrial_design",
            ),
            _payload(
                post_no=2,
                interview_clean="2024학년도 (정시) 국민대 제품디자인 합격!!",
                university="국민대",
                department_raw="제품디자인",
                normalized_dept="product_design",
                anchor_group="국민대_product_design",
            ),
        ]

        mappings = enrich_anchor_metadata.build_exact_mappings(payloads)

        self.assertNotIn("제품디자인", mappings["department_raw"])
        self.assertNotIn("제품디자인", mappings["header_department"])
        self.assertIn("제품디자인", mappings["ambiguous"]["department_raw"])
        self.assertIn("제품디자인", mappings["ambiguous"]["header_department"])

    def test_enrich_metadata_applies_strict_mapping_and_reports_expected_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            metadata_dir = data_root / "metadata"
            raw_images_dir = data_root / "raw_images"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            raw_images_dir.mkdir(parents=True, exist_ok=True)

            for post_no in range(1, 16):
                (raw_images_dir / f"{post_no}.jpg").write_bytes(b"jpg")
                payload = _payload(
                    post_no=post_no,
                    interview_clean="2020학년도 (정시) 국민대 영상디자인 합격!!",
                    university="국민대",
                    department_raw="영상디자인",
                    normalized_dept="visual_design",
                    anchor_group="국민대_visual_design",
                )
                (metadata_dir / f"{post_no}.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )

            (raw_images_dir / "100.jpg").write_bytes(b"jpg")
            kyunghee_source = _payload(
                post_no=100,
                interview_clean="2018학년도 (정시) 경희대 회화과 합격!!",
                university="경희대",
                department_raw="회화과",
                normalized_dept="fine_art",
                anchor_group="경희대_fine_art",
            )
            (metadata_dir / "100.json").write_text(json.dumps(kyunghee_source, ensure_ascii=False), encoding="utf-8")

            (raw_images_dir / "2167.jpg").write_bytes(b"jpg")
            candidate_2167 = _payload(
                post_no=2167,
                interview_clean="2020학년도 (정시) 국민대 영상디자인 합격!!",
                university="국민대",
                normalized_dept="other",
            )
            (metadata_dir / "2167.json").write_text(json.dumps(candidate_2167, ensure_ascii=False), encoding="utf-8")

            (raw_images_dir / "221.jpg").write_bytes(b"jpg")
            candidate_221 = _payload(
                post_no=221,
                interview_clean="2018학년도 (정시) 경희대 회화과 합격!!",
            )
            (metadata_dir / "221.json").write_text(json.dumps(candidate_221, ensure_ascii=False), encoding="utf-8")

            (raw_images_dir / "5000.jpg").write_bytes(b"jpg")
            unmatched = _payload(
                post_no=5000,
                interview_clean="2024학년도 (정시) SADI 산업디자인 합격!!",
            )
            (metadata_dir / "5000.json").write_text(json.dumps(unmatched, ensure_ascii=False), encoding="utf-8")

            report_path = Path(temp_dir) / "anchor_group_enrichment_report.json"
            report = enrich_anchor_metadata.enrich_metadata(
                metadata_dir=metadata_dir,
                min_group_size=15,
                apply=True,
                report_path=report_path,
            )

            payload_2167 = json.loads((metadata_dir / "2167.json").read_text(encoding="utf-8"))
            self.assertEqual(payload_2167["department_raw"], "영상디자인")
            self.assertEqual(payload_2167["normalized_dept"], "visual_design")
            self.assertEqual(payload_2167["anchor_group"], "국민대_visual_design")

            payload_221 = json.loads((metadata_dir / "221.json").read_text(encoding="utf-8"))
            self.assertEqual(payload_221["university"], "경희대")
            self.assertEqual(payload_221["department_raw"], "회화과")
            self.assertEqual(payload_221["normalized_dept"], "fine_art")
            self.assertEqual(payload_221["anchor_group"], "경희대_fine_art")

            payload_5000 = json.loads((metadata_dir / "5000.json").read_text(encoding="utf-8"))
            self.assertEqual(payload_5000["department_raw"], "산업디자인")
            self.assertEqual(payload_5000["anchor_group"], "")

            self.assertEqual(report["totals"]["modified_files"], 3)
            self.assertEqual(report["eligibility"]["expected_eligible_delta"], 1)
            self.assertEqual(report["field_changes"]["anchor_group"], 2)
            self.assertEqual(report["field_changes"]["normalized_dept"], 2)
            self.assertEqual(report["field_changes"]["department_raw"], 3)
            self.assertEqual(report["field_changes"]["university"], 1)
            sadi_change = next(change for change in report["changed_files"] if change["file"] == "5000.json")
            self.assertIn("unmatched_header_university", sadi_change["unmatched_reasons"])
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
