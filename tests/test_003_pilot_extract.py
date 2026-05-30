from __future__ import annotations

import json
from pathlib import Path
import os
import unittest


class TestPilotExtractManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(os.environ.get("ONF_PROJECT_ROOT", Path.cwd())).resolve()

    def latest_manifest(self) -> Path | None:
        candidates = sorted((self.root / "artifacts" / "logs").glob("003_pilot_extract_manifest_*.json"))
        candidates = [p for p in candidates if "FAILED" not in p.name]
        return candidates[-1] if candidates else None

    def test_manifest_if_present_has_required_roles(self) -> None:
        path = self.latest_manifest()
        if path is None:
            self.skipTest("Step 003 pilot manifest not created yet")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("ok", manifest.get("status"))
        by_role = {row.get("role"): row for row in manifest.get("tables", [])}
        for role in ["13f_holdings", "crsp_monthly_stock", "crsp_daily_stock", "crsp_stock_names"]:
            self.assertIn(role, by_role)
            self.assertEqual("pass", by_role[role].get("status"))
            self.assertGreater(int(by_role[role].get("n_rows") or 0), 0)

    def test_manifest_if_present_does_not_include_raw_record_previews(self) -> None:
        path = self.latest_manifest()
        if path is None:
            self.skipTest("Step 003 pilot manifest not created yet")
        text = path.read_text(encoding="utf-8").lower()
        forbidden = ["sample_records", "head_records", "record_preview", "records_preview"]
        self.assertFalse(any(token in text for token in forbidden))


if __name__ == "__main__":
    unittest.main()
