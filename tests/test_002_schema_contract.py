from __future__ import annotations

import json
import os
from pathlib import Path
import unittest


class TestStep002SchemaContract(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(os.environ.get("ONF_PROJECT_ROOT", Path.cwd())).resolve()

    def _latest_contract(self) -> Path:
        candidates = sorted(
            (self.root / "artifacts" / "schema").glob("002_schema_contract_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

        fixture = self.root / "tests" / "fixtures" / "schema_contract_public.json"
        self.assertTrue(
            fixture.exists(),
            "No local Step 002 schema contract found and public fixture is missing.",
        )
        return fixture

    def _load_contract(self) -> dict:
        return json.loads(self._latest_contract().read_text(encoding="utf-8"))

    def test_contract_is_metadata_only(self) -> None:
        contract = self._load_contract()
        self.assertTrue(contract.get("metadata_only") or contract.get("data_policy", {}).get("no_raw_vendor_rows"))
        text = json.dumps(contract).lower()
        forbidden = ["head_records", "records_preview", "sample_rows", "raw_vendor_rows"]
        self.assertFalse(any(token in text for token in forbidden))

    def test_core_roles_have_selected_tables(self) -> None:
        contract = self._load_contract()
        roles = contract.get("source_roles") or contract.get("roles") or contract.get("selected_sources") or {}
        required = {"13f_holdings", "crsp_monthly_stock", "crsp_stock_names"}
        self.assertTrue(required.issubset(set(roles)), f"missing roles: {required - set(roles)}")

        for role in required:
            record = roles[role]
            self.assertTrue(record.get("library"), role)
            self.assertTrue(record.get("table"), role)


if __name__ == "__main__":
    unittest.main()
