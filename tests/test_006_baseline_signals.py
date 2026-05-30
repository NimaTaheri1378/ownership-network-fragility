from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import unittest


class TestStep006BaselineSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ.get("ONF_PROJECT_ROOT", Path.cwd())).resolve()
        path = cls.root / "scripts" / "006_baseline_signal_tests.py"
        spec = importlib.util.spec_from_file_location("step006", path)
        assert spec and spec.loader
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_expected_features_have_no_duplicates(self) -> None:
        features = self.mod.CORE_FEATURES + self.mod.CONTROL_CONTEXT_FEATURES
        self.assertEqual(len(features), len(set(features)))
        self.assertIn("network_peer_sell_pressure", features)
        self.assertIn("stock_sell_pressure", features)

    def test_directional_features_are_subset(self) -> None:
        features = set(self.mod.CORE_FEATURES + self.mod.CONTROL_CONTEXT_FEATURES)
        self.assertTrue(set(self.mod.NEGATIVE_HIGH_MINUS_LOW_EXPECTED).issubset(features))

    def test_safe_sql_identifier_rejects_bad_names(self) -> None:
        self.assertEqual(self.mod.qident("owner_count"), '"owner_count"')
        with self.assertRaises(ValueError):
            self.mod.qident("owner_count; DROP TABLE x")


if __name__ == "__main__":
    unittest.main()
