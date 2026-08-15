import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_experiment_registry import validate_registry  # noqa: E402


class ExperimentRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(
            (ROOT / "experiments" / "registry.json").read_text(encoding="utf-8")
        )

    def test_repository_registry_is_valid(self):
        self.assertEqual(validate_registry(self.payload, ROOT), [])

    def test_duplicate_experiment_id_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["entries"].append(copy.deepcopy(payload["entries"][0]))
        errors = validate_registry(payload, ROOT)
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_unknown_run_condition_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["entries"][0]["runs"][0]["condition"] = "UNKNOWN"
        errors = validate_registry(payload, ROOT)
        self.assertTrue(any("unknown condition" in error for error in errors))

    def test_completed_experiment_requires_runs(self):
        payload = copy.deepcopy(self.payload)
        payload["entries"][0]["runs"] = []
        errors = validate_registry(payload, ROOT)
        self.assertTrue(any("needs run provenance" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
