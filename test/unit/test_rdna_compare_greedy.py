"""Token-id compare used by the RDNA hardware smoke. No GPU."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load():
    path = Path(__file__).resolve().parents[2] / "scripts" / "rdna" / "compare_greedy.py"
    spec = importlib.util.spec_from_file_location("compare_greedy", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class CompareGreedyTest(unittest.TestCase):
    def test_match_and_first_divergence(self):
        mod = _load()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "a.json"
            right = root / "b.json"
            left.write_text(json.dumps({"output_ids": [1, 2, 3]}))
            right.write_text(json.dumps([{"meta_info": {"output_ids": [1, 2, 3]}}]))
            self.assertEqual(mod.ids_of(mod.load(left)), [1, 2, 3])
            self.assertEqual(mod.ids_of(mod.load(right)), [1, 2, 3])
            right.write_text(json.dumps({"output_ids": [1, 9, 3]}))
            self.assertEqual(mod.ids_of(mod.load(left))[1], 2)
            self.assertNotEqual(mod.ids_of(mod.load(left)), mod.ids_of(mod.load(right)))


if __name__ == "__main__":
    unittest.main()
