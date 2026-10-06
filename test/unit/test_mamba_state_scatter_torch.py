"""CPU check that the gfx1030 mamba state copy stays on the destination device.

Loads the module by path so the test does not execute sglang's package init.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "mamba"
    / "mamba_state_scatter_torch.py"
)
_spec = importlib.util.spec_from_file_location(
    "mamba_state_scatter_torch", _MODULE_PATH
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
torch_mamba_state_scatter_with_mask = _mod.torch_mamba_state_scatter_with_mask


class TorchMambaStateScatterTest(unittest.TestCase):
    def test_copies_valid_steps_and_leaves_invalid(self):
        layers, cache, spec, draft, state = 2, 6, 3, 4, 5
        dst = torch.zeros(layers, cache, state)
        src = torch.arange(layers * spec * draft * state, dtype=torch.float32).reshape(
            layers, spec, draft, state
        )
        dst_indices = torch.tensor([1, 4, 2])
        step_indices = torch.tensor([0, -1, 3])
        torch_mamba_state_scatter_with_mask(dst, src, dst_indices, step_indices)
        self.assertTrue(torch.equal(dst[:, 1], src[:, 0, 0]))
        self.assertTrue(torch.equal(dst[:, 2], src[:, 2, 3]))
        self.assertTrue(torch.equal(dst[:, 4], torch.zeros(layers, state)))
        self.assertEqual(dst.device, torch.device("cpu"))

    def test_empty_and_all_masked(self):
        dst = torch.ones(1, 2, 3)
        src = torch.zeros(1, 1, 2, 3)
        torch_mamba_state_scatter_with_mask(
            dst,
            src,
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        )
        self.assertTrue(torch.equal(dst, torch.ones(1, 2, 3)))
        torch_mamba_state_scatter_with_mask(
            dst, src, torch.tensor([0]), torch.tensor([-1])
        )
        self.assertTrue(torch.equal(dst, torch.ones(1, 2, 3)))


if __name__ == "__main__":
    unittest.main()
