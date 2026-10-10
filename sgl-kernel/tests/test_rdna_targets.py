#!/usr/bin/env python3
"""Arch policy for the ROCm sgl-kernel build. No torch and no GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdna_targets import (  # noqa: E402
    RdnaArchError,
    rocm_compile_profile,
    validate_amdgpu_targets,
)


class RdnaTargetsTest(unittest.TestCase):
    def test_single_rdna_archs(self):
        self.assertEqual(validate_amdgpu_targets("gfx1030"), ["gfx1030"])
        self.assertEqual(validate_amdgpu_targets("gfx1100:sramecc+:xnack-"), ["gfx1100"])

    def test_cdna_pair_still_allowed(self):
        self.assertEqual(
            validate_amdgpu_targets("gfx942;gfx950"),
            ["gfx942", "gfx950"],
        )

    def test_refuses_mixed_rdna(self):
        with self.assertRaises(RdnaArchError) as ctx:
            validate_amdgpu_targets("gfx1030;gfx1100")
        self.assertIn("WMMA", str(ctx.exception))

    def test_refuses_rdna_with_cdna(self):
        with self.assertRaises(RdnaArchError):
            validate_amdgpu_targets("gfx942;gfx1030")

    def test_unknown_arch_keeps_warning_text(self):
        with self.assertRaises(RdnaArchError) as ctx:
            validate_amdgpu_targets("gfx908")
        message = str(ctx.exception)
        self.assertIn("Warning: Unsupported GPU architecture detected 'gfx908'.", message)
        self.assertIn("gfx942", message)
        self.assertIn("gfx1030", message)

    def test_profiles(self):
        gfx1030 = rocm_compile_profile("gfx1030")
        gfx1100 = rocm_compile_profile("gfx1100")
        gfx942 = rocm_compile_profile("gfx942")
        gfx950 = rocm_compile_profile("gfx950")
        for rdna in (gfx1030, gfx1100):
            self.assertTrue(rdna["is_rdna"])
            self.assertTrue(rdna["wave32"])
            self.assertFalse(rdna["enable_fp8"])
            self.assertIsNone(rdna["fp8_macro"])
            self.assertFalse(rdna["custom_allreduce"])
            self.assertEqual(rdna["topk_dynamic_smem_bytes"], 48 * 1024)
        self.assertFalse(gfx942["is_rdna"])
        self.assertTrue(gfx942["enable_fp8"])
        self.assertEqual(gfx942["fp8_macro"], "-DHIP_FP8_TYPE_FNUZ")
        self.assertEqual(gfx942["topk_dynamic_smem_bytes"], 48 * 1024)
        self.assertTrue(gfx942["custom_allreduce"])
        self.assertFalse(gfx942["wave32"])
        self.assertEqual(gfx950["fp8_macro"], "-DHIP_FP8_TYPE_E4M3")
        self.assertEqual(gfx950["topk_dynamic_smem_bytes"], 32 * 1024 * 4)


if __name__ == "__main__":
    unittest.main()
