"""Known pixel geometry and real two-protocol comparison regressions."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage4_assembly"))
from assemble import Fragment, resolve_config  # noqa: E402
from compare_methods import compare, normalize_fragments, render_method  # noqa: E402


class ComparisonTests(unittest.TestCase):
    def fragment(self, rgba, transform=None, target=None):
        height, width = rgba.shape[:2]
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
        return Fragment("A", {"image_corners_px": corners.tolist()}, rgba,
                        np.eye(3) if transform is None else transform,
                        corners if target is None else target)

    def test_global_scale_keeps_texture_and_target_aligned(self):
        y, x = np.mgrid[:6, :8]
        reference = np.stack([20 + x * 8, 30 + y * 9, 80 + x + y], axis=-1).astype(np.uint8)
        rgba = np.dstack([reference, np.full((6, 8), 255, np.uint8)])
        double = np.diag([2., 2., 1.])
        target = np.array([[0, 0], [16, 0], [16, 12], [0, 12]], float)
        fragments, _ = normalize_fragments([self.fragment(rgba, double, target)],
                                            (16., 12.), (8, 6), 0)
        raw, _, mask, metrics = render_method(fragments, reference,
                                             resolve_config({"color_correction": False}))
        np.testing.assert_array_equal(raw, reference)
        self.assertTrue(mask.all())
        self.assertTrue(metrics["raw"]["identical"])

    def test_opaque_black_pixels_still_count_as_coverage(self):
        reference = np.zeros((8, 8, 3), np.uint8)
        rgba = np.dstack([reference, np.full((8, 8), 255, np.uint8)])
        _, _, mask, metrics = render_method([self.fragment(rgba)], reference, resolve_config(None))
        self.assertTrue(mask.all())
        self.assertEqual(metrics["raster"]["covered_fraction"], 1)

    def test_known_target_is_not_warped_measured_geometry(self):
        rgba = np.full((8, 8, 4), 255, np.uint8)
        target = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], float)
        f = self.fragment(rgba, np.diag([0.5, 1., 1.]), target)
        _, _, _, metrics = render_method([f], rgba[:, :, :3], resolve_config(None))
        self.assertAlmostEqual(metrics["target_geometry"]["coverage_fraction"], 1.)
        self.assertAlmostEqual(metrics["warped_measured_geometry"]["coverage_fraction"], 0.5)
        self.assertLess(metrics["raster"]["covered_fraction"], 1.)

    def test_real_geometry_and_sift_handoffs(self):
        with tempfile.TemporaryDirectory() as directory:
            report = compare(output=Path(directory) / "comparison", write_native=False)
            self.assertEqual(report["evaluation_size_px"], [1280, 720])
            for name in ("geometry", "sift"):
                method = report["methods"][name]
                self.assertEqual(method["piece_count"], 16)
                self.assertTrue(np.isfinite(method["raw"]["psnr_db"]))
                self.assertTrue(np.isfinite(method["corrected"]["ssim_gray"]))
            self.assertEqual(report["methods"]["sift"]["matching_review"], ["S05"])
            self.assertEqual(report["methods"]["sift"]["target_geometry_source"], "known_cut_layout")
            self.assertTrue((Path(directory) / "comparison/comparison_report.json").exists())


if __name__ == "__main__":
    unittest.main()
