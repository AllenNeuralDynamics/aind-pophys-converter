"""Tests for job module."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image


class TestAddBorderAndLabel(unittest.TestCase):
    """Tests for add_border_and_label."""

    def setUp(self):
        """Import here to avoid top-level side effects."""
        from aind_pophys_converter.utils.image_utils import (
            add_border_and_label,
        )

        self.fn = add_border_and_label

    def test_output_is_image(self):
        """Returns a PIL Image."""
        img = Image.new("L", (100, 100))
        result = self.fn(img, "Test")
        self.assertIsInstance(result, Image.Image)

    def test_output_wider_than_input(self):
        """Border increases width."""
        img = Image.new("L", (100, 100))
        result = self.fn(img, "Test", border=5)
        self.assertGreater(result.width, 100)

    def test_output_taller_than_input(self):
        """Label area and border increase height."""
        img = Image.new("L", (100, 100))
        result = self.fn(img, "Test", border=5)
        self.assertGreater(result.height, 100)

    def test_output_is_rgb(self):
        """Output is always RGB."""
        img = Image.new("L", (50, 50))
        result = self.fn(img, "Label")
        self.assertEqual(result.mode, "RGB")


class TestIsChildSessionViaPlatformJson(unittest.TestCase):
    """Tests for is_child_session_via_platform_json."""

    def setUp(self):
        """Import and set up temp dir."""
        from aind_pophys_converter.utils.platform_utils import (
            is_child_session_via_platform_json,
        )

        self.fn = is_child_session_via_platform_json
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up."""
        self.tmpdir.cleanup()

    def _write_platform(self, data):
        """Write a platform.json file and return its path."""
        fp = self.tmp_path / "platform.json"
        with open(fp, "w") as f:
            json.dump(data, f)
        return fp

    def test_returns_true_when_parent_session_present(self):
        """Returns True when parent_session key exists."""
        fp = self._write_platform({"parent_session": "some_session_id"})
        self.assertTrue(self.fn(fp))

    def test_returns_false_when_parent_session_absent(self):
        """Returns False when parent_session key is missing."""
        fp = self._write_platform({"imaging_plane_groups": []})
        self.assertFalse(self.fn(fp))

    def test_returns_false_when_parent_session_is_null(self):
        """Returns False when parent_session is None."""
        fp = self._write_platform({"parent_session": None})
        self.assertFalse(self.fn(fp))


class TestCreateVasculature(unittest.TestCase):
    """Tests for create_vasculature."""

    def setUp(self):
        """Set up temp directories."""
        from aind_pophys_converter.qc import create_vasculature

        self.fn = create_vasculature
        self.tmpdir = tempfile.TemporaryDirectory()
        self.pophys_dir = Path(self.tmpdir.name) / "pophys"
        self.output_dir = Path(self.tmpdir.name) / "output"
        self.pophys_dir.mkdir()
        self.output_dir.mkdir()

    def tearDown(self):
        """Clean up."""
        self.tmpdir.cleanup()

    def test_skips_when_no_vasculature_tif(self):
        """Does not create output directory when no vasculature TIF."""
        self.fn(self.pophys_dir, self.output_dir)
        self.assertFalse((self.output_dir / "vasculature").exists())

    def test_creates_png_and_metric(self):
        """Creates vasculature.png and vasculature_metric.json."""
        # Write a minimal grayscale TIFF as the vasculature source
        src = self.pophys_dir / "session_vasculature.tif"
        img = Image.fromarray(
            np.zeros((64, 64), dtype=np.uint8), mode="L"
        )
        img.save(src)

        self.fn(self.pophys_dir, self.output_dir)

        vas_dir = self.output_dir / "vasculature"
        self.assertTrue((vas_dir / "vasculature.png").exists())
        self.assertTrue((vas_dir / "vasculature_metric.json").exists())


class TestPairDepthTifsWithAvgDepthPngs(unittest.TestCase):
    """Tests for pair_depth_tifs_with_avg_depth_pngs."""

    def setUp(self):
        """Set up temp directories and import."""
        from aind_pophys_converter.qc import (
            pair_depth_tifs_with_avg_depth_pngs,
        )

        self.fn = pair_depth_tifs_with_avg_depth_pngs
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.pophys_dir = self.tmp_path / "pophys"
        self.avg_dir = self.tmp_path / "avg"
        self.output_dir = self.tmp_path / "output"
        for d in (self.pophys_dir, self.avg_dir, self.output_dir):
            d.mkdir()

    def tearDown(self):
        """Clean up."""
        self.tmpdir.cleanup()

    def _write_platform(self, planes):
        """Write a platform.json with imaging planes."""
        fp = self.tmp_path / "platform.json"
        with open(fp, "w") as f:
            data = {"imaging_plane_groups": [{"imaging_planes": planes}]}
            json.dump(data, f)
        return fp

    def test_skips_when_no_imaging_planes(self):
        """Does nothing when platform.json has no imaging planes."""
        fp = self._write_platform([])
        self.fn(self.pophys_dir, self.avg_dir, fp, self.output_dir)
        self.assertEqual(list(self.output_dir.glob("*.json")), [])

    def test_skips_when_no_child_tifs(self):
        """Does nothing when avg_slice_dir has no TIF files."""
        planes = [
            {
                "intended_depth": 200,
                "targeted_structure_id": "VISp",
                "scanimage_scanfield_z": -276.0,
            }
        ]
        fp = self._write_platform(planes)
        self.fn(self.pophys_dir, self.avg_dir, fp, self.output_dir)
        self.assertEqual(list(self.output_dir.glob("*.json")), [])

    def test_writes_metric_and_png(self):
        """Writes merged PNG and fov metric JSON for a matched plane."""
        planes = [
            {
                "intended_depth": 200,
                "targeted_structure_id": "VISp",
                "scanimage_scanfield_z": -276.0,
            }
        ]
        fp = self._write_platform(planes)

        # Parent depth TIF in pophys_dir
        parent_arr = np.random.randint(
            600, 1400, (64, 64), dtype=np.uint16
        )
        import tifffile

        tifffile.imwrite(
            self.pophys_dir / "ts_200_VISp_depth.tif", parent_arr
        )

        # Child averaged TIF in avg_dir named by abs(z)
        child_arr = np.random.uniform(4, 32, (64, 64)).astype(np.float32)
        tifffile.imwrite(self.avg_dir / "276.0.tif", child_arr)

        self.fn(self.pophys_dir, self.avg_dir, fp, self.output_dir)

        # Metric written
        metric_fp = self.output_dir / "VISp_200_fov_metric.json"
        self.assertTrue(metric_fp.exists())
        # Merged PNG written
        pngs = list(self.output_dir.glob("*_merged.png"))
        self.assertEqual(len(pngs), 1)
        # Child TIF deleted
        self.assertFalse((self.avg_dir / "276.0.tif").exists())


class TestWriteAvgDepthSlices(unittest.TestCase):
    """Tests for write_avg_depth_slices."""

    def setUp(self):
        """Set up temp dir."""
        from aind_pophys_converter.qc import write_avg_depth_slices

        self.fn = write_avg_depth_slices
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name) / "output"

    def tearDown(self):
        """Clean up."""
        self.tmpdir.cleanup()

    def test_creates_tif_per_plane(self):
        """Creates one TIF file per ROI/z entry in the manifest."""
        import tifffile

        mock_splitter = MagicMock()
        mock_splitter.roi_z_int_manifest = [(0, 0), (1, 1)]
        mock_splitter._z_from_int.side_effect = lambda i: float(i * -100)

        fake_arr = np.zeros((64, 64), dtype=np.float32)

        def fake_write(i_roi, z_value, output_path):
            """Stub that writes a blank TIFF."""
            tifffile.imwrite(output_path, fake_arr)

        mock_splitter.write_output_file.side_effect = fake_write

        self.fn(mock_splitter, self.output_dir)

        tifs = list(self.output_dir.glob("*.tif"))
        self.assertEqual(len(tifs), 2)


if __name__ == "__main__":
    unittest.main()
