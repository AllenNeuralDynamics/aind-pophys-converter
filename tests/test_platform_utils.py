"""Tests for aind_pophys_converter.utils.platform_utils."""

import json
import tempfile
import unittest
from pathlib import Path

from aind_pophys_converter.utils.platform_utils import (
    load_platform_plane_positions,
)

MESO_CHILD_PLATFORM = next(
    (Path(__file__).parent / "fixtures" / "meso_child" / "v1" / "pophys").glob(
        "*platform.json"
    )
)


class TestLoadPlatformPlanePositions(unittest.TestCase):
    """Tests for load_platform_plane_positions."""

    def setUp(self) -> None:
        """Create a temporary directory for platform files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.platform_fp = Path(self.temp_dir.name) / "platform.json"

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self.temp_dir.cleanup()

    def _write(self, planes: list) -> Path:
        """Write a one-group platform.json and return its path."""
        platform = {"imaging_plane_groups": [{"imaging_planes": planes}]}
        self.platform_fp.write_text(json.dumps(platform))
        return self.platform_fp

    def test_platform_lists_planes_in_group_order(self) -> None:
        """Each group's planes are returned in the order platform.json lists
        them, with the scanfield z the TIFF records for each."""
        positions = load_platform_plane_positions(MESO_CHILD_PLATFORM)
        self.assertEqual(len(positions), 8)
        self.assertEqual(positions[0], (0, -345.0, 156.0))
        self.assertEqual(positions[1], (0, -304.0, 172.0))
        self.assertEqual(positions[7], (0, -200.0, 276.0))

    def test_missing_targeted_depth_is_none(self) -> None:
        """A plane without targeted_depth reports no depth."""
        fp = self._write(
            [{"scanimage_roi_index": 1, "scanimage_scanfield_z": -10}]
        )
        self.assertEqual(load_platform_plane_positions(fp), [(1, -10.0, None)])

    def test_missing_position_raises(self) -> None:
        """A plane without its ScanImage position cannot be located."""
        for plane in (
            {"scanimage_scanfield_z": -10},
            {"scanimage_roi_index": 0},
        ):
            with self.subTest(plane=plane):
                fp = self._write([plane])
                with self.assertRaises(ValueError) as cm:
                    load_platform_plane_positions(fp)
                self.assertIn("imaging_planes[0]", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
