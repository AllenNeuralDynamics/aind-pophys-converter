"""Tests for the BergamoTiffStitcher class"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np

from aind_pophys_converter.bergamo_stitcher import (
    BergamoSettings,
    BergamoTiffStitcher,
)


def _make_settings(input_dir, output_dir, acquisition_fp=None):
    """Return a BergamoSettings with sensible defaults for testing."""
    if acquisition_fp is None:
        acquisition_fp = input_dir / "acquisition.json"
    return BergamoSettings(
        input_dir=input_dir,
        output_dir=output_dir,
        acquisition_fp=acquisition_fp,
    )


class TestBergamoTiffStitcher(unittest.TestCase):
    """Test cases for BergamoTiffStitcher class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.acquisition_file = self.input_dir / "acquisition.json"
        self.acquisition_file.write_text(
            json.dumps(
                {
                    "subject_id": "000000",
                    "acquisition_start_time": "2025-05-17T10:00:00-07:00",
                    "acquisition_end_time": "2025-05-17T12:00:00-07:00",
                    "instrument_id": "test_instrument",
                    "acquisition_type": "Other",
                    "data_streams": [],
                    "stimulus_epochs": [],
                }
            )
        )

        self.tiff_files = ["test_tiff_00001.tif", "test_tiff_00002.tif"]
        self.settings = _make_settings(
            self.input_dir, self.output_dir, self.acquisition_file
        )

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    @patch("aind_pophys_converter.bergamo_stitcher.get_imaging_planes")
    @patch("aind_pophys_converter.bergamo_stitcher.load_acquisition")
    @patch("aind_pophys_converter.bergamo_stitcher.ScanImageTiffReader")
    def test_run_converter_creates_hdf5(
        self, mock_reader, mock_load_acq, mock_get_planes
    ):
        """Test that run_converter creates an HDF5 file with the
        expected structure.
        """
        mock_plane = Mock()
        mock_plane.targeted_structure.acronym = "MO"
        mock_get_planes.return_value = [mock_plane]
        mock_load_acq.return_value = Mock()

        mock_instance = Mock()
        mock_reader.return_value = mock_instance
        mock_instance.__enter__ = Mock(return_value=mock_instance)
        mock_instance.__exit__ = Mock(return_value=None)
        mock_instance.data.return_value = np.zeros((5, 10, 10), dtype=np.int16)
        mock_instance.metadata.return_value = "Test metadata"
        mock_instance.shape.return_value = (5, 10, 10)

        for fname in self.tiff_files:
            (self.input_dir / fname).touch()

        stitcher = BergamoTiffStitcher(self.settings)
        output_path = stitcher.run_converter()

        self.assertTrue(os.path.exists(output_path[0]))
        with h5py.File(output_path[0], "r") as f:
            self.assertIn("data", f)
            self.assertIn("trial_locations", f)
            self.assertIn("epoch_locations", f)
            self.assertIn("epoch_filenames", f)
            self.assertIn("metadata", f)
            self.assertEqual(f["data"].shape[1:], (10, 10))

    def test_from_args_creates_stitcher(self):
        """Test that from_args creates a BergamoTiffStitcher from
        command line arguments.
        """
        args = [
            "--input-dir",
            str(self.input_dir),
            "--acquisition-file",
            str(self.acquisition_file),
        ]
        stitcher = BergamoTiffStitcher.from_args(args)
        self.assertIsInstance(stitcher, BergamoTiffStitcher)
        self.assertEqual(stitcher.input_dir, self.input_dir)

    def test_from_args_with_all_arguments(self):
        """Test from_args with all optional arguments."""
        args = [
            "--input-dir",
            str(self.input_dir),
            "--output-dir",
            str(self.output_dir),
            "--acquisition-file",
            str(self.acquisition_file),
        ]
        stitcher = BergamoTiffStitcher.from_args(args)
        self.assertEqual(stitcher.input_dir, self.input_dir)
        self.assertEqual(stitcher.output_dir, self.output_dir)

    def test_discover_tiff_epochs_groups_by_stem(self):
        """Test that tiff files are grouped by stem and sorted numerically."""
        files = [
            "spont_00003.tif",
            "spont_00001.tif",
            "spont_00002.tif",
            "spontpost_00001.tif",
        ]
        for fname in files:
            (self.input_dir / fname).touch()

        stitcher = BergamoTiffStitcher(self.settings)
        result = stitcher._discover_tiff_epochs()

        self.assertIn("spont", result)
        self.assertIn("spontpost", result)
        self.assertEqual(len(result["spont"]["tiff_files"]), 3)
        self.assertEqual(len(result["spontpost"]["tiff_files"]), 1)
        # Sorted by numeric suffix
        basenames = [
            os.path.basename(f) for f in result["spont"]["tiff_files"]
        ]
        self.assertEqual(
            basenames,
            ["spont_00001.tif", "spont_00002.tif", "spont_00003.tif"],
        )

    def test_discover_tiff_epochs_excludes_stack(self):
        """Test that files with the 'stack' stem are excluded."""
        (self.input_dir / "spont_00001.tif").touch()
        (self.input_dir / "stack_00001.tif").touch()
        (self.input_dir / "stack_00002.tif").touch()

        stitcher = BergamoTiffStitcher(self.settings)
        result = stitcher._discover_tiff_epochs()

        self.assertIn("spont", result)
        self.assertNotIn("stack", result)

    def test_discover_tiff_epochs_no_tiffs_raises(self):
        """Test that an empty directory raises ValueError."""
        stitcher = BergamoTiffStitcher(self.settings)
        with self.assertRaises(ValueError):
            stitcher._discover_tiff_epochs()

    def test_discover_tiff_epochs_only_stack_raises(self):
        """Test that only stack files (no eligible tiffs) raises ValueError."""
        (self.input_dir / "stack_00001.tif").touch()
        stitcher = BergamoTiffStitcher(self.settings)
        with self.assertRaises(ValueError):
            stitcher._discover_tiff_epochs()

    def test_discover_tiff_epochs_debug_limits_to_one_file(self):
        """Test that debug=True clips each epoch to its first tiff file."""
        for stem in ("spont", "spontpost"):
            for i in range(1, 4):
                (self.input_dir / f"{stem}_0000{i}.tif").touch()

        debug_settings = BergamoSettings(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            acquisition_fp=self.acquisition_file,
            debug=True,
        )
        stitcher = BergamoTiffStitcher(debug_settings)
        result = stitcher._discover_tiff_epochs()

        self.assertEqual(len(result["spont"]["tiff_files"]), 1)
        self.assertEqual(len(result["spontpost"]["tiff_files"]), 1)
        # First file of each epoch is kept
        self.assertIn("spont_00001.tif", result["spont"]["tiff_files"][0])
        self.assertIn(
            "spontpost_00001.tif", result["spontpost"]["tiff_files"][0]
        )


if __name__ == "__main__":
    unittest.main()
