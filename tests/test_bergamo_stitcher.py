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


def _minimal_acquisition_json(stimulus_epochs):
    """Return a minimal valid v2 Acquisition JSON string."""
    return json.dumps(
        {
            "subject_id": "000000",
            "acquisition_start_time": "2025-05-17T10:00:00-07:00",
            "acquisition_end_time": "2025-05-17T12:00:00-07:00",
            "instrument_id": "test_instrument",
            "acquisition_type": "Other",
            "data_streams": [],
            "stimulus_epochs": stimulus_epochs,
        }
    )


def _make_acquisition_json(tiff_stem, tiff_files):
    """Return a minimal v2 Acquisition JSON string for testing.

    output_parameters are serialized as JSON in epoch.notes, matching
    the auto-upgrader behaviour for the v1 custom field.
    """
    output_params = json.dumps(
        {"tiff_stem": tiff_stem, "tiff_files": tiff_files}
    )
    epochs = [
        {
            "stimulus_start_time": "2025-05-17T10:00:00-07:00",
            "stimulus_end_time": "2025-05-17T10:30:00-07:00",
            "stimulus_name": "test_stimulus",
            "stimulus_modalities": ["Visual"],
            "notes": output_params,
        }
    ]
    return _minimal_acquisition_json(epochs)


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
        self.unique_id = "test_data"
        self.tiff_stem = "test_stem"
        self.tiff_files = ["test_tiff_1.tif", "test_tiff_2.tif"]

        self.acquisition_file.write_text(
            _make_acquisition_json(self.tiff_stem, self.tiff_files)
        )

        self.settings = BergamoSettings(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            unique_id=self.unique_id,
            acquisition_fp=self.acquisition_file,
        )

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    @patch("aind_pophys_converter.bergamo_stitcher.ScanImageTiffReader")
    def test_run_converter_creates_hdf5(self, mock_reader):
        """Test that run_converter creates an HDF5 file with the
        expected structure.
        """
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
            "--unique-id",
            "test_unique_id",
            "--acquisition-file",
            str(self.acquisition_file),
        ]
        stitcher = BergamoTiffStitcher.from_args(args)
        self.assertEqual(stitcher.input_dir, self.input_dir)
        self.assertEqual(stitcher.output_dir, self.output_dir)
        self.assertEqual(stitcher.unique_id, "test_unique_id")

    def test_no_stimulus_epochs_raises_error(self):
        """Test that missing stimulus epochs raises ValueError."""
        self.acquisition_file.write_text(_minimal_acquisition_json([]))
        stitcher = BergamoTiffStitcher(self.settings)
        with self.assertRaises(ValueError) as context:
            stitcher.run_converter()
        self.assertIn("No stimulus epochs found", str(context.exception))

    def test_epoch_without_notes_is_skipped(self):
        """Test that epochs whose notes don't parse as JSON are skipped."""
        epochs = [
            {
                # No notes field — no output_parameters
                "stimulus_start_time": "2025-05-17T10:00:00-07:00",
                "stimulus_end_time": "2025-05-17T10:15:00-07:00",
                "stimulus_name": "no_params_epoch",
                "stimulus_modalities": ["Visual"],
            },
            {
                "stimulus_start_time": "2025-05-17T10:15:00-07:00",
                "stimulus_end_time": "2025-05-17T10:30:00-07:00",
                "stimulus_name": "with_params_epoch",
                "stimulus_modalities": ["Visual"],
                "notes": json.dumps(
                    {
                        "tiff_stem": self.tiff_stem,
                        "tiff_files": self.tiff_files,
                    }
                ),
            },
        ]
        self.acquisition_file.write_text(_minimal_acquisition_json(epochs))
        for fname in self.tiff_files:
            (self.input_dir / fname).touch()
        stitcher = BergamoTiffStitcher(self.settings)
        acquisition_data = stitcher._load_acquisition()
        result = stitcher._extract_tiff_from_acquisition(acquisition_data)
        # Only the epoch with valid notes is returned
        self.assertEqual(len(result), 1)
        self.assertIn(self.tiff_stem, result)


if __name__ == "__main__":
    unittest.main()
