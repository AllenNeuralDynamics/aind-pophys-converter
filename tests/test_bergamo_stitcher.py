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
    BergamoSettings, BergamoTiffStitcher)


class TestBergamoTiffStitcher(unittest.TestCase):
    """Test cases for BergamoTiffStitcher class."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for input and output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create a temporary session file
        self.session_file = self.input_dir / "session.json"
        self.unique_id = "test_data"
        
        # Create mock session data
        self.create_mock_session_file()
        
        # Create mock TIFF files
        self.create_mock_tiff_files()
        
        # Create BergamoSettings instance
        self.settings = BergamoSettings(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            unique_id=self.unique_id,
            session_fp=self.session_file
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def create_mock_session_file(self):
        """Create a mock session file for testing."""
        # Create a simplified session file
        tiff_stem = "test_stem"
        tiff_files = ["test_tiff_1.tif", "test_tiff_2.tif"]
        
        session_data = {
            "stimulus_epochs": [
                {
                    "stimulus_start_time": "2025-05-17T10:00:00",
                    "stimulus_name": "test_stimulus",
                    "output_parameters": {
                        "tiff_stem": tiff_stem,
                        "tiff_files": tiff_files
                    }
                }
            ]
        }
        
        # Write session data to file
        with open(self.session_file, "w") as f:
            json.dump(session_data, f)
    
    def create_mock_tiff_files(self):
        """Create mock TIFF files for testing."""
        # We'll mock the ScanImageTiffReader instead of creating actual TIFF files
        pass
    
    @patch('aind_pophys_converter.bergamo_stitcher.ScanImageTiffReader')
    def test_run_converter_creates_hdf5(self, mock_reader):
        """Test that run_converter creates an HDF5 file with the expected structure."""
        # Mock ScanImageTiffReader behavior
        mock_instance = Mock()
        mock_reader.return_value = mock_instance
        mock_instance.__enter__ = Mock(return_value=mock_instance)
        mock_instance.__exit__ = Mock(return_value=None)
        
        # Mock data() method to return a small image stack
        mock_instance.data.return_value = np.zeros((5, 10, 10), dtype=np.int16)
        
        # Mock metadata() method
        mock_instance.metadata.return_value = "Test metadata"
        
        # Mock shape() method to return image dimensions
        mock_instance.shape.return_value = (5, 10, 10)
        
        # Create mock TIFF files
        tiff_file_1 = self.input_dir / "test_tiff_1.tif"
        tiff_file_2 = self.input_dir / "test_tiff_2.tif"
        
        # Create empty files to be found by glob
        tiff_file_1.touch()
        tiff_file_2.touch()
        
        # Initialize the stitcher
        stitcher = BergamoTiffStitcher(self.settings)
        
        # Run the converter
        output_path = stitcher.run_converter()
        
        # Check if the output file exists
        self.assertTrue(os.path.exists(output_path[0]))
        
        # Check if it's a valid HDF5 file
        with h5py.File(output_path[0], 'r') as f:
            # Check if the expected datasets exist
            self.assertIn('data', f)
            self.assertIn('trial_locations', f)
            self.assertIn('epoch_locations', f)
            self.assertIn('epoch_filenames', f)
            self.assertIn('metadata', f)
            
            # Check data dimensions
            self.assertEqual(f['data'].shape[1:], (10, 10))


    def test_from_args_creates_stitcher(self):
        """Test that from_args creates a BergamoTiffStitcher from command line arguments."""
        # Test with minimal required arguments
        args = [
            "--input-dir", str(self.input_dir),
            "--session-file", str(self.session_file),
        ]
        
        stitcher = BergamoTiffStitcher.from_args(args)
        self.assertIsInstance(stitcher, BergamoTiffStitcher)
        self.assertEqual(stitcher.input_dir, self.input_dir)
        
    def test_from_args_with_all_arguments(self):
        """Test from_args with all optional arguments."""
        args = [
            "--input-dir", str(self.input_dir),
            "--output-dir", str(self.output_dir),
            "--unique-id", "test_unique_id",
            "--session-file", str(self.session_file),
        ]
        
        stitcher = BergamoTiffStitcher.from_args(args)
        self.assertEqual(stitcher.input_dir, self.input_dir)
        self.assertEqual(stitcher.output_dir, self.output_dir)
        self.assertEqual(stitcher.unique_id, "test_unique_id")
        
    def test_no_stimulus_epochs_raises_error(self):
        """Test that missing stimulus epochs raises ValueError."""
        # Create a session file without stimulus_epochs
        session_data = {"no_stimulus_epochs": []}
        
        with open(self.session_file, "w") as f:
            json.dump(session_data, f)
            
        stitcher = BergamoTiffStitcher(self.settings)
        
        with self.assertRaises(ValueError) as context:
            stitcher.run_converter()
            
        self.assertIn("No stimulus epochs found", str(context.exception))
        
    def test_epoch_without_output_parameters_skipped(self):
        """Test that epochs without output_parameters are skipped."""
        # Create session data with an epoch missing output_parameters
        session_data = {
            "stimulus_epochs": [
                {
                    "stimulus_start_time": "2025-05-17T10:00:00",
                    "stimulus_name": "test_stimulus_no_params",
                    # No output_parameters
                },
                {
                    "stimulus_start_time": "2025-05-17T10:00:00",
                    "stimulus_name": "test_stimulus_with_params",
                    "output_parameters": {
                        "tiff_stem": "test_stem",
                        "tiff_files": ["test_tiff_1.tif", "test_tiff_2.tif"]
                    }
                }
            ]
        }
        
        with open(self.session_file, "w") as f:
            json.dump(session_data, f)
            
        # Create mock TIFF files
        tiff_file_1 = self.input_dir / "test_tiff_1.tif"
        tiff_file_2 = self.input_dir / "test_tiff_2.tif"
        tiff_file_1.touch()
        tiff_file_2.touch()
        
        stitcher = BergamoTiffStitcher(self.settings)
        
        # This should work and only process the epoch with output_parameters
        with patch('aind_pophys_converter.bergamo_stitcher.ScanImageTiffReader'):
            session_data = stitcher._load_session()
            epochs = stitcher._extract_tiff_from_session(session_data)
            # Should only contain the epoch with output_parameters
            self.assertEqual(len(epochs), 1)
            self.assertIn("test_stem", epochs)


if __name__ == "__main__":
    unittest.main()
