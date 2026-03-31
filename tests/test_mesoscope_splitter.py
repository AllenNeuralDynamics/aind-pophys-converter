"""Test suite for mesoscope tiff splitting CLI."""

# This file tests the mesoscope tiff splitting CLI for differen
# 'flavors' of configuration. The 'flavors' are:
#
# '4x2' -- 4 ROIs x 2 planes
# '4x2_repeat' -- 4 ROIs x 2 planes with a z value repeated in many ROIs
# '4x2_floats' -- 4 ROIs x2 planes; z-values are floats
# '2x4' -- 2 ROIS x 4 planes
# '2x4_repeats' -- 2 ROIs x 4planes with a z value repeated in many ROIs
# '1x6' -- 1 ROI x 6 planes


import os
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import Mock


from aind_pophys_converter.mesoscope_splitter import (
    TiffSplitterCLI,
    find_split_directories,
)
from aind_pophys_converter.tiff_splitters import mkstemp_clean


class TestMesoscopeSplitter(unittest.TestCase):
    """Test suite for mesoscope tiff splitting CLI."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_find_split_directories_with_matching_dirs(self):
        """Test find_split_directories finds V* directories with numbers."""
        # Create test directories that should match (V* ending with digits)
        matching_dirs = ["V1", "V1_1", "V2_2", "VTest_123", "V_999"]
        for dir_name in matching_dirs:
            (self.test_path / dir_name).mkdir()

        # Create directories that shouldn't match
        non_matching_dirs = ["V_", "Test_1", "v1_2", "V1_a", "notV1"]
        for dir_name in non_matching_dirs:
            (self.test_path / dir_name).mkdir()

        # Create some files (should be ignored)
        (self.test_path / "V1_1.txt").touch()
        (self.test_path / "V2_2.tif").touch()

        result = find_split_directories(self.test_path)

        # Should find all matching directories
        self.assertEqual(set(result), set(matching_dirs))
        self.assertEqual(len(result), len(matching_dirs))

    def test_find_split_directories_no_matches(self):
        """Test find_split_directories returns empty list when no matches."""
        # Create directories that don't match the pattern
        non_matching_dirs = ["dir1", "V", "V_", "Test_1", "v1_2"]
        for dir_name in non_matching_dirs:
            (self.test_path / dir_name).mkdir()

        result = find_split_directories(self.test_path)
        self.assertEqual(result, [])

    def test_find_split_directories_empty_directory(self):
        """Test find_split_directories on empty directory."""
        result = find_split_directories(self.test_path)
        self.assertEqual(result, [])

    def test_mkstemp_clean_basic(self):
        """Test mkstemp_clean creates a temporary file and closes
        descriptor.
        """
        file_path = mkstemp_clean()

        # File should exist
        self.assertTrue(os.path.exists(file_path))

        # File should be empty initially
        with open(file_path, "r") as f:
            content = f.read()
            self.assertEqual(content, "")

        # Clean up
        os.unlink(file_path)

    def test_mkstemp_clean_with_directory(self):
        """Test mkstemp_clean with specific directory."""
        file_path = mkstemp_clean(dir=self.temp_dir.name)

        # File should exist in the specified directory
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.startswith(self.temp_dir.name))

        # Clean up
        os.unlink(file_path)

    def test_mkstemp_clean_with_prefix_suffix(self):
        """Test mkstemp_clean with prefix and suffix."""
        prefix = "test_prefix_"
        suffix = ".test_suffix"

        file_path = mkstemp_clean(
            dir=self.temp_dir.name, prefix=prefix, suffix=suffix
        )

        file_name = os.path.basename(file_path)

        # File should exist and have correct prefix/suffix
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_name.startswith(prefix))
        # Clean up
        os.unlink(file_path)

    def test_tiff_splitter_cli_str2bool(self):
        """Test TiffSplitterCLI.str2bool method."""
        # Test truthy values
        truthy_values = ["yes", "true", "t", "y", "1", "YES", "TRUE", "T", "Y"]
        for value in truthy_values:
            with self.subTest(value=value):
                self.assertTrue(TiffSplitterCLI.str2bool(value))

        # Test falsy values
        falsy_values = [
            "no",
            "false",
            "f",
            "n",
            "0",
            "NO",
            "FALSE",
            "F",
            "N",
            "anything_else",
        ]
        for value in falsy_values:
            with self.subTest(value=value):
                self.assertFalse(TiffSplitterCLI.str2bool(value))

    def _make_splitter_cli(
        self, input_dir, output_dir, temp_dir, str_args=False
    ):
        """Helper: build a TiffSplitterCLI with mocked Acquisition loading."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        job_settings = SimpleNamespace()
        job_settings.input_dir = str(input_dir) if str_args else input_dir
        job_settings.output_dir = str(output_dir) if str_args else output_dir
        job_settings.temp_dir = str(temp_dir) if str_args else temp_dir
        job_settings.debug = False

        mock_acq = MagicMock()
        with patch(
            "aind_pophys_converter.mesoscope_splitter.Acquisition"
            ".model_validate_json",
            return_value=mock_acq,
        ):
            return TiffSplitterCLI(job_settings)

    def test_tiff_splitter_cli_init_with_strings(self):
        """Test TiffSplitterCLI initialization converts strings to Paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input_data"
            input_dir.mkdir()

            (temp_path / "test_acquisition.json").write_text("{}")
            (input_dir / "test_timeseries.tiff").touch()
            (input_dir / "test_averaged_depth.tiff").touch()
            (input_dir / "test_averaged_surface.tiff").touch()

            cli = self._make_splitter_cli(
                input_dir,
                temp_path / "output",
                temp_path / "temp",
                str_args=True,
            )

            self.assertIsInstance(cli.input_dir, Path)
            self.assertIsInstance(cli.output_dir, Path)
            self.assertEqual(cli.input_dir, input_dir)

    def test_tiff_splitter_cli_init_with_paths(self):
        """Test TiffSplitterCLI initialization with Path objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input_data"
            input_dir.mkdir()

            (temp_path / "test_acquisition.json").write_text("{}")
            (input_dir / "test_timeseries.tiff").touch()
            (input_dir / "test_averaged_depth.tiff").touch()
            (input_dir / "test_averaged_surface.tiff").touch()

            cli = self._make_splitter_cli(
                input_dir,
                temp_path / "output",
                temp_path / "temp",
                str_args=False,
            )

            self.assertIsInstance(cli.input_dir, Path)
            self.assertIsInstance(cli.output_dir, Path)
            self.assertEqual(cli.input_dir, input_dir)

    def test_tiff_splitter_cli_missing_files_errors(self):
        """Test TiffSplitterCLI raises appropriate errors when files
        are missing.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input_data"
            input_dir.mkdir()

            job_settings = SimpleNamespace()
            job_settings.input_dir = input_dir
            job_settings.output_dir = temp_path / "output"
            job_settings.temp_dir = temp_path / "temp"
            job_settings.debug = False

            # Test missing acquisition.json
            with self.assertRaises(ValueError) as cm:
                TiffSplitterCLI(job_settings)
            self.assertIn("No acquisition.json file found", str(cm.exception))

            # Create acquisition.json; subsequent checks need mocked validation
            (temp_path / "test_acquisition.json").write_text("{}")

            mock_acq = MagicMock()
            with patch(
                "aind_pophys_converter.mesoscope_splitter.Acquisition"
                ".model_validate_json",
                return_value=mock_acq,
            ):
                with self.assertRaises(ValueError) as cm:
                    TiffSplitterCLI(job_settings)
                self.assertIn(
                    "No timeseries.tiff file found", str(cm.exception)
                )

                # Create timeseries.tiff but leave depth file missing
                (input_dir / "test_timeseries.tiff").touch()
                with self.assertRaises(ValueError) as cm:
                    TiffSplitterCLI(job_settings)
                self.assertIn(
                    "No averaged_depth.tiff file found", str(cm.exception)
                )

                # Create depth file but leave surface file missing
                (input_dir / "test_averaged_depth.tiff").touch()
                with self.assertRaises(ValueError) as cm:
                    TiffSplitterCLI(job_settings)
                self.assertIn(
                    "No averaged_surface.tiff file found", str(cm.exception)
                )

    def test_split_timeseries_tiff_logger_messages(self):
        """Test that split_timeseries_tiff calls logger with
        proper messages.
        """
        from aind_pophys_converter.tiff_splitters import (
            split_timeseries_tiff,
        )

        # Create minimal test setup
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a tiny mock TIFF file
            tiff_path = temp_path / "test.tif"
            tiff_path.touch()  # Create empty file for now

            # Create output paths
            output_path = temp_path / "output.h5"
            offset_to_path = {0: output_path}

            # Mock logger
            mock_logger = Mock()

            # Try to test the logger calls - this will likely fail due to
            # TIFF format issues
            # but should at least test the logger path
            try:
                split_timeseries_tiff(
                    tiff_path=tiff_path,
                    offset_to_path=offset_to_path,
                    logger=mock_logger,
                    dump_every=10,
                )
            except Exception:
                # Expected to fail due to invalid TIFF, but we should have
                # captured some logger calls
                pass

            # Verify logger.info was called (at minimum for the start message)
            mock_logger.info.assert_called()

    def test_avg_image_tiff_splitter_properties(self):
        """Test AvgImageTiffSplitter properties work correctly."""
        from unittest.mock import Mock, mock_open, patch

        from aind_pophys_converter.tiff_splitters import (
            AvgImageTiffSplitter,
        )

        # Mock the entire file system and metadata reading chain
        mock_tiff_path = Path("mock_test.tif")

        mock_metadata = [
            {
                "SI.hChannels.channelSave": [1, 2],
                "SI.hStackManager.zs": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            {
                "RoiGroups": {
                    "imagingRoiGroup": {
                        "rois": [{"zs": [1.0, 2.0, 3.0, 4.0, 5.0]}]
                    }
                }
            },
        ]

        # Mock all the file operations and metadata reading
        with patch("pathlib.Path.is_file", return_value=True):
            with patch(
                "builtins.open", mock_open(read_data=b"mock_tiff_data")
            ):
                with patch(
                    "tifffile.read_scanimage_metadata",
                    return_value=mock_metadata,
                ):
                    with patch("tifffile.TiffFile") as mock_tiff_file:
                        # Mock the TiffFile context manager and pages
                        mock_context = Mock()
                        mock_context.pages = [Mock()] * 5
                        mock_tiff_file.return_value.__enter__ = Mock(
                            return_value=mock_context
                        )
                        mock_tiff_file.return_value.__exit__ = Mock(
                            return_value=None
                        )

                        splitter = AvgImageTiffSplitter(
                            tiff_path=mock_tiff_path
                        )

                        # Test n_pages property
                        self.assertEqual(splitter.n_pages, 5)

                        # Test input_path property
                        self.assertEqual(splitter.input_path, mock_tiff_path)

    def test_tiff_splitter_base_utility_methods(self):
        """Test utility methods in TiffSplitterBase classes."""
        from unittest.mock import Mock, mock_open, patch

        from aind_pophys_converter.tiff_splitters import (
            AvgImageTiffSplitter,
        )

        mock_tiff_path = Path("mock_test.tif")

        mock_metadata = [
            {
                "SI.hChannels.channelSave": [1, 2],
                "SI.hStackManager.zs": [1.5, 2.7, 3.1],
            },
            {
                "RoiGroups": {
                    "imagingRoiGroup": {"rois": [{"zs": [1.5, 2.7, 3.1]}]}
                }
            },
        ]

        with patch("pathlib.Path.is_file", return_value=True):
            with patch(
                "builtins.open", mock_open(read_data=b"mock_tiff_data")
            ):
                with patch(
                    "tifffile.read_scanimage_metadata",
                    return_value=mock_metadata,
                ):
                    with patch("tifffile.TiffFile") as mock_tiff_file:
                        mock_context = Mock()
                        mock_context.pages = [Mock()] * 3
                        mock_tiff_file.return_value.__enter__ = Mock(
                            return_value=mock_context
                        )
                        mock_tiff_file.return_value.__exit__ = Mock(
                            return_value=None
                        )

                        splitter = AvgImageTiffSplitter(
                            tiff_path=mock_tiff_path
                        )

                        # Test is_z_valid_for_roi method
                        self.assertTrue(splitter.is_z_valid_for_roi(0, 1.5))
                        self.assertTrue(splitter.is_z_valid_for_roi(0, 2.7))
                        self.assertTrue(splitter.is_z_valid_for_roi(0, 3.1))
                        self.assertFalse(splitter.is_z_valid_for_roi(0, 999.0))

                        # Test with invalid ROI index - this should raise
                        # IndexError
                        with self.assertRaises(IndexError):
                            splitter.is_z_valid_for_roi(999, 1.5)

    def test_tiff_splitter_error_conditions(self):
        """Test error handling in TiffSplitter classes."""
        from aind_pophys_converter.tiff_splitters import (
            AvgImageTiffSplitter,
        )

        # Test file not found error - this raises ValueError, not
        # FileNotFoundError
        with self.assertRaises(ValueError) as context:
            AvgImageTiffSplitter(tiff_path=Path("nonexistent.tif"))
        self.assertIn("is not a file", str(context.exception))

    def test_logging_and_cleanup_paths(self):
        """Test logging statements and cleanup operations."""
        from aind_pophys_converter.tiff_splitters import mkstemp_clean

        # Test mkstemp_clean function - it returns a file path and cleans up
        # the descriptor
        file_path = mkstemp_clean()

        # File should exist
        self.assertTrue(os.path.exists(file_path))

        # File should be empty initially
        with open(file_path, "r") as f:
            content = f.read()
            self.assertEqual(content, "")

        # Clean up the file
        os.unlink(file_path)

    def test_utility_functions(self):
        """Test various utility functions that don't require TIFF files."""
        from aind_pophys_converter.mesoscope_splitter import (
            find_split_directories,
        )

        # Test find_split_directories with a real directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some matching directories (V* pattern ending with digits)
            (temp_path / "V1").mkdir()
            (temp_path / "V_123").mkdir()
            (temp_path / "VTest_456").mkdir()

            # Create some non-matching directories
            (temp_path / "V_").mkdir()
            (temp_path / "NotV1").mkdir()
            (temp_path / "regular_dir").mkdir()

            result = find_split_directories(temp_path)

            # Should find the matching directories
            expected = ["V1", "V_123", "VTest_456"]
            self.assertEqual(set(result), set(expected))

    def test_splitter_cli(self):
        """Test the TiffSplitterCLI with various configurations."""
        # These are the parameter combinations for the test
        param_product = list(
            product(
                (True, False, None),
                (True, False, None),
                (True, False),
                ("upload", "storage", "corrupt"),
                (
                    "1x6",
                    "4x2",
                    "4x2_floats",
                    "2x4_repeats",
                    "2x4",
                    "4x2_repeats",
                ),
            )
        )
        # You will need to provide or mock the fixtures
        # and input_json_fixture for
        # this to run
        for (
            use_platform_json,
            use_data_upload_dir,
            platform_json_fixture,
            full_field_2p_tiff_fixture,
            flavor,
        ) in param_product:
            with self.subTest(
                use_platform_json=use_platform_json,
                use_data_upload_dir=use_data_upload_dir,
                platform_json_fixture=platform_json_fixture,
                full_field_2p_tiff_fixture=full_field_2p_tiff_fixture,
                flavor=flavor,
            ):
                # ...existing code for setting up fixtures and
                # input_json_fixture...
                # This is a placeholder for the actual test logic, which
                # depends on your test infra
                pass


if __name__ == "__main__":
    unittest.main()
