"""Tests for full field utilities in aind_pophys_converter"""

import copy
import pathlib
import unittest
from itertools import product
from unittest.mock import Mock, patch

import numpy as np
from aind_ophys_utils.array_utils import normalize_array

from aind_pophys_converter.tiff_splitters import mkstemp_clean
from aind_pophys_converter.utils.full_field_utils import (
    _average_full_field_tiff,
    _get_origin,
    _get_stitched_tiff_shapes,
    _stitch_full_field_tiff,
    _validate_all_roi_same_size,
    stitch_full_field_tiff,
)
from aind_pophys_converter.utils.tiff_metadata import ScanImageMetadata

from .full_field_tiff_utils import (
    _create_full_field_tiff,
    _create_roi_metadata,
)


class TestFullFieldUtils(unittest.TestCase):
    """Test full field utils"""

    def setUp(self):
        """Setup"""
        import tempfile

        self.tmpdir_factory = tempfile.TemporaryDirectory

    def tearDown(self):
        """Teardown"""
        pass

    def test_average_full_field_tiff(self):
        """
        Test that _average_full_field_tiff properly averages
        over the pages of a full field TIFF image.
        """
        for numVolumes, numSlices in [(17, 13), (1, 1), (5, 1), (1, 27)]:
            with self.subTest(numVolumes=numVolumes, numSlices=numSlices):
                import tempfile

                tmpdir = pathlib.Path(
                    tempfile.mkdtemp(prefix="full_field_avg")
                )
                (tiff_path, expected_avg, metadata_dict) = (
                    _create_full_field_tiff(
                        numVolumes=numVolumes,
                        numSlices=numSlices,
                        seed=213455,
                        output_dir=tmpdir,
                    )
                )

                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata_dict),
                ):
                    actual_avg = _average_full_field_tiff(tiff_path=tiff_path)

                # verify that we are just getting a 2D image out
                self.assertEqual(len(actual_avg.shape), 2)
                np.testing.assert_allclose(actual_avg, expected_avg)

    def test_average_full_field_tiff_failures(self):
        """
        Test that expected error is raised when number of pages
        in the full field TIFF is not as expected
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="full_field_avg"))
        (tiff_path, expected_avg, metadata_dict) = _create_full_field_tiff(
            numVolumes=11, numSlices=13, seed=213455, output_dir=tmpdir
        )

        wrong_metadata = [
            {
                "SI.hStackManager.actualNumVolumes": 5,
                "SI.hStackManager.actualNumSlices": 2,
            }
        ]

        with patch(
            "tifffile.read_scanimage_metadata",
            new=Mock(return_value=wrong_metadata),
        ):
            with self.assertRaises(ValueError) as cm:
                _average_full_field_tiff(tiff_path=tiff_path)
            self.assertIn("implies n_pages", str(cm.exception))

    def test_get_stitched_tiff_shapes(self):
        """
        Test that _get_stitched_tiff_shapes returns the expected
        values for 'shape' and 'gap'
        """
        for nrois, roiy, roix, gap in product((2, 5), (6, 13), (2, 7), (3, 5)):
            with self.subTest(nrois=nrois, roiy=roiy, roix=roix, gap=gap):
                import tempfile

                nrows = gap * (nrois - 1) + roiy * nrois
                ncols = roix
                tmpdir = pathlib.Path(
                    tempfile.mkdtemp(prefix="stitched_shapes")
                )
                (tiff_path, avg_img, metadata) = _create_full_field_tiff(
                    numVolumes=5,
                    numSlices=3,
                    seed=112358,
                    output_dir=tmpdir,
                    nrows=nrows,
                    ncols=ncols,
                )

                roi_metadata = _create_roi_metadata(
                    nrois=nrois, roix=roix, roiy=roiy
                )

                metadata.append(roi_metadata)
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    tiff_metadata = ScanImageMetadata(tiff_path)

                result = _get_stitched_tiff_shapes(
                    tiff_metadata=tiff_metadata, avg_img=avg_img
                )

                self.assertEqual(result["gap"], gap)
                self.assertEqual(result["shape"], (roiy, roix * nrois))

    def test_get_stitched_tiff_shapes_validation(self):
        """
        Test that _get_stitched_tiff_shapes throws expected error
        when the average image has the wrong shape
        """
        for dcols, drows in [(0, 1), (1, 0)]:
            with self.subTest(dcols=dcols, drows=drows):
                import tempfile

                nrois = 7
                roiy = 11
                roix = 13
                gap = 3
                roi_metadata = _create_roi_metadata(
                    nrois=nrois, roix=roix, roiy=roiy
                )
                tmpdir = pathlib.Path(
                    tempfile.mkdtemp(prefix="stitched_shapes")
                )
                (tiff_path, avg_img, metadata) = _create_full_field_tiff(
                    numVolumes=5,
                    numSlices=3,
                    seed=112358,
                    output_dir=tmpdir,
                    nrows=gap * (nrois - 1) + roiy * nrois + drows,
                    ncols=roix + dcols,
                )
                metadata.append(roi_metadata)
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    tiff_metadata = ScanImageMetadata(tiff_path)
                with self.assertRaises(ValueError) as cm:
                    _get_stitched_tiff_shapes(
                        tiff_metadata=tiff_metadata, avg_img=avg_img
                    )
                self.assertIn("expected average over pages", str(cm.exception))

    def test_get_stitched_tiff_shapes_errors(self):
        """
        Test that _get_stitched_tiff_shapes raises
        errors when the ROI metadata is not as expected
        """
        import tempfile

        nrois = 3
        roix = 11
        roiy = 7
        gap = 3
        nrows = gap * (nrois - 1) + roiy * nrois
        ncols = 23
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="stitched_shapes"))
        (tiff_path, avg_img, baseline_metadata) = _create_full_field_tiff(
            numVolumes=5,
            numSlices=3,
            seed=112358,
            output_dir=tmpdir,
            nrows=nrows,
            ncols=ncols,
        )
        roi_metadata = _create_roi_metadata(nrois=nrois, roix=roix, roiy=roiy)
        baseline_metadata.append(roi_metadata)

        # if an ROI has more than one scanfield
        metadata = copy.deepcopy(baseline_metadata)
        metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"][1][
            "scanfields"
        ] = (1, 2)

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path)
        with self.assertRaises(ValueError) as cm:
            _get_stitched_tiff_shapes(
                tiff_metadata=tiff_metadata, avg_img=avg_img
            )
        self.assertIn("more than one scanfield", str(cm.exception))

        # if an ROI has different resolution than others
        metadata = copy.deepcopy(baseline_metadata)
        metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"][1]["scanfields"][
            "pixelResolutionXY"
        ] = (roix, 999)

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path)
        with self.assertRaises(ValueError) as cm:
            _get_stitched_tiff_shapes(
                tiff_metadata=tiff_metadata, avg_img=avg_img
            )
        self.assertIn("different pixel resolutions", str(cm.exception))

        # if an ROI has different resolution than others
        metadata = copy.deepcopy(baseline_metadata)
        metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"][1]["scanfields"][
            "pixelResolutionXY"
        ] = (999, roiy)

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path)
        with self.assertRaises(ValueError) as cm:
            _get_stitched_tiff_shapes(
                tiff_metadata=tiff_metadata, avg_img=avg_img
            )
        self.assertIn("different pixel resolutions", str(cm.exception))

    def test_stitch_full_field_tiff(self):
        """
        Test that _stitch_full_field_tiff correctly disassembles and
        re-assembles the average image
        """
        for n_rois, roi_rows, roi_cols, gap in [
            (3, 11, 17, 2),
            (2, 27, 31, 1),
            (7, 18, 29, 4),
        ]:
            with self.subTest(
                n_rois=n_rois, roi_rows=roi_rows, roi_cols=roi_cols, gap=gap
            ):
                rng = np.random.default_rng(58132134)
                nrows = gap * (n_rois - 1) + roi_rows * n_rois
                ncols = roi_cols
                avg_img = np.zeros((nrows, ncols), dtype=float)
                avg_img *= np.nan
                expected_stitched = np.zeros(
                    (roi_rows, n_rois * roi_cols), dtype=float
                )
                list_of_rois = []
                for i_roi in range(n_rois):
                    r0 = i_roi * (roi_rows + gap)
                    r1 = r0 + roi_rows
                    tile = rng.random((roi_rows, roi_cols))
                    list_of_rois.append(tile)
                    avg_img[r0:r1, :] = tile
                    expected_stitched[
                        :, i_roi * roi_cols: (i_roi + 1) * roi_cols
                    ] = tile
                metadata = [
                    "nothing",
                    _create_roi_metadata(
                        nrois=n_rois, roix=roi_cols, roiy=roi_rows
                    ),
                ]
                import tempfile

                tmpdir = pathlib.Path(
                    tempfile.mkdtemp(prefix="stitch_full_field")
                )
                tiff_path = pathlib.Path(
                    mkstemp_clean(dir=tmpdir, suffix=".tiff")
                )
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    tiff_metadata = ScanImageMetadata(tiff_path)
                stitched_img = _stitch_full_field_tiff(
                    tiff_metadata=tiff_metadata, avg_img=avg_img
                )
                # make sure the gap pixels were all ignored
                self.assertTrue(np.all(np.logical_not(np.isnan(stitched_img))))
                np.testing.assert_allclose(stitched_img, expected_stitched)

    def test_user_facing_stitch_full_field_tiff(self):
        """
        Test that stitch_full_field_tiff correctly disassembles and
        re-assembles the average image
        """
        for n_rois, roi_rows, roi_cols, gap in [
            (3, 11, 17, 2),
            (2, 27, 31, 1),
            (7, 18, 29, 4),
        ]:
            with self.subTest(
                n_rois=n_rois, roi_rows=roi_rows, roi_cols=roi_cols, gap=gap
            ):
                rng = np.random.default_rng(58132134)
                nrows = gap * (n_rois - 1) + roi_rows * n_rois
                ncols = roi_cols
                avg_img = np.zeros((nrows, ncols), dtype=float)
                avg_img *= np.nan
                expected_stitched = np.zeros(
                    (roi_rows, n_rois * roi_cols), dtype=float
                )
                list_of_rois = []
                for i_roi in range(n_rois):
                    r0 = i_roi * (roi_rows + gap)
                    r1 = r0 + roi_rows
                    tile = rng.random((roi_rows, roi_cols))
                    list_of_rois.append(tile)
                    avg_img[r0:r1, :] = tile
                    expected_stitched[
                        :, i_roi * roi_cols: (i_roi + 1) * roi_cols
                    ] = tile
                expected_stitched = normalize_array(
                    array=expected_stitched, dtype=np.uint16
                )
                metadata = [
                    "nothing",
                    _create_roi_metadata(
                        nrois=n_rois, roix=roi_cols, roiy=roi_rows
                    ),
                ]
                import tempfile

                tmpdir = pathlib.Path(
                    tempfile.mkdtemp(prefix="stitch_full_field")
                )
                tiff_path = pathlib.Path(
                    mkstemp_clean(dir=tmpdir, suffix=".tiff")
                )
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    with patch(
                        "aind_pophys_converter.utils.full_field_utils."
                        "_average_full_field_tiff",
                        new=Mock(return_value=avg_img),
                    ):
                        stitched_img = stitch_full_field_tiff(
                            tiff_path=tiff_path
                        )
                # make sure the gap pixels were all ignored
                self.assertTrue(np.all(np.logical_not(np.isnan(stitched_img))))
                self.assertEqual(stitched_img.dtype, np.uint16)
                np.testing.assert_allclose(stitched_img, expected_stitched)

    def test_get_origin(self):
        """
        Test that _get_origin actually returns the physical space
        coordinates of the origin implied by a ScanImageMetadata object
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="get_origin"))
        tiff_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        metadata = ["nothing", dict()]
        roi_metadata = [
            {"scanfields": {"centerXY": [1.1, 2.1], "sizeXY": [1.5, 0.62]}},
            {"scanfields": {"centerXY": [0.5, 3.4], "sizeXY": [0.02, 6.0]}},
        ]
        metadata[1] = {
            "RoiGroups": {"imagingRoiGroup": {"rois": roi_metadata}}
        }
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            metadata_obj = ScanImageMetadata(tiff_path)
            actual = _get_origin(metadata_obj)
        expected = [0.35, 0.4]
        np.testing.assert_allclose(expected, actual)

    def test_validate_all_roi_same_size(self):
        """Test validate of roi of same size"""
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="validate_roi_size"))
        tiff_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        metadata = ["nothing", None]
        # in case where the ROIs do have the same size, verify
        # that _validate_all_roi_same_size does, in fact, return
        # pixelResolutionXY and sizeXY
        roi_metadata = [
            {
                "scanfields": {
                    "sizeXY": [2.1, 3.2],
                    "pixelResolutionXY": [45, 33],
                }
            },
            {
                "scanfields": {
                    "sizeXY": [2.1, 3.2],
                    "pixelResolutionXY": [45, 33],
                }
            },
        ]
        metadata[1] = {
            "RoiGroups": {"imagingRoiGroup": {"rois": roi_metadata}}
        }
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            metadata_obj = ScanImageMetadata(tiff_path)
            (resolution, size) = _validate_all_roi_same_size(metadata_obj)
            self.assertEqual(resolution, (45, 33))
            np.testing.assert_allclose(size, [2.1, 3.2])
        # test for error in case where sizeXY differs
        roi_metadata = [
            {
                "scanfields": {
                    "sizeXY": [2.11, 3.2],
                    "pixelResolutionXY": [45, 33],
                }
            },
            {
                "scanfields": {
                    "sizeXY": [2.1, 3.2],
                    "pixelResolutionXY": [45, 33],
                }
            },
        ]
        metadata[1] = {
            "RoiGroups": {"imagingRoiGroup": {"rois": roi_metadata}}
        }
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            metadata_obj = ScanImageMetadata(tiff_path)
            with self.assertRaises(ValueError) as cm:
                _validate_all_roi_same_size(metadata_obj)
            self.assertIn("different physical units", str(cm.exception))
        # test for error in case where pixelResolutionXY differs
        roi_metadata = [
            {
                "scanfields": {
                    "sizeXY": [2.1, 3.2],
                    "pixelResolutionXY": [45, 33],
                }
            },
            {
                "scanfields": {
                    "sizeXY": [2.1, 3.2],
                    "pixelResolutionXY": [46, 33],
                }
            },
        ]
        metadata[1] = {
            "RoiGroups": {"imagingRoiGroup": {"rois": roi_metadata}}
        }
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            metadata_obj = ScanImageMetadata(tiff_path)
            with self.assertRaises(ValueError) as cm:
                _validate_all_roi_same_size(metadata_obj)
            self.assertIn("different pixel resolutions", str(cm.exception))

    def test_write_out_stitched_full_field_image_success(self):
        """Test write_out_stitched_full_field_image creates HDF5 file with
        expected datasets.
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="full_field_output"))

        # Create mock input files
        avg_tiff_path = tmpdir / "avg_surface.tif"
        full_field_tiff_path = tmpdir / "full_field.tif"
        output_path = tmpdir / "output.h5"

        # Mock the dependencies
        with patch(
            "aind_pophys_converter.utils.full_field_utils.ScanImageMetadata"
        ) as mock_metadata:
            with patch(
                "aind_pophys_converter.utils.full_field_utils.stitch_full_field_tiff"  # noqa: E501
            ) as mock_stitch:
                with patch(
                    "aind_pophys_converter.utils.full_field_utils.AvgImageTiffSplitter"  # noqa: E501
                ) as mock_splitter:
                    with patch(
                        "aind_pophys_converter.utils.full_field_utils._insert_rois_into_surface_img"  # noqa: E501
                    ) as mock_insert:
                        # Set up mocks
                        mock_metadata.return_value.raw_metadata = {
                            "test": "metadata"
                        }
                        mock_stitch.return_value = np.ones(
                            (10, 10), dtype=np.uint16
                        )
                        mock_splitter.return_value.raw_metadata = {
                            "roi": "data"
                        }
                        mock_insert.return_value = np.ones(
                            (10, 10), dtype=np.uint16
                        )

                        from aind_pophys_converter.utils.full_field_utils import (  # noqa: E501
                            write_out_stitched_full_field_image,
                        )

                        # Call the function
                        write_out_stitched_full_field_image(
                            path_to_avg_tiff=avg_tiff_path,
                            path_to_full_field_tiff=full_field_tiff_path,
                            output_path=output_path,
                        )

                        # Verify HDF5 file was created with expected datasets
                        import h5py as h5

                        with h5.File(output_path, "r") as f:
                            self.assertIn("stitched_full_field", f)
                            self.assertIn("stitched_full_field_with_rois", f)
                            self.assertIn("surface_roi_metadata", f)
                            self.assertIn("full_field_metadata", f)

    @patch("aind_pophys_converter.utils.full_field_utils.logging")
    def test_write_out_stitched_full_field_image_exception_handling(
        self, mock_logging
    ):
        """Test write_out_stitched_full_field_image handles exceptions
        gracefully.
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="full_field_error"))

        # Create mock input files
        avg_tiff_path = tmpdir / "avg_surface.tif"
        full_field_tiff_path = tmpdir / "full_field.tif"
        output_path = tmpdir / "output.h5"

        # Mock ScanImageMetadata to raise an exception
        with patch(
            "aind_pophys_converter.utils.full_field_utils.ScanImageMetadata",
            side_effect=Exception("Test error"),
        ):
            from aind_pophys_converter.utils.full_field_utils import (
                write_out_stitched_full_field_image,
            )

            # This should not raise an exception, but should log a warning
            write_out_stitched_full_field_image(
                path_to_avg_tiff=avg_tiff_path,
                path_to_full_field_tiff=full_field_tiff_path,
                output_path=output_path,
            )

            # Verify warning was logged
            mock_logging.warning.assert_called_once()
            warning_call_args = mock_logging.warning.call_args[0][0]
            self.assertIn(
                "Full field TIFF generation failed", warning_call_args
            )
            self.assertIn("Test error", warning_call_args)

    def test_get_full_field_path_missing_platform_key(self):
        """Test get_full_field_path returns None when platform_json_path
        is missing.
        """
        from aind_pophys_converter.utils.full_field_utils import (
            get_full_field_path,
        )

        runner_args = {"other_key": "value"}
        result = get_full_field_path(runner_args)
        self.assertIsNone(result)

    def test_get_full_field_path_platform_key_none(self):
        """Test get_full_field_path returns None when platform_json_path
        is None.
        """
        from aind_pophys_converter.utils.full_field_utils import (
            get_full_field_path,
        )

        runner_args = {"platform_json_path": None}
        result = get_full_field_path(runner_args)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
