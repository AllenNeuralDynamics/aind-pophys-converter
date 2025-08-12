import copy
import pathlib
import unittest
from unittest.mock import Mock, patch

from aind_pophys_converter.mesoscope_splitter import mkstemp_clean
from aind_pophys_converter.utils.tiff_metadata import ScanImageMetadata


def mock_2x3_metadata_zsAllActuators():
    """
    Return a list representing everything we need from
    the metadata for 2 ROIs each sampled at 3 depths

    (use SI.hStackManager.zsAllActuators)

    Return
    ------
    metadata
    expected breakdown of all_zs
    expected result of defined_rois
    """

    z_breakdown = [[1, 4, 2], [9, 5, 11], [2, 6, 1]]

    si_metadata = dict()
    si_metadata["SI.hStackManager.zsAllActuators"] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = [{"zs": [1, 2, 4]}, {"zs": [5, 9, 11]}, {"zs": [1, 2, 6]}]
    img_group["rois"] = rois
    roi_metadata["RoiGroups"] = {"imagingRoiGroup": img_group}

    metadata = [si_metadata, roi_metadata]
    return (metadata, copy.deepcopy(z_breakdown), copy.deepcopy(rois))


def mock_2x3_metadata_zs():
    """
    Return a list representing everything we need from
    the metadata for 2 ROIs each sampled at 3 depths

    (use SI.hStackManager.zs)
    """

    z_breakdown = [[1, 4, 2], [9, 5, 11], [2, 6, 1]]

    si_metadata = dict()
    si_metadata["SI.hStackManager.zs"] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = [{"zs": [1, 2, 4]}, {"zs": [5, 9, 11]}, {"zs": [1, 2, 6]}]
    img_group["rois"] = rois
    roi_metadata["RoiGroups"] = {"imagingRoiGroup": img_group}

    metadata = [si_metadata, roi_metadata]
    return (metadata, copy.deepcopy(z_breakdown), copy.deepcopy(rois))


def mock_2x3_metadata_nozs():
    """
    Return a list representing everything we need from
    the metadata for 2 ROIs each sampled at 3 depths

    (use invalid key for SI.hStackManager.zsAllActuators)
    """

    z_breakdown = [[1, 4, 2], [9, 5, 11], [2, 6, 1]]

    si_metadata = dict()
    si_metadata["SI.hStackManager.silly"] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = [{"zs": [1, 2, 4]}, {"zs": [5, 9, 11]}, {"zs": [1, 2, 6]}]
    img_group["rois"] = rois
    roi_metadata["RoiGroups"] = {"imagingRoiGroup": img_group}

    metadata = [si_metadata, roi_metadata]
    return metadata


def mock_3x1_metadata_zsAllActuators():
    """
    Return a list representing everything we need from
    the metadata for 3 ROIs each sampled at 1 depth

    (use SI.hStackManager.zsAllActuators)
    """

    z_breakdown = [[1, 0], [9, 0], [2, 0]]

    si_metadata = dict()
    si_metadata["SI.hStackManager.zsAllActuators"] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = [{"zs": 1}, {"zs": 9}, {"zs": 2}]
    img_group["rois"] = rois
    roi_metadata["RoiGroups"] = {"imagingRoiGroup": img_group}

    metadata = [si_metadata, roi_metadata]
    return (metadata, copy.deepcopy(z_breakdown), copy.deepcopy(rois))


def mock_1x3_metadata_zsAllActuators():
    """
    Return a list representing everything we need from
    the metadata for 1 ROI each sampled at 3 depths

    (use SI.hStackManager.zsAllActuators)
    """

    z_breakdown = [[1, 2, 9]]

    si_metadata = dict()
    si_metadata["SI.hStackManager.zsAllActuators"] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = {"zs": [1, 2, 9]}
    img_group["rois"] = rois
    roi_metadata["RoiGroups"] = {"imagingRoiGroup": img_group}

    metadata = [si_metadata, roi_metadata]
    return (
        metadata,
        copy.deepcopy(z_breakdown),
        [
            copy.deepcopy(rois),
        ],
    )


class TestTiffMetadata(unittest.TestCase):
    def test_no_file_error(self):
        """
        Test that, ScanImageMetadata raises an error if you
        give it a path that doesn't point to a file
        """
        dummy_path = pathlib.Path("not_a_file.tiff")
        with self.assertRaises(ValueError) as cm:
            ScanImageMetadata(tiff_path=dummy_path)
        self.assertIn("is not a file", str(cm.exception))

    def test_all_zs(self):
        """
        Test that ScanImageMetadata.all_zs() returns the expected result
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="test_all_zs"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        param_list = [
            mock_2x3_metadata_zs(),
            mock_2x3_metadata_zsAllActuators(),
            mock_1x3_metadata_zsAllActuators(),
            mock_3x1_metadata_zsAllActuators(),
        ]
        for metadata_fixture in param_list:
            expected = metadata_fixture[1]
            with self.subTest(metadata_fixture=metadata_fixture):
                with patch(
                    to_replace, new=Mock(return_value=metadata_fixture[0])
                ):
                    metadata = ScanImageMetadata(tiff_path=tmp_path)
                    self.assertEqual(expected, metadata.all_zs())

    def test_raw_metadata(self):
        """
        Test that ScanImageMetadata.raw_metadata returns the expected result
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="test_all_zs"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        param_list = [
            mock_2x3_metadata_zs()[0],
            mock_2x3_metadata_zsAllActuators()[0],
            mock_1x3_metadata_zsAllActuators()[0],
            mock_3x1_metadata_zsAllActuators()[0],
        ]
        for metadata_fixture in param_list:
            expected = metadata_fixture
            with self.subTest(metadata_fixture=metadata_fixture):
                with patch(
                    to_replace, new=Mock(return_value=metadata_fixture)
                ):
                    metadata = ScanImageMetadata(tiff_path=tmp_path)
                    self.assertEqual(expected, metadata.raw_metadata)

    def test_all_zs_error(self):
        """
        Test that expected exception is thrown when metadata lacks both
        SI.hStackManager.zs and SI.hStackManager.zsAllActuators
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="test_all_zs"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        metadata_nozs = mock_2x3_metadata_nozs()
        with patch(to_replace, new=Mock(return_value=metadata_nozs)):
            metadata = ScanImageMetadata(tiff_path=tmp_path)
        with self.assertRaises(ValueError) as cm:
            metadata.all_zs()
        self.assertIn("Cannot load all_zs", str(cm.exception))

    def test_zs_for_roi(self):
        """
        Test behavior of ScanImageMetadata.zs_for_roi
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="test_all_zs"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        param_list = [
            mock_2x3_metadata_zs(),
            mock_2x3_metadata_zsAllActuators(),
            mock_1x3_metadata_zsAllActuators(),
            mock_3x1_metadata_zsAllActuators(),
        ]
        for metadata_fixture in param_list:
            expected_rois = metadata_fixture[2]
            with self.subTest(metadata_fixture=metadata_fixture):
                with patch(
                    to_replace, new=Mock(return_value=metadata_fixture[0])
                ):
                    metadata = ScanImageMetadata(tiff_path=tmp_path)
                self.assertEqual(metadata.n_rois, len(expected_rois))
                for i_roi in range(metadata.n_rois):
                    self.assertEqual(
                        metadata.zs_for_roi(i_roi), expected_rois[i_roi]["zs"]
                    )
                with self.assertRaises(ValueError) as cm:
                    metadata.zs_for_roi(metadata.n_rois)
                self.assertIn("You asked for ROI", str(cm.exception))

    def test_defined_rois(self):
        """
        Test that ScanImageMetadata.defined_rois returns the expected results
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="test_all_zs"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        param_list = [
            mock_2x3_metadata_zs(),
            mock_2x3_metadata_zsAllActuators(),
            mock_1x3_metadata_zsAllActuators(),
            mock_3x1_metadata_zsAllActuators(),
        ]
        for metadata_fixture in param_list:
            expected_rois = metadata_fixture[2]
            with self.subTest(metadata_fixture=metadata_fixture):
                with patch(
                    to_replace, new=Mock(return_value=metadata_fixture[0])
                ):
                    metadata = ScanImageMetadata(tiff_path=tmp_path)
                    self.assertEqual(metadata.defined_rois, expected_rois)

    def test_numVolumes_errors(self):
        """
        Test that errors are raised when
        SI.hStackManager.actualNumVolumes is not an int
        """
        import tempfile

        wrong_numVolumes_list = [2.1, (1, 2, 3), [1, 2, 3], "abcde"]
        for wrong_numVolumes in wrong_numVolumes_list:
            with self.subTest(wrong_numVolumes=wrong_numVolumes):
                tmpdir = pathlib.Path(
                    tempfile.mkdtemp(prefix="numVolumes_errors")
                )
                tiff_path = pathlib.Path(
                    mkstemp_clean(dir=tmpdir, suffix="tiff")
                )
                metadata = [
                    {"SI.hStackManager.actualNumVolumes": wrong_numVolumes}
                ]
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    obj = ScanImageMetadata(tiff_path=tiff_path)
                    with self.assertRaises(ValueError) as cm:
                        _ = obj.numVolumes
                    self.assertIn("expected int", str(cm.exception))

    def test_numSlices_errors(self):
        """
        Test that errors are raised when
        SI.hStackManager.actualNumSlices is not an int
        """
        import tempfile

        wrong_numSlices_list = [2.1, (1, 2, 3), [1, 2, 3], "abcde"]
        for wrong_numSlices in wrong_numSlices_list:
            with self.subTest(wrong_numSlices=wrong_numSlices):
                tmpdir = pathlib.Path(
                    tempfile.mkdtemp(prefix="numSlices_errors")
                )
                tiff_path = pathlib.Path(
                    mkstemp_clean(dir=tmpdir, suffix="tiff")
                )
                metadata = [
                    {"SI.hStackManager.actualNumSlices": wrong_numSlices}
                ]
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    obj = ScanImageMetadata(tiff_path=tiff_path)
                    with self.assertRaises(ValueError) as cm:
                        _ = obj.numSlices
                    self.assertIn("expected int", str(cm.exception))

    def test_roi_size(self):
        """
        Test that ScanImageMetadata.roi_size returns expected values
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="roi_size"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        roi_to_size = [(10.1, 2.3), (4.5, 6.1), (8.3, 2.5)]
        roi_list = [
            {"scanfields": {"sizeXY": roi_to_size[0]}},
            {
                "scanfields": [
                    {"sizeXY": roi_to_size[1]},
                    {"sizeXY": roi_to_size[1]},
                ]
            },
            {"scanfields": {"sizeXY": roi_to_size[2]}},
        ]
        metadata = ["nothing", dict()]
        metadata[1]["RoiGroups"] = {"imagingRoiGroup": {"rois": roi_list}}
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
        for ii in range(3):
            self.assertEqual(tiff_metadata.roi_size(ii), roi_to_size[ii])
        with self.assertRaises(ValueError) as cm:
            tiff_metadata.roi_size(3)
        self.assertIn("there are only 3", str(cm.exception))

    def test_roi_size_errors(self):
        """
        Test that ScanImageMetadata.roi_size returns errors when scanfields
        have inconsistent sizes
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="roi_size"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        roi_to_size = [(10.1, 2.3), (4.5, 6.1), (8.3, 2.5)]
        # test for error when an ROI has two conflicting values for scanfields:sizeXY
        roi_list = [
            {"scanfields": {"sizeXY": roi_to_size[0]}},
            {
                "scanfields": [
                    {"sizeXY": roi_to_size[1]},
                    {"sizeXY": roi_to_size[2]},
                ]
            },
            {"scanfields": {"sizeXY": roi_to_size[2]}},
        ]
        metadata = ["nothing", dict()]
        metadata[1]["RoiGroups"] = {"imagingRoiGroup": {"rois": roi_list}}
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
            with self.assertRaises(ValueError) as cm:
                tiff_metadata.roi_size(1)
            self.assertIn("differing sizeXY", str(cm.exception))
        # test for error when an ROI's scanfields are of the wrong datatype
        roi_list = [
            {"scanfields": {"sizeXY": roi_to_size[0]}},
            {"scanfields": "this is just a string, huh?"},
            {"scanfields": {"sizeXY": roi_to_size[2]}},
        ]
        metadata = ["nothing", dict()]
        metadata[1]["RoiGroups"] = {"imagingRoiGroup": {"rois": roi_list}}
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
            with self.assertRaises(RuntimeError) as cm:
                tiff_metadata.roi_size(1)
            self.assertIn("either a list or a dict", str(cm.exception))

    def test_roi_resolution(self):
        """
        Test that ScanImageMetadata.roi_resolution returns expected values
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="roi_resolution"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        roi_to_res = [(10, 2), (4, 6), (8, 2)]
        roi_list = [
            {"scanfields": {"pixelResolutionXY": roi_to_res[0]}},
            {
                "scanfields": [
                    {"pixelResolutionXY": roi_to_res[1]},
                    {"pixelResolutionXY": roi_to_res[1]},
                ]
            },
            {"scanfields": {"pixelResolutionXY": roi_to_res[2]}},
        ]
        metadata = ["nothing", dict()]
        metadata[1]["RoiGroups"] = {"imagingRoiGroup": {"rois": roi_list}}
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
        for ii in range(3):
            self.assertEqual(tiff_metadata.roi_resolution(ii), roi_to_res[ii])
        with self.assertRaises(ValueError) as cm:
            tiff_metadata.roi_size(3)
        self.assertIn("there are only 3", str(cm.exception))

    def test_roi_resolution_errors(self):
        """
        Test that ScanImageMetadata.roi_size returns errors when scanfields
        have inconsistent sizes
        """
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="roi_resolution"))
        tmp_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
        roi_to_res = [(10, 2), (4, 6), (8, 2)]
        # test for error when an ROI has two conflicting values for scanfields:pixelResolutionXY
        roi_list = [
            {"scanfields": {"pixelResolutionXY": roi_to_res[0]}},
            {
                "scanfields": [
                    {"pixelResolutionXY": roi_to_res[1]},
                    {"pixelResolutionXY": roi_to_res[2]},
                ]
            },
            {"scanfields": {"pixelResolutionXY": roi_to_res[2]}},
        ]
        metadata = ["nothing", dict()]
        metadata[1]["RoiGroups"] = {"imagingRoiGroup": {"rois": roi_list}}
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
            with self.assertRaises(ValueError) as cm:
                tiff_metadata.roi_resolution(1)
            self.assertIn("differing pixelResolutionXY", str(cm.exception))
        # test for error when an ROI's scanfields are of the wrong datatype
        roi_list = [
            {"scanfields": {"sizeXY": roi_to_res[0]}},
            {"scanfields": "this is just a string, huh?"},
            {"scanfields": {"sizeXY": roi_to_res[2]}},
        ]
        metadata = ["nothing", dict()]
        metadata[1]["RoiGroups"] = {"imagingRoiGroup": {"rois": roi_list}}
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
            with self.assertRaises(RuntimeError) as cm:
                tiff_metadata.roi_resolution(1)
            self.assertIn("either a list or a dict", str(cm.exception))

    def test_defined_rois_error_handling(self):
        """Test defined_rois property handles invalid data types."""
        tmp_path = pathlib.Path(mkstemp_clean(suffix=".tif"))

        # Test when roi_group is neither dict nor list
        metadata = ["nothing", dict()]
        metadata[1]["RoiGroups"] = {
            "imagingRoiGroup": {
                "rois": "invalid_string_type"  # Neither dict nor list
            }
        }

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
            with self.assertRaises(RuntimeError) as cm:
                _ = tiff_metadata.defined_rois
            self.assertIn("unable to parse", str(cm.exception))
            self.assertIn("of type <class 'str'>", str(cm.exception))

    def test_numVolumes_error_handling(self):
        """Test numVolumes property handles invalid data types."""
        tmp_path = pathlib.Path(mkstemp_clean(suffix=".tif"))

        # Test when actualNumVolumes is not an int
        metadata = ["nothing", dict()]
        metadata[0] = {"SI.hStackManager.actualNumVolumes": "not_an_int"}

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
            with self.assertRaises(ValueError) as cm:
                _ = tiff_metadata.numVolumes
            self.assertIn(
                "SI.hStackManager.actualNumVolumes is a", str(cm.exception)
            )
            self.assertIn("expected int", str(cm.exception))

    def test_numSlices_error_handling(self):
        """Test numSlices property handles invalid data types."""
        tmp_path = pathlib.Path(mkstemp_clean(suffix=".tif"))

        # Test when actualNumSlices is not an int
        metadata = ["nothing", dict()]
        metadata[0] = {
            "SI.hStackManager.actualNumSlices": 3.14  # float instead of int
        }

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
            with self.assertRaises(ValueError) as cm:
                _ = tiff_metadata.numSlices
            self.assertIn(
                "SI.hStackManager.actualNumSlices is a", str(cm.exception)
            )
            self.assertIn("expected int", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
