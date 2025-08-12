import pathlib
import unittest
from unittest.mock import Mock, patch

from aind_pophys_converter.mesoscope_splitter import (
    AvgImageTiffSplitter,
    mkstemp_clean,
)


class TestMalformedTiff(unittest.TestCase):
    def setUp(self):
        # Use a temp directory for each test
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_repeated_z_error(self):
        """
        Test that, if one ROI has the same z more than once,
        an error is raised upon metadata validation
        """
        tmp_path = pathlib.Path(
            mkstemp_clean(dir=self.tmp_dir, suffix=".tiff")
        )

        z_lineup = [[1, 2], [3, 4]]
        roi_list = [{"zs": [1, 1]}, {"zs": [3, 4]}]
        img_grp = {"rois": roi_list}
        roi_grp = {"imagingRoiGroup": img_grp}
        roi_metadata = {"RoiGroups": roi_grp}
        metadata = [
            {
                "SI.hStackManager.zsAllActuators": z_lineup,
                "SI.hChannels.channelSave": [1, 2],
            },
            roi_metadata,
        ]

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            with self.assertRaises(RuntimeError) as cm:
                AvgImageTiffSplitter(tiff_path=tmp_path)
            self.assertIn("has duplicate zs", str(cm.exception))

    def test_roi_order_error(self):
        """
        Test that, if the zs in the SI.hstackManager.zsAllActuators field
        are not listed in ROI order, an error is raised
        """
        tmp_path = pathlib.Path(
            mkstemp_clean(dir=self.tmp_dir, suffix=".tiff")
        )

        z_lineup = [[1, 2], [3, 4]]
        roi_list = [{"zs": [1, 3]}, {"zs": [2, 4]}]
        img_grp = {"rois": roi_list}
        roi_grp = {"imagingRoiGroup": img_grp}
        roi_metadata = {"RoiGroups": roi_grp}
        metadata = [
            {
                "SI.hStackManager.zsAllActuators": z_lineup,
                "SI.hChannels.channelSave": [1, 2],
            },
            roi_metadata,
        ]

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            with self.assertRaises(RuntimeError) as cm:
                AvgImageTiffSplitter(tiff_path=tmp_path)
            self.assertIn("not in correct order for ROIs", str(cm.exception))

    def test_uneven_z_per_roi(self):
        """
        Test that an error is raised if there are not the same number
        of zs per ROI
        """
        tmp_path = pathlib.Path(
            mkstemp_clean(dir=self.tmp_dir, suffix=".tiff")
        )

        z_lineup = [[1, 2, 3], [4, 5]]
        roi_list = [{"zs": [1, 2, 3]}, {"zs": [4, 5]}]
        img_grp = {"rois": roi_list}
        roi_grp = {"imagingRoiGroup": img_grp}
        roi_metadata = {"RoiGroups": roi_grp}
        metadata = [
            {
                "SI.hStackManager.zsAllActuators": z_lineup,
                "SI.hChannels.channelSave": [1, 2],
            },
            roi_metadata,
        ]

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            with self.assertRaises(RuntimeError) as cm:
                AvgImageTiffSplitter(tiff_path=tmp_path)
            self.assertIn("equal number of zs per ROI", str(cm.exception))

    def test_zs_not_list(self):
        """
        Test that an error is raised if zsAllActuators is not a list
        """
        tmp_path = pathlib.Path(
            mkstemp_clean(dir=self.tmp_dir, suffix=".tiff")
        )

        z_lineup = 5
        roi_list = [{"zs": [1, 2, 3]}, {"zs": [4, 5]}]
        img_grp = {"rois": roi_list}
        roi_grp = {"imagingRoiGroup": img_grp}
        roi_metadata = {"RoiGroups": roi_grp}
        metadata = [
            {
                "SI.hStackManager.zsAllActuators": z_lineup,
                "SI.hChannels.channelSave": [1, 2],
            },
            roi_metadata,
        ]

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            with self.assertRaises(RuntimeError) as cm:
                AvgImageTiffSplitter(tiff_path=tmp_path)
            self.assertIn("Unclear how to split", str(cm.exception))

    def test_zs_no_placeholder(self):
        """
        Test that error is raised if the placeholder zeros are missing
        when SI.hChannels.channelSave == 1
        """
        tmp_path = pathlib.Path(
            mkstemp_clean(dir=self.tmp_dir, suffix=".tiff")
        )

        z_lineup = [[5, 4], [6, 7], [8, 9]]
        roi_list = [{"zs": [1, 2, 3]}, {"zs": [4, 5]}]
        img_grp = {"rois": roi_list}
        roi_grp = {"imagingRoiGroup": img_grp}
        roi_metadata = {"RoiGroups": roi_grp}
        metadata = [
            {
                "SI.hStackManager.zsAllActuators": z_lineup,
                "SI.hChannels.channelSave": 1,
            },
            roi_metadata,
        ]

        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            with self.assertRaises(RuntimeError) as cm:
                AvgImageTiffSplitter(tiff_path=tmp_path)
            self.assertIn("channelSave==1", str(cm.exception))

    def test_illegal_channelSave(self):
        """
        Test that error is raised if channelSave is of an unexpected type
        """
        for channelSave in [2, 3, [1, 5], [1, 2, 3]]:
            tmp_path = pathlib.Path(
                mkstemp_clean(dir=self.tmp_dir, suffix=".tiff")
            )

            z_lineup = [[5, 4], [6, 7], [8, 9]]
            roi_list = [{"zs": [1, 2, 3]}, {"zs": [4, 5]}]
            img_grp = {"rois": roi_list}
            roi_grp = {"imagingRoiGroup": img_grp}
            roi_metadata = {"RoiGroups": roi_grp}
            metadata = [
                {
                    "SI.hStackManager.zsAllActuators": z_lineup,
                    "SI.hChannels.channelSave": channelSave,
                },
                roi_metadata,
            ]

            with patch(
                "tifffile.read_scanimage_metadata",
                new=Mock(return_value=metadata),
            ):
                with self.assertRaises(RuntimeError) as cm:
                    AvgImageTiffSplitter(tiff_path=tmp_path)
                self.assertIn("Expect channelSave == 1 or", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
