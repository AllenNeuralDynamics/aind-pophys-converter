"""test malformed zstack"""

import copy
import pathlib
import unittest
from typing import Dict, List
from unittest.mock import patch

import numpy as np

from aind_pophys_converter.tiff_splitters import (
    ZStackSplitter,
    mkstemp_clean,
)


class TestMalformedZStack(unittest.TestCase):
    """Test malformed zstack"""

    def setUp(self) -> None:
        """setup"""
        self.n_rois = 3
        self.baseline_zstack_metadata = self._make_baseline_zstack_metadata()

    def _make_baseline_zstack_metadata(self) -> List[dict]:
        """Baseline zstack metadata"""
        n_rois = 3
        main_roi_list = []
        for ii in range(n_rois):
            main_roi_list.append(
                {
                    "zs": -10,  # for z_stack, ;zs' does not matter
                    "discretePlaneMode": 1,
                }
            )
        main_roi_metadata = {
            "RoiGroups": {"imagingRoiGroup": {"rois": main_roi_list}}
        }
        z0_values = np.linspace(1.0, 3.0, 13)
        z1_values = np.linspace(3.0, 5.0, 13)
        zz = []
        for z0, z1 in zip(z0_values, z1_values):
            zz.append([z0, z1])
        return [
            {
                "SI.hStackManager.zsAllActuators": zz,
                "SI.hChannels.channelSave": [1, 2],
            },
            main_roi_metadata,
        ]

    def _z_stack_many_discrete_plane_zero(self) -> Dict[str, List[dict]]:
        """Test discrete plane"""
        import tempfile

        path_to_metadata = dict()
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="z_stack_many"))
        for ii in range(3):
            tmp_path = mkstemp_clean(dir=tmp_dir, suffix=".tiff")
            metadata = copy.deepcopy(self.baseline_zstack_metadata)
            rois = metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"]
            if ii == 0:
                rois[0]["discretePlaneMode"] = 0
                rois[1]["discretePlaneMode"] = 0
            else:
                rois[ii]["discretePlaneMode"] = 0
            # Key on the resolved path: the lookup below resolves too,
            # and on macOS /var is a symlink to /private/var.
            resolved = str(pathlib.Path(tmp_path).resolve())
            path_to_metadata[resolved] = metadata
        return path_to_metadata

    def test_many_discrete_planes(self) -> None:
        """
        Test that an error is raised if more than one ROI is marked
        with discretePlaneMode==0
        """
        z_stack_path_to_metadata = self._z_stack_many_discrete_plane_zero()

        def mock_read_metadata(tiff_path: pathlib.Path) -> List[dict]:
            """mock read metadataq"""
            str_path = str(tiff_path.resolve().absolute())
            return z_stack_path_to_metadata[str_path]

        tiff_path_list = [
            pathlib.Path(n) for n in z_stack_path_to_metadata.keys()
        ]
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        with patch(to_replace, new=mock_read_metadata):
            with self.assertRaises(RuntimeError) as cm:
                ZStackSplitter(tiff_path_list=tiff_path_list)
            self.assertIn("More than one ROI has discrete", str(cm.exception))

    def _z_stack_no_discrete_plane_zero(self) -> Dict[str, List[dict]]:
        """Tests no discrete plane"""
        import tempfile

        path_to_metadata = dict()
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="z_stack_none"))
        for ii in range(3):
            tmp_path = mkstemp_clean(dir=tmp_dir, suffix=".tiff")
            metadata = copy.deepcopy(self.baseline_zstack_metadata)
            rois = metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"]
            if ii != 1:
                rois[ii]["discretePlaneMode"] = 0
            # Key on the resolved path: the lookup below resolves too,
            # and on macOS /var is a symlink to /private/var.
            resolved = str(pathlib.Path(tmp_path).resolve())
            path_to_metadata[resolved] = metadata
        return path_to_metadata

    def test_no_discrete_planes(self) -> None:
        """
        Test that an error is raised if no ROI is marked
        with discretePlaneMode==0
        """
        z_stack_path_to_metadata = self._z_stack_no_discrete_plane_zero()

        def mock_read_metadata(tiff_path: pathlib.Path) -> List[dict]:
            """mock read metadata"""
            str_path = str(tiff_path.resolve().absolute())
            return z_stack_path_to_metadata[str_path]

        tiff_path_list = [
            pathlib.Path(n) for n in z_stack_path_to_metadata.keys()
        ]
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        with patch(to_replace, new=mock_read_metadata):
            with self.assertRaises(RuntimeError) as cm:
                ZStackSplitter(tiff_path_list=tiff_path_list)
            self.assertIn("Could not find discrete", str(cm.exception))

    def _z_stack_z_array_odd_shape(self) -> Dict[str, List[dict]]:
        """Test odd shape"""
        import tempfile

        path_to_metadata = dict()
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="z_stack_none"))
        for ii in range(3):
            tmp_path = mkstemp_clean(dir=tmp_dir, suffix=".tiff")
            metadata = copy.deepcopy(self.baseline_zstack_metadata)
            rois = metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"]
            rois[ii]["discretePlaneMode"] = 0
            if ii == 1:
                key_name = "SI.hStackManager.zsAllActuators"
                metadata[0].pop(key_name)
                metadata[0][key_name] = [[1, 2, 3], [1, 2, 3]]
            # Key on the resolved path: the lookup below resolves too,
            # and on macOS /var is a symlink to /private/var.
            resolved = str(pathlib.Path(tmp_path).resolve())
            path_to_metadata[resolved] = metadata
        return path_to_metadata

    def test_z_odd_shape(self) -> None:
        """
        Test that an error is raised if the z_array in the metadata
        is not (N, 2)
        """
        z_stack_path_to_metadata = self._z_stack_z_array_odd_shape()

        def mock_read_metadata(tiff_path: pathlib.Path) -> List[dict]:
            """test mock read metadata"""
            str_path = str(tiff_path.resolve().absolute())
            return z_stack_path_to_metadata[str_path]

        tiff_path_list = [
            pathlib.Path(n) for n in z_stack_path_to_metadata.keys()
        ]
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        with patch(to_replace, new=mock_read_metadata):
            with self.assertRaises(RuntimeError) as cm:
                ZStackSplitter(tiff_path_list=tiff_path_list)
            self.assertIn("has odd shape", str(cm.exception))

    def _z_stack_roi_missing_tiff(self) -> Dict[str, List[dict]]:
        """Test missing zstack roi"""
        import tempfile

        path_to_metadata = dict()
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="z_stack_many"))
        for ii in range(3):
            tmp_path = mkstemp_clean(dir=tmp_dir, suffix=".tiff")
            metadata = copy.deepcopy(self.baseline_zstack_metadata)
            rois = metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"]
            if ii == 0 or ii == 1:
                rois[0]["discretePlaneMode"] = 0
            else:
                rois[ii]["discretePlaneMode"] = 0
            # Key on the resolved path: the lookup below resolves too,
            # and on macOS /var is a symlink to /private/var.
            resolved = str(pathlib.Path(tmp_path).resolve())
            path_to_metadata[resolved] = metadata
        return path_to_metadata

    def test_roi_missing_tiff(self) -> None:
        """
        Test that an error is raised if an ROI does not have
        a corresponding z-stack TIFF with discretePlaneMode==0
        """
        z_stack_path_to_metadata = self._z_stack_roi_missing_tiff()

        def mock_read_metadata(tiff_path: pathlib.Path) -> List[dict]:
            """mock read metadata"""
            str_path = str(tiff_path.resolve().absolute())
            return z_stack_path_to_metadata[str_path]

        tiff_path_list = [
            pathlib.Path(n) for n in z_stack_path_to_metadata.keys()
        ]
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        with patch(to_replace, new=mock_read_metadata):
            with self.assertRaises(RuntimeError) as cm:
                ZStackSplitter(tiff_path_list=tiff_path_list)
            self.assertIn(
                "are represented in the local z-stack", str(cm.exception)
            )

    def _z_stack_bad_channelSave(self) -> Dict[str, List[dict]]:
        """test bad channel save"""
        import tempfile

        path_to_metadata = dict()
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="z_stack_many"))
        for ii in range(3):
            tmp_path = mkstemp_clean(dir=tmp_dir, suffix=".tiff")
            metadata = copy.deepcopy(self.baseline_zstack_metadata)
            if ii == 1:
                metadata[0]["SI.hChannels.channelSave"] = 1
            # Key on the resolved path: the lookup below resolves too,
            # and on macOS /var is a symlink to /private/var.
            resolved = str(pathlib.Path(tmp_path).resolve())
            path_to_metadata[resolved] = metadata
        return path_to_metadata

    def test_zstack_bad_channelSave(self) -> None:
        """
        Test that an error is raised if a zstack file
        does not have channelSave==[1, 2]
        """
        z_stack_path_to_metadata = self._z_stack_bad_channelSave()

        def mock_read_metadata(tiff_path: pathlib.Path) -> List[dict]:
            """mock read metadata"""
            str_path = str(tiff_path.resolve().absolute())
            return z_stack_path_to_metadata[str_path]

        tiff_path_list = [
            pathlib.Path(n) for n in z_stack_path_to_metadata.keys()
        ]
        to_replace = "aind_pophys_converter.utils.tiff_metadata._read_metadata"
        with patch(to_replace, new=mock_read_metadata):
            with self.assertRaises(RuntimeError) as cm:
                ZStackSplitter(tiff_path_list=tiff_path_list)
            self.assertIn("can only handle channelSave", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
