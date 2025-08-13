"""Test cases for the AvgImageTiffSplitter, TimeSeriesSplitter,
and ZStackSplitter classes.
"""

import copy
import json
import pathlib
import unittest
import warnings
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch

import h5py
import numpy as np
import tifffile
from aind_ophys_utils.array_utils import normalize_array

from aind_pophys_converter.mesoscope_splitter import (
    AvgImageTiffSplitter,
    TimeSeriesSplitter,
    ZStackSplitter,
    mkstemp_clean,
)


def _create_tiff_pages_and_lookup(z_value_list, n_rois):
    """Create TIFF pages and page lookup dictionary."""
    page_lookup = dict()
    tiff_pages = []
    n_pages = 5
    n_z_per_roi = len(z_value_list) // n_rois
    for i_page in range(n_pages):
        for i_z, z_value in enumerate(z_value_list):
            i_roi = i_z // n_z_per_roi
            if (i_roi, z_value) not in page_lookup:
                page_lookup[(i_roi, z_value)] = []
            value = i_roi + z_value + i_page * len(z_value_list)
            page = np.arange(value, value + 24 * 24).astype(np.uint16)
            page = page.reshape((24, 24))
            page[i_roi: i_roi + 4, i_roi: i_roi + 4] = 3
            page[i_z: i_z + 2, i_z: i_z + 2] = 0
            tiff_pages.append(page)
            page_lookup[(i_roi, z_value)].append(page)
    return tiff_pages, page_lookup


def _create_z_array(z_value_list, is_surface):
    """Create z array based on surface type."""
    if is_surface:
        return [[z_value, 0] for z_value in z_value_list]
    else:
        return [
            [z_value_list[ii], z_value_list[1 + ii]]
            for ii in range(0, len(z_value_list), 2)
        ]


def _create_metadata(z_array, use_zs, is_surface, z_value_list, n_rois):
    """Create metadata dictionary."""
    key_name = (
        "SI.hStackManager.zs" if use_zs else "SI.hStackManager.zsAllActuators"
    )  # noqa: E501
    metadata = [{key_name: z_array}]
    metadata[0]["SI.hChannels.channelSave"] = 1 if is_surface else [1, 2]
    n_z_per_roi = len(z_value_list) // n_rois
    roi_list = []
    for ii in range(0, len(z_value_list), n_z_per_roi):
        this_list = copy.deepcopy(list(z_value_list[ii: ii + n_z_per_roi]))
        this_list.sort()
        roi_list.append({"zs": this_list})
    roi_metadata = {"RoiGroups": {"imagingRoiGroup": {"rois": roi_list}}}
    metadata.append(roi_metadata)
    return metadata


def _create_image_tiff(
    tmp_dir: pathlib.Path,
    z_value_list: List[int],
    n_rois: int,
    use_zs: bool = False,
    is_surface: bool = False,
) -> Tuple[
    pathlib.Path,
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], List[np.ndarray]],
    dict,
]:
    """
    A fixture simulating a depth TIFF sampling 2 ROIs,
    4 zs at each ROI.

    if use_zs == True, then use SI.hStackManager.zs
    in place of SI.hStackManager.zsAllActuators

    if is_surface == True, SI.hStackManager.zsAllActuators
    will look like [[z0, 0], [z1, 0], [z2, 0]...]

    Returns
    -------
    the path to the TIFF

    a dict mapping roi_id, z -> raw average image

    a dict mapping roi_id, z -> normalized average image

    a dict mapping roi_id, z -> expected tiff pages

    a dict that our mock of tifffile.read_scanimage_metadata needs
    to return
    """
    tiff_pages, page_lookup = _create_tiff_pages_and_lookup(
        z_value_list, n_rois
    )  # noqa: E501
    tmp_path = pathlib.Path(mkstemp_clean(dir=tmp_dir, suffix="tiff"))
    tifffile.imwrite(tmp_path, tiff_pages)
    tiff_pages = np.array(tiff_pages)
    n_z_per_roi = len(z_value_list) // n_rois
    avg_img_lookup = dict()
    raw_avg_img_lookup = dict()
    for i_z, z_value in enumerate(z_value_list):
        sub_arr = tiff_pages[i_z:: len(z_value_list), :, :]
        mean_img = np.mean(sub_arr, axis=0)
        roi_id = i_z // n_z_per_roi
        raw_avg_img_lookup[(roi_id, z_value)] = mean_img
        avg_img_lookup[(roi_id, z_value)] = normalize_array(mean_img)
    z_array = _create_z_array(z_value_list, is_surface)
    metadata = _create_metadata(
        z_array, use_zs, is_surface, z_value_list, n_rois
    )  # noqa: E501
    return (
        tmp_path,
        raw_avg_img_lookup,
        avg_img_lookup,
        page_lookup,
        metadata,
    )


class TestTiffSplitter(unittest.TestCase):
    """Test cases for the AvgImageTiffSplitter, TimeSeriesSplitter,
    and ZStackSplitter classes.
    """

    def setUp(self):
        """Set up test fixtures and suppress ResourceWarnings"""
        # Suppress ResourceWarnings during testing to reduce noise
        warnings.filterwarnings("ignore", category=ResourceWarning)

    def tearDown(self):
        """Clean up after tests"""
        # Reset warnings to default behavior
        warnings.resetwarnings()

    def test_depth_splitter(self):
        """Test the AvgImageTiffSplitter with various parameters."""
        param_list = [
            (list(range(8)), 2, True),
            (list(range(8)), 4, True),
            (list(range(6)), 3, True),
            ((0, 2, 3, 5, 7, 4, 6, 11), 4, True),
            ((0, 2, 3, 5, 7, 4, 6, 11), 2, True),
            ((0, 2, 3, 5, 7, 4, 6, 5), 4, True),
            ((0, 2, 3, 5, 7, 4, 6, 5), 2, True),
            ((0, 4, 5, 2, 8, 7), 1, True),
            (list(range(8)), 2, False),
            (list(range(8)), 4, False),
            (list(range(6)), 3, False),
            ((0, 2, 3, 5, 7, 4, 6, 11), 4, False),
            ((0, 2, 3, 5, 7, 4, 6, 11), 2, False),
            ((0, 2, 3, 5, 7, 4, 6, 5), 4, False),
            ((0, 2, 3, 5, 7, 4, 6, 5), 2, False),
            ((0, 4, 5, 2, 8, 7), 1, False),
        ]
        import tempfile

        for z_value_list, n_rois, use_zs in param_list:
            with self.subTest(
                z_value_list=z_value_list, n_rois=n_rois, use_zs=use_zs
            ):
                tmp_dir = pathlib.Path(
                    tempfile.mkdtemp(prefix="test_depth_2x4")
                )
                depth_tiff = _create_image_tiff(
                    tmp_dir,
                    z_value_list,
                    n_rois,
                    use_zs=use_zs,
                    is_surface=False,
                )
                tiff_path = depth_tiff[0]
                raw_avg_img_lookup = depth_tiff[1]
                avg_img_lookup = depth_tiff[2]
                page_lookup = depth_tiff[3]
                metadata = depth_tiff[4]
                n_z_per_roi = len(z_value_list) // n_rois
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    splitter = AvgImageTiffSplitter(tiff_path=tiff_path)
                for i_z, z_value in enumerate(z_value_list):
                    i_roi = i_z // n_z_per_roi
                    arr = splitter._get_pages(i_roi=i_roi, z_value=z_value)
                    self.assertEqual(len(arr), 5)
                    for i_page in range(5):
                        expected = page_lookup[(i_roi, z_value)][i_page]
                        np.testing.assert_array_equal(expected, arr[i_page])
                    actual = splitter.get_avg_img(i_roi=i_roi, z_value=z_value)
                    np.testing.assert_allclose(
                        actual, raw_avg_img_lookup[(i_roi, z_value)]
                    )
                    tmp_path = mkstemp_clean(dir=tmp_dir, suffix=".tiff")
                    tmp_path = pathlib.Path(tmp_path)
                    splitter.write_output_file(
                        i_roi=i_roi, z_value=z_value, output_path=tmp_path
                    )
                    with tifffile.TiffFile(tmp_path, mode="rb") as tiff_file:
                        self.assertEqual(len(tiff_file.pages), 1)
                        actual = tiff_file.pages[0].asarray()
                        output_metadata = tiff_file.shaped_metadata[0][
                            "scanimage_metadata"
                        ]
                        self.assertEqual(output_metadata, metadata)
                    np.testing.assert_array_equal(
                        actual, avg_img_lookup[(i_roi, z_value)]
                    )
                    if tmp_path.is_file():
                        tmp_path.unlink()
                if tiff_path.is_file():
                    tiff_path.unlink()

    def test_splitter_manifest(self):
        """Test the AvgImageTiffSplitter manifest with various parameters."""
        param_list = [
            (list(range(8)), 2, True),
            (list(range(8)), 4, True),
            (list(range(6)), 3, True),
            ((0, 2, 3, 5, 7, 4, 6, 11), 4, True),
            ((0, 2, 3, 5, 7, 4, 6, 11), 2, True),
            ((0, 2, 3, 5, 7, 4, 6, 5), 4, True),
            ((0, 2, 3, 5, 7, 4, 6, 5), 2, True),
            ((0, 4, 5, 2, 8, 7), 1, True),
            (list(range(8)), 2, False),
            (list(range(8)), 4, False),
            (list(range(6)), 3, False),
            ((0, 2, 3, 5, 7, 4, 6, 11), 4, False),
            ((0, 2, 3, 5, 7, 4, 6, 11), 2, False),
            ((0, 2, 3, 5, 7, 4, 6, 5), 4, False),
            ((0, 2, 3, 5, 7, 4, 6, 5), 2, False),
            ((0, 4, 5, 2, 8, 7), 1, False),
        ]
        import tempfile

        for z_value_list, n_rois, use_zs in param_list:
            with self.subTest(
                z_value_list=z_value_list, n_rois=n_rois, use_zs=use_zs
            ):
                tmp_dir = pathlib.Path(
                    tempfile.mkdtemp(prefix="test_depth_2x4")
                )
                depth_tiff = _create_image_tiff(
                    tmp_dir,
                    z_value_list,
                    n_rois,
                    use_zs=use_zs,
                    is_surface=False,
                )
                tiff_path = depth_tiff[0]
                metadata = depth_tiff[4]
                n_z_per_roi = len(z_value_list) // n_rois
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    splitter = AvgImageTiffSplitter(tiff_path=tiff_path)
                self.assertEqual(splitter.n_pages, 5 * len(z_value_list))
                self.assertEqual(splitter.n_valid_zs, len(z_value_list))
                self.assertEqual(splitter.n_rois, n_rois)
                offset = 0
                for i_roi in range(n_rois):
                    for zz in z_value_list[offset: offset + n_z_per_roi]:
                        self.assertTrue(
                            splitter.is_z_valid_for_roi(
                                i_roi=i_roi, z_value=zz
                            )
                        )
                    offset += n_z_per_roi
                self.assertEqual(
                    len(splitter.roi_z_int_manifest), len(z_value_list)
                )
                for pair in splitter.roi_z_int_manifest:
                    i_roi = pair[0]
                    roi_grp = metadata[1]["RoiGroups"]["imagingRoiGroup"][
                        "rois"
                    ]
                    if isinstance(roi_grp, dict):
                        self.assertEqual(i_roi, 0)
                        roi_z_ints = splitter._int_from_z(
                            z_value=roi_grp["zs"]
                        )
                    else:
                        roi_z_ints = [
                            splitter._int_from_z(z_value=zz)
                            for zz in roi_grp[i_roi]["zs"]
                        ]
                    if isinstance(roi_z_ints, int):
                        self.assertEqual(pair[1], roi_z_ints)
                    else:
                        self.assertIn(pair[1], roi_z_ints)
                if tiff_path.is_file():
                    tiff_path.unlink()

    def test_surface_splitter(self):
        """Test the AvgImageTiffSplitter with surface parameters."""
        param_list = [
            (list(range(3)), True),
            (list(range(4)), True),
            (list(range(2)), True),
            ((0, 5, 4), True),
            (list(range(3)), False),
            (list(range(4)), False),
            (list(range(2)), False),
            ((0, 5, 4), False),
        ]
        import tempfile

        for z_value_list, use_zs in param_list:
            with self.subTest(z_value_list=z_value_list, use_zs=use_zs):
                n_rois = len(z_value_list)
                n_z_per_roi = 1
                tmp_dir = pathlib.Path(
                    tempfile.mkdtemp(prefix="test_depth_2x4")
                )
                surface_tiff = _create_image_tiff(
                    tmp_dir,
                    z_value_list,
                    n_rois,
                    use_zs=use_zs,
                    is_surface=True,
                )
                tiff_path = surface_tiff[0]
                raw_avg_img_lookup = surface_tiff[1]
                avg_img_lookup = surface_tiff[2]
                page_lookup = surface_tiff[3]
                metadata = surface_tiff[4]
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    splitter = AvgImageTiffSplitter(tiff_path=tiff_path)
                for i_z, z_value in enumerate(z_value_list):
                    i_roi = i_z // n_z_per_roi
                    arr = splitter._get_pages(i_roi=i_roi, z_value=z_value)
                    self.assertEqual(len(arr), 5)
                    for i_page in range(5):
                        expected = page_lookup[(i_roi, z_value)][i_page]
                        np.testing.assert_array_equal(expected, arr[i_page])
                    actual = splitter.get_avg_img(i_roi=i_roi, z_value=None)
                    np.testing.assert_allclose(
                        actual, raw_avg_img_lookup[(i_roi, z_value)]
                    )
                    tmp_path = mkstemp_clean(dir=tmp_dir, suffix=".tiff")
                    tmp_path = pathlib.Path(tmp_path)
                    splitter.write_output_file(
                        i_roi=i_roi, z_value=None, output_path=tmp_path
                    )
                    with tifffile.TiffFile(tmp_path, mode="rb") as tiff_file:
                        self.assertEqual(len(tiff_file.pages), 1)
                        actual = tiff_file.pages[0].asarray()
                    np.testing.assert_array_equal(
                        actual, avg_img_lookup[(i_roi, z_value)]
                    )
                    if tmp_path.is_file():
                        tmp_path.unlink()
                if tiff_path.is_file():
                    tiff_path.unlink()

    def test_time_splitter(self):
        """Test the AvgImageTiffSplitter with time series parameters."""
        param_list = [
            (list(range(8)), 2, True),
            (list(range(8)), 4, True),
            (list(range(6)), 3, True),
            ((0, 2, 3, 5, 7, 4, 6, 11), 4, True),
            ((0, 2, 3, 5, 7, 4, 6, 11), 2, True),
            ((0, 2, 3, 5, 7, 4, 6, 5), 4, True),
            ((0, 2, 3, 5, 7, 4, 6, 5), 2, True),
            ((0, 4, 5, 2, 8, 7), 1, True),
            (list(range(8)), 2, False),
            (list(range(8)), 4, False),
            (list(range(6)), 3, False),
            ((0, 2, 3, 5, 7, 4, 6, 11), 4, False),
            ((0, 2, 3, 5, 7, 4, 6, 11), 2, False),
            ((0, 2, 3, 5, 7, 4, 6, 5), 4, False),
            ((0, 2, 3, 5, 7, 4, 6, 5), 2, False),
            ((0, 4, 5, 2, 8, 7), 1, False),
        ]
        import tempfile

        for z_value_list, n_rois, use_zs in param_list:
            with self.subTest(
                z_value_list=z_value_list, n_rois=n_rois, use_zs=use_zs
            ):
                n_rois = len(z_value_list)
                n_z_per_roi = 1
                tmp_dir = pathlib.Path(
                    tempfile.mkdtemp(prefix="test_depth_2x4")
                )
                time_tiff = _create_image_tiff(
                    tmp_dir,
                    z_value_list,
                    n_rois,
                    use_zs=use_zs,
                    is_surface=True,
                )
                tiff_path = time_tiff[0]
                page_lookup = time_tiff[3]
                metadata = time_tiff[4]
                with patch(
                    "tifffile.read_scanimage_metadata",
                    new=Mock(return_value=metadata),
                ):
                    splitter = TimeSeriesSplitter(tiff_path=tiff_path)
                output_path_map = dict()
                for i_z, z_value in enumerate(z_value_list):
                    i_roi = i_z // n_z_per_roi
                    output_path = pathlib.Path(
                        mkstemp_clean(dir=tmp_dir, suffix=".h5")
                    )
                    output_path_map[(i_roi, z_value)] = output_path
                splitter.write_output_files(
                    output_path_map=output_path_map,
                    tmp_dir=tmp_dir,
                    dump_every=5,
                )
                for i_z, z_value in enumerate(z_value_list):
                    i_roi = i_z // n_z_per_roi
                    tmp_path = output_path_map[(i_roi, z_value)]
                    with h5py.File(tmp_path, "r") as in_file:
                        actual = in_file["data"][()]
                    expected = np.stack(page_lookup[(i_roi, z_value)])
                    np.testing.assert_array_equal(expected, actual)
                    if tmp_path.is_file():
                        tmp_path.unlink()
                if tiff_path.is_file():
                    tiff_path.unlink()

    def test_invalid_timeseries_output_map(self):
        """Test the TimeSeriesSplitter with an invalid output path map."""
        import tempfile

        z_value_list = list(range(8))
        n_rois = 2
        use_zs = True
        n_rois = len(z_value_list)
        n_z_per_roi = 1
        tmp_dir = pathlib.Path(
            tempfile.mkdtemp(prefix="test_invalid_timeseries")
        )
        time_tiff = _create_image_tiff(
            tmp_dir, z_value_list, n_rois, use_zs=use_zs, is_surface=True
        )
        tiff_path = time_tiff[0]
        metadata = time_tiff[4]
        with patch(
            "tifffile.read_scanimage_metadata", new=Mock(return_value=metadata)
        ):
            splitter = TimeSeriesSplitter(tiff_path=tiff_path)
        output_path_map = dict()
        for i_z, z_value in enumerate(z_value_list):
            i_roi = i_z // n_z_per_roi
            output_path = pathlib.Path(
                mkstemp_clean(dir=tmp_dir, suffix=".h5")
            )
            output_path_map[(i_roi, z_value)] = output_path
        # test i_roi that is too large
        output_path_map[(11, 2.0)] = tmp_dir / "junk.h5"
        with self.assertRaises(ValueError) as cm:
            splitter.write_output_files(output_path_map=output_path_map)
        self.assertIn("there are only", str(cm.exception))
        output_path_map.pop((11, 2.0))
        # test i_roi < 0
        z_value = z_value_list[1]
        output_path_map[(-2, z_value)] = output_path_map[(1, z_value)]
        with self.assertRaises(ValueError) as cm:
            splitter.write_output_files(output_path_map=output_path_map)
        self.assertIn("i_roi must be >= 0", str(cm.exception))
        output_path_map.pop((-2, z_value))
        # test z value that does not match i_roi
        output_path_map[(3, 11111.0)] = tmp_dir / "junk.h5"
        with self.assertRaises(ValueError) as cm:
            splitter.write_output_files(output_path_map=output_path_map)
        self.assertIn("is not a valid z value for ROI", str(cm.exception))
        output_path_map.pop((3, 11111.0))
        # test when the file suffix is not .h5
        z_value = z_value_list[0]
        output_path_map[(0, z_value)] = tmp_dir / "junk.json"
        with self.assertRaises(ValueError) as cm:
            splitter.write_output_files(output_path_map=output_path_map)
        self.assertIn("expected HDF5 output path", str(cm.exception))
        # test when you haven't specified a path for every ROI
        output_path_map.pop((0, z_value))
        with self.assertRaises(ValueError) as cm:
            splitter.write_output_files(output_path_map=output_path_map)
        self.assertIn("says it contains", str(cm.exception))


def _create_roi_metadata(n_rois):
    """Create the main ROI metadata structure."""
    main_roi_list = []
    for ii in range(n_rois):
        main_roi_list.append(
            {
                "zs": -10,  # for z_stack, ;zs' does not matter
                "discretePlaneMode": 1,
            }
        )
    return {"RoiGroups": {"imagingRoiGroup": {"rois": main_roi_list}}}


def _create_z_values(roi_to_z_mapping, i_roi):
    """Create z values for a specific ROI."""
    z0 = roi_to_z_mapping[i_roi][0]
    z0_values = np.linspace(z0 - 5.0, z0 + 5.0, 7)
    z1 = roi_to_z_mapping[i_roi][1]
    z1_values = np.linspace(z1 - 5.0, z1 + 5.0, 7)
    z_values = []
    for ii in range(len(z0_values)):
        z_values.append([z0_values[ii], z1_values[ii]])
    return z_values, z0_values


def _generate_tiff_pages(
    rng,
    i_roi,
    z0_values,
    roi_to_z_mapping,
    n_repeats,
    tiff_pages_lookup,
    tiff_path_lookup,
    str_path,
):
    """Generate TIFF pages for a specific ROI."""
    this_tiff = []
    for i_z in range(len(z0_values)):
        for i_repeat in range(n_repeats):
            for ii in (0, 1):
                this_z = roi_to_z_mapping[i_roi][ii]
                if (i_roi, this_z) not in tiff_pages_lookup:
                    tiff_pages_lookup[(i_roi, this_z)] = []
                    tiff_path_lookup[(i_roi, this_z)] = str_path
                page = rng.integers(0, 2**16 - 1, (24, 24)).astype(np.int16)
                page[i_roi: i_roi + 5, i_roi: i_roi + 5] = 0
                page[i_z: i_z + 2, i_z: i_z + 2] = 1000
                tiff_pages_lookup[(i_roi, this_z)].append(page)
                this_tiff.append(page)
    return this_tiff


def _process_single_roi(
    tmpdir,
    i_roi,
    roi_to_z_mapping,
    main_roi_metadata,
    z_key,
    rng,
    n_repeats,
    tiff_pages_lookup,
    z_stack_path_to_metadata,
    tiff_path_lookup,
):
    """Process a single ROI to create TIFF stack and metadata."""
    stack_path = pathlib.Path(mkstemp_clean(dir=tmpdir, suffix=".tiff"))
    roi_metadata = copy.deepcopy(main_roi_metadata)
    rois = roi_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
    rois[i_roi]["discretePlaneMode"] = 0

    z_values, z0_values = _create_z_values(roi_to_z_mapping, i_roi)

    metadata = [
        {z_key: z_values, "SI.hChannels.channelSave": [1, 2]},
        roi_metadata,
    ]
    str_path = str(stack_path.resolve().absolute())
    z_stack_path_to_metadata[str_path] = metadata

    this_tiff = _generate_tiff_pages(
        rng,
        i_roi,
        z0_values,
        roi_to_z_mapping,
        n_repeats,
        tiff_pages_lookup,
        tiff_path_lookup,
        str_path,
    )

    tifffile.imwrite(stack_path, this_tiff)


def _create_z_stack_tiffs(
    tmpdir: pathlib.Path,
    roi_to_z_mapping: List[List[int]],
    use_zs: bool = False,
) -> Tuple[Dict[str, dict], Dict[Tuple[int, int], List[np.ndarray]]]:
    """
    Returns
    -------
    dict mapping z_stack_path to metadata
    dict mapping (i_roi, z_value) to tiff_pages_lookup
    dict mapping (i_roi, z_value) to z_stack_path
    """
    rng = np.random.default_rng(662211)
    n_rois = len(roi_to_z_mapping)
    n_repeats = 5

    tiff_pages_lookup = dict()
    z_stack_path_to_metadata = dict()
    tiff_path_lookup = dict()

    z_key = (
        "SI.hStackManager.zs" if use_zs else "SI.hStackManager.zsAllActuators"
    )
    main_roi_metadata = _create_roi_metadata(n_rois)

    for i_roi in range(n_rois):
        _process_single_roi(
            tmpdir,
            i_roi,
            roi_to_z_mapping,
            main_roi_metadata,
            z_key,
            rng,
            n_repeats,
            tiff_pages_lookup,
            z_stack_path_to_metadata,
            tiff_path_lookup,
        )

    return (z_stack_path_to_metadata, tiff_pages_lookup, tiff_path_lookup)


def test_z_stack_splitter(self):
    """Test the ZStackSplitter with various parameters."""
    param_list = [
        (list(range(8)), True),
        (list(range(8)), True),
        (list(range(6)), True),
        ((0, 2, 3, 5, 7, 4, 6, 11), True),
        ((0, 2, 3, 5, 7, 4, 6, 11), True),
        ((0, 2, 3, 5, 7, 4, 6, 5), True),
        ((0, 2, 3, 5, 7, 4, 6, 5), True),
        ((0, 4, 5, 2, 8, 7), True),
        (list(range(8)), False),
        (list(range(8)), False),
        (list(range(6)), False),
        ((0, 2, 3, 5, 7, 4, 6, 11), False),
        ((0, 2, 3, 5, 7, 4, 6, 11), False),
        ((0, 2, 3, 5, 7, 4, 6, 5), False),
        ((0, 2, 3, 5, 7, 4, 6, 5), False),
        ((0, 4, 5, 2, 8, 7), False),
    ]
    import tempfile

    for z_value_list, use_zs in param_list:
        with self.subTest(z_value_list=z_value_list, use_zs=use_zs):
            tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="z_stack_test"))
            n_z_per_roi = 2  # a requirement of the z-stacks
            n_rois = len(z_value_list) // n_z_per_roi
            roi_to_z_mapping = []
            for i_roi in range(n_rois):
                i0 = i_roi * n_z_per_roi
                i1 = i0 + n_z_per_roi
                these_zs = z_value_list[i0:i1]
                roi_to_z_mapping.append(these_zs)
            dataset = _create_z_stack_tiffs(
                tmpdir=tmpdir,
                roi_to_z_mapping=roi_to_z_mapping,
                use_zs=use_zs,
            )
            z_stack_path_to_metadata = dataset[0]
            tiff_pages_lookup = dataset[1]
            tiff_path_lookup = dataset[2]

            def mock_read_metadata(tiff_path):
                """Mock function to return metadata for a TIFF path."""
                str_path = str(tiff_path.resolve().absolute())
                return z_stack_path_to_metadata[str_path]

            to_replace = (
                "aind_pophys_converter.utils.tiff_metadata._read_metadata"
            )
            with patch(to_replace, new=mock_read_metadata):
                z_stack_path_list = [
                    pathlib.Path(k) for k in z_stack_path_to_metadata
                ]
                splitter = ZStackSplitter(tiff_path_list=z_stack_path_list)
            for i_roi in range(n_rois):
                for z_value in roi_to_z_mapping[i_roi]:
                    expected_metadata = mock_read_metadata(
                        pathlib.Path(tiff_path_lookup[(i_roi, z_value)])
                    )
                    actual = splitter._get_pages(i_roi=i_roi, z_value=z_value)
                    expected = np.stack(tiff_pages_lookup[(i_roi, z_value)])
                    np.testing.assert_array_equal(actual, expected)
                    tmp_h5 = mkstemp_clean(dir=tmpdir, suffix=".h5")
                    tmp_h5 = pathlib.Path(tmp_h5)
                    splitter.write_output_file(
                        i_roi=i_roi, z_value=z_value, output_path=tmp_h5
                    )
                    with h5py.File(tmp_h5, "r") as in_file:
                        actual_metadata = json.loads(
                            in_file["scanimage_metadata"][()].decode("utf-8")
                        )
                        self.assertEqual(actual_metadata, expected_metadata)
                        actual = in_file["data"][()]
                    np.testing.assert_array_equal(actual, expected)
                    if tmp_h5.is_file():
                        tmp_h5.unlink()
            for z_stack_path in z_stack_path_list:
                if z_stack_path.is_file():
                    z_stack_path.unlink()


if __name__ == "__main__":
    unittest.main()
