"""Test cases for the timeseries utilities in the
aind_pophys_converter package.
These tests cover the functionality of dumping,
gathering, and splitting timeseries data."""

import json
import pathlib
import unittest
from itertools import product

import h5py
import numpy as np
import tifffile

from aind_pophys_converter.tiff_splitters import (
    DEBUG_N_FRAMES,
    _dump_timeseries_caches,
    _gather_timeseries_caches,
    _split_timeseries_tiff,
    split_timeseries_tiff,
)


class TestTimeseriesUtils(unittest.TestCase):
    """Test cases for the timeseries utilities in the
    aind_pophys_converter package."""

    def setUp(self) -> None:
        """Set up test fixtures and suppress ResourceWarnings."""
        rng = np.random.RandomState(1123)
        # int16 matches the dtype ScanImage raw movies carry through the
        # split into per-plane HDF5.
        self.timeseries_fixtures = [
            rng.randint(-200, 3000, (37, 12, 12)).astype(np.int16),
            rng.randint(-200, 3000, (37, 12, 12)).astype(np.int16),
            rng.randint(-200, 3000, (36, 12, 12)).astype(np.int16),
        ]

    def make_timeseries_tiff(self, tmpdir: pathlib.Path) -> pathlib.Path:
        """Create a TIFF file from the timeseries fixtures."""
        tiff_path = tmpdir / "eg_timeseries.tiff"
        all_data = []
        for ii in range(37):
            for offset in range(3):
                if ii < self.timeseries_fixtures[offset].shape[0]:
                    all_data.append(self.timeseries_fixtures[offset][ii, :, :])
        all_data = np.stack(all_data)
        tifffile.imwrite(tiff_path, all_data)
        return tiff_path

    def test_dump_timeseries_caches(self) -> None:
        """Test the _dump_timeseries_caches function with various
        chunk sizes.
        """
        chunk_sizes_list = [(5, 11, 9), (12, 30, 6), (4, 18, 22), (11, -1, 6)]
        import tempfile

        for chunk_sizes in chunk_sizes_list:
            with self.subTest(chunk_sizes=chunk_sizes):
                tmpdir_list = [
                    pathlib.Path(tempfile.mkdtemp(prefix="dump_1")),
                    pathlib.Path(tempfile.mkdtemp(prefix="dump_2")),
                    pathlib.Path(tempfile.mkdtemp(prefix="dump_3")),
                ]
                offset_to_valid_cache = dict()
                offset_to_cache = dict()
                offset_to_tmp_files = dict()
                offset_to_tmp_dir = dict()
                for offset in range(3):
                    cache = np.zeros(
                        (
                            100,
                            self.timeseries_fixtures[offset].shape[1],
                            self.timeseries_fixtures[offset].shape[2],
                        ),
                        dtype=self.timeseries_fixtures[offset].dtype,
                    )
                    if chunk_sizes[offset] > 0:
                        chunk = self.timeseries_fixtures[offset][
                            : chunk_sizes[offset], :, :
                        ]
                        cache[: chunk_sizes[offset], :, :] = chunk
                    offset_to_tmp_files[offset] = []
                    offset_to_tmp_dir[offset] = tmpdir_list[offset]
                    offset_to_valid_cache[offset] = chunk_sizes[offset]
                    offset_to_cache[offset] = cache
                _dump_timeseries_caches(
                    offset_to_cache=offset_to_cache,
                    offset_to_valid_cache=offset_to_valid_cache,
                    offset_to_tmp_files=offset_to_tmp_files,
                    offset_to_tmp_dir=offset_to_tmp_dir,
                )
                for offset in range(3):
                    self.assertEqual(offset_to_valid_cache[offset], -1)
                    if chunk_sizes[offset] > 0:
                        self.assertEqual(len(offset_to_tmp_files[offset]), 1)
                        filepath = offset_to_tmp_files[offset][0]
                        with h5py.File(filepath, "r") as in_file:
                            expected = in_file["data"][()]
                        actual = self.timeseries_fixtures[offset][
                            : chunk_sizes[offset], :, :
                        ]
                        np.testing.assert_allclose(expected, actual)
                    else:
                        self.assertEqual(len(offset_to_tmp_files[offset]), 0)

    def test_gather_timeseries_caches(self) -> None:
        """Test the _gather_timeseries_caches function with various
        metadata parameters."""
        param_list = [{"a": 1, "b": 2}, None]
        import tempfile

        for expected_metadata in param_list:
            with self.subTest(expected_metadata=expected_metadata):
                tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="gather"))
                finaldir = pathlib.Path(
                    tempfile.mkdtemp(prefix="gather_output")
                )
                rng = np.random.default_rng(581321)
                data = rng.integers(-200, 3000, (29, 13, 13), dtype=np.int16)
                file_path_list = []
                for fname, i0, i1 in zip(
                    ("cache0.h5", "cache1.h5", "cache2.h5"),
                    (0, 13, 22),
                    (13, 22, 29),
                ):
                    file_path = tmpdir / fname
                    file_path_list.append(file_path)
                    with h5py.File(file_path, "w") as out_file:
                        out_file.create_dataset("data", data=data[i0:i1, :, :])
                full_path = finaldir / "full.h5"
                _gather_timeseries_caches(
                    file_path_list=file_path_list,
                    final_output_path=full_path,
                    metadata=expected_metadata,
                )
                with h5py.File(full_path, "r") as in_file:
                    if expected_metadata is None:
                        self.assertNotIn("scanimage_metadata", in_file.keys())
                    else:
                        actual_metadata = json.loads(
                            in_file["scanimage_metadata"][()].decode("utf-8")
                        )
                        self.assertEqual(actual_metadata, expected_metadata)
                    actual = in_file["data"][()]
                np.testing.assert_allclose(data, actual)
                filename_list = [f for f in tmpdir.iterdir()]
                self.assertEqual(len(filename_list), 0)

    def test_split_timeseries_tiff_worker(self) -> None:
        """Test the _split_timeseries_tiff function with various
        parameters including metadata and temporary directories."""
        param_list = list(
            product((50, 17, 8), (True, False), (None, {"a": 1, "b": 2}))
        )
        import tempfile

        for dump_every, same_tmpdir, expected_metadata in param_list:
            with self.subTest(
                dump_every=dump_every,
                same_tmpdir=same_tmpdir,
                expected_metadata=expected_metadata,
            ):
                offset_to_tmp_dir = dict()
                if same_tmpdir:
                    tmpdir = pathlib.Path(
                        tempfile.mkdtemp(prefix="split_caches")
                    )
                    for offset in range(3):
                        offset_to_tmp_dir[offset] = tmpdir
                else:
                    for offset in range(3):
                        tmpdir = pathlib.Path(
                            tempfile.mkdtemp(prefix=f"split_caches{offset}")
                        )
                        offset_to_tmp_dir[offset] = tmpdir
                final_dir = pathlib.Path(
                    tempfile.mkdtemp(prefix="final_timeseries")
                )
                offset_to_tmp_files = dict()
                offset_to_path = dict()
                for offset in range(3):
                    offset_to_tmp_files[offset] = []
                    offset_to_path[offset] = final_dir / f"final{offset}.h5"
                tiff_path = self.make_timeseries_tiff(final_dir)
                _split_timeseries_tiff(
                    tiff_path=tiff_path,
                    offset_to_path=offset_to_path,
                    offset_to_tmp_files=offset_to_tmp_files,
                    offset_to_tmp_dir=offset_to_tmp_dir,
                    dump_every=dump_every,
                    metadata=expected_metadata,
                )
                for offset in range(3):
                    expected = self.timeseries_fixtures[offset]
                    with h5py.File(offset_to_path[offset], "r") as in_file:
                        if expected_metadata is None:
                            self.assertNotIn(
                                "scanimage_metadata", in_file.keys()
                            )
                        else:
                            actual_metadata = json.loads(
                                in_file["scanimage_metadata"][()].decode(
                                    "utf-8"
                                )
                            )
                            self.assertEqual(
                                actual_metadata, expected_metadata
                            )
                        actual = in_file["data"][()]
                    np.testing.assert_allclose(expected, actual)

    def test_split_timeseries_tiff(self) -> None:
        """Test the split_timeseries_tiff function with various
        parameters including metadata and temporary directories."""
        param_list = list(product((50, 17, 8), (None, {"a": 1, "b": 2})))
        import tempfile

        for dump_every, expected_metadata in param_list:
            with self.subTest(
                dump_every=dump_every, expected_metadata=expected_metadata
            ):
                tmpdir = pathlib.Path(
                    tempfile.mkdtemp(prefix="split_tiff_parent")
                )
                final_dir = pathlib.Path(
                    tempfile.mkdtemp(prefix="final_timeseries")
                )
                offset_to_tmp_files = dict()
                offset_to_path = dict()
                for offset in range(3):
                    offset_to_tmp_files[offset] = []
                    offset_to_path[offset] = final_dir / f"final{offset}.h5"
                tiff_path = self.make_timeseries_tiff(final_dir)
                split_timeseries_tiff(
                    tiff_path=tiff_path,
                    offset_to_path=offset_to_path,
                    tmp_dir=tmpdir,
                    dump_every=dump_every,
                    metadata=expected_metadata,
                )
                for offset in range(3):
                    expected = self.timeseries_fixtures[offset]
                    with h5py.File(offset_to_path[offset], "r") as in_file:
                        if expected_metadata is None:
                            self.assertNotIn(
                                "scanimage_metadata", in_file.keys()
                            )
                        else:
                            actual_metadata = json.loads(
                                in_file["scanimage_metadata"][()].decode(
                                    "utf-8"
                                )
                            )
                            self.assertEqual(
                                actual_metadata, expected_metadata
                            )
                        actual = in_file["data"][()]
                    np.testing.assert_allclose(expected, actual)
                cache_paths = [f for f in tmpdir.iterdir()]
                self.assertEqual(len(cache_paths), 0)

    def test_split_timeseries_tiff_debug_cap_is_logged(self) -> None:
        """debug=True caps the movie and says so, naming the frame count."""
        import logging
        import tempfile

        n_extra_pages = 5
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="split_tiff_debug"))
        tiff_path = tmpdir / "debug_timeseries.tiff"
        tifffile.imwrite(
            tiff_path,
            np.zeros((DEBUG_N_FRAMES + n_extra_pages, 12, 12), dtype=np.int16),
        )
        offset_to_path = {0: tmpdir / "final0.h5"}

        with self.assertLogs(level=logging.INFO) as captured:
            split_timeseries_tiff(
                tiff_path=tiff_path,
                offset_to_path=offset_to_path,
                tmp_dir=tmpdir,
                dump_every=10,
                debug=True,
            )

        messages = "\n".join(captured.output)
        self.assertIn("Debug frame cap active", messages)
        self.assertIn("Debug frame cap reached", messages)
        self.assertIn(str(DEBUG_N_FRAMES), messages)
        with h5py.File(offset_to_path[0], "r") as in_file:
            self.assertEqual(in_file["data"].shape[0], DEBUG_N_FRAMES)


if __name__ == "__main__":
    unittest.main()
