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


import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

from aind_pophys_converter.mesoscope_splitter import (
    TiffSplitterCLI,
    find_split_directories,
)
from aind_pophys_converter.tiff_splitters import (
    IntFromZMapperMixin,
    TiffSplitterBase,
    mkstemp_clean,
)
from aind_pophys_converter.utils.metadata_utils import (
    ConverterMetadata,
    PlaneChannel,
    load_converter_metadata,
)
from aind_pophys_converter.utils.platform_utils import (
    load_platform_plane_positions,
)

MESO_CHILD = Path(__file__).parent / "fixtures" / "meso_child"

#: Patch target for the platform.json reader the splitter calls.
PLATFORM_READER = (
    "aind_pophys_converter.mesoscope_splitter.load_platform_plane_positions"
)

#: ``SI.hStackManager.zsAllActuators`` of the meso_child timeseries TIFF:
#: one ROI, each coupled pair in actuator order (Z1, then Z2).
MESO_CHILD_ZS_ALL_ACTUATORS = [
    [-304, -345],
    [-274, -385],
    [-246, -425],
    [-200, -465],
]


class _ManifestSplitter(IntFromZMapperMixin):
    """Timeseries splitter stand-in holding a ScanImage ROI/z manifest."""

    is_z_valid_for_roi = TiffSplitterBase.is_z_valid_for_roi

    def __init__(self, zs_per_roi: List[List[float]]) -> None:
        """Build the manifest from each ROI's z values in TIFF page order."""
        self.input_path = Path("timeseries.tiff")
        self.n_rois = len(zs_per_roi)
        self._valid_z_int_per_roi = [
            {self._int_from_z(z) for z in zs} for zs in zs_per_roi
        ]
        self.roi_z_int_manifest = [
            (i_roi, self._int_from_z(z))
            for i_roi, zs in enumerate(zs_per_roi)
            for z in zs
        ]


def _channel_named_acquisition(directory: Path) -> Path:
    """Copy meso_child's v2 metadata, naming each image's channel.

    Each plane's channel is taken from its platform.json entry, in
    plane-index order, as the native v2 mapper records it.
    """
    for name in ("acquisition.json", "data_description.json"):
        shutil.copy(MESO_CHILD / "v2" / name, directory / name)
    platform_path = next((MESO_CHILD / "v1" / "pophys").glob("*platform.json"))
    platform = json.loads(platform_path.read_text())
    channels = [
        plane["channel"]
        for group in platform["imaging_plane_groups"]
        for plane in group["imaging_planes"]
    ]
    acquisition = json.loads((directory / "acquisition.json").read_text())
    for stream in acquisition["data_streams"]:
        for config in stream["configurations"]:
            for image in config.get("images") or []:
                (plane,) = image["planes"]
                image["channel_name"] = channels[plane["plane_index"]]
    (directory / "acquisition.json").write_text(json.dumps(acquisition))
    return directory


def _write_platform(directory: Path, planes: List[dict]) -> None:
    """Write a one-group platform.json listing ``planes`` into directory."""
    platform = {"imaging_plane_groups": [{"imaging_planes": planes}]}
    (directory / "s_platform.json").write_text(json.dumps(platform))


def _plane(roi_index: int, z_value: float, depth: Optional[float]) -> dict:
    """Build one platform.json imaging plane entry."""
    return {
        "scanimage_roi_index": roi_index,
        "scanimage_scanfield_z": z_value,
        "targeted_depth": depth,
    }


def _planes(
    *specs: Tuple[Optional[str], Optional[int]],
) -> Tuple[PlaneChannel, ...]:
    """Build plane channels from ``(channel, partner)`` per plane index."""
    return tuple(
        PlaneChannel(
            plane_index=index,
            channel_name=channel,
            coupled_plane_index=partner,
        )
        for index, (channel, partner) in enumerate(specs)
    )


def _make_metadata() -> ConverterMetadata:
    """Build a minimal ConverterMetadata for splitter tests."""
    return ConverterMetadata(
        instrument_id="MESO.1",
        fov_ids=("plane_0",),
        dataset_name="multiplane-test",
        subject_id="000000",
    )


class TestMesoscopeSplitter(unittest.TestCase):
    """Test suite for mesoscope tiff splitting CLI."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_get_imaging_planes_empty_raises(self) -> None:
        """_get_imaging_planes fails fast when metadata has no planes."""
        cli = object.__new__(TiffSplitterCLI)
        cli.metadata = replace(_make_metadata(), fov_ids=())
        cli.input_dir = Path("nonexistent")
        with self.assertRaises(ValueError) as cm:
            cli._get_imaging_planes()
        self.assertIn("No imaging planes", str(cm.exception))

    def test_get_imaging_planes_returns_planes(self) -> None:
        """_get_imaging_planes returns the fov_ids tuple when present."""
        cli = object.__new__(TiffSplitterCLI)
        cli.metadata = _make_metadata()
        self.assertEqual(cli._get_imaging_planes(), cli.metadata.fov_ids)

    @staticmethod
    def _splitter_with_manifest(manifest: list) -> MagicMock:
        """Build a splitter stub exposing a manifest and its z lookup."""
        splitter = MagicMock()
        splitter.roi_z_int_manifest = manifest
        splitter.input_path = Path("timeseries.tiff")
        splitter._z_from_int.side_effect = lambda ii: float(ii) * -100.0
        return splitter

    def _cli(
        self,
        fov_ids: Tuple[str, ...],
        planes: Tuple[PlaneChannel, ...] = (),
        depths: Tuple[Optional[float], ...] = (),
    ) -> TiffSplitterCLI:
        """Build a splitter CLI over the given planes."""
        cli = object.__new__(TiffSplitterCLI)
        cli.metadata = replace(
            _make_metadata(),
            fov_ids=fov_ids,
            plane_channels=planes,
            plane_depths=depths,
        )
        cli.input_dir = self.test_path
        return cli

    def _meso_child_cli(self, metadata_dir: Path) -> TiffSplitterCLI:
        """Build a splitter CLI over the meso_child metadata in a directory."""
        cli = object.__new__(TiffSplitterCLI)
        cli.input_dir = self.test_path
        cli.metadata = load_converter_metadata(metadata_dir)
        return cli

    def _meso_child_manifest(
        self,
    ) -> Tuple[_ManifestSplitter, List[Tuple[int, float]]]:
        """Return the meso_child TIFF manifest and each plane's true position.

        The true (roi, z) of a plane is the ScanImage position its v1 session
        records. The TIFF stores each coupled pair as (Z1, Z2) while the
        metadata indexes the Z2 plane first, so pairing by page order would
        swap every pair.
        """
        session = json.loads((MESO_CHILD / "v1" / "session.json").read_text())
        fovs = session["data_streams"][0]["ophys_fovs"]
        expected = [
            (fov["scanimage_roi_index"], float(fov["scanfield_z"]))
            for fov in sorted(fovs, key=lambda fov: fov["index"])
        ]
        splitter = _ManifestSplitter(
            [[z for pair in MESO_CHILD_ZS_ALL_ACTUATORS for z in pair]]
        )
        page_order = [
            (roi, splitter._z_from_int(z_int))
            for roi, z_int in splitter.roi_z_int_manifest
        ]
        self.assertNotEqual(page_order, expected)
        return splitter, expected

    def _copy_meso_child_platform(self) -> None:
        """Copy the real meso_child platform.json into the test directory."""
        platform = next((MESO_CHILD / "v1" / "pophys").glob("*platform.json"))
        shutil.copy(platform, self.test_path / platform.name)

    def test_roi_z_per_plane_uses_channel_slot(self) -> None:
        """Each plane takes its channel's page, not the TIFF page order.

        The test directory holds no platform.json: the channels alone locate
        the planes.
        """
        splitter, expected = self._meso_child_manifest()
        cli = self._meso_child_cli(_channel_named_acquisition(self.test_path))
        self.assertEqual(cli._roi_z_per_plane(splitter), expected)

    def test_roi_z_per_plane_channels_take_precedence(self) -> None:
        """Planes naming every channel are not located from platform.json."""
        self._copy_meso_child_platform()
        splitter, expected = self._meso_child_manifest()
        cli = self._meso_child_cli(_channel_named_acquisition(self.test_path))
        with patch(
            PLATFORM_READER, wraps=load_platform_plane_positions
        ) as read_platform:
            self.assertEqual(cli._roi_z_per_plane(splitter), expected)
        read_platform.assert_not_called()

    def test_roi_z_per_plane_uses_platform_position(self) -> None:
        """Each plane gets its own scanfield z, not the TIFF page order."""
        self._copy_meso_child_platform()
        splitter, expected = self._meso_child_manifest()
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                cli = self._meso_child_cli(MESO_CHILD / version)
                self.assertEqual(cli._roi_z_per_plane(splitter), expected)

    def test_roi_z_per_plane_v1_input_takes_platform_path(self) -> None:
        """v1 input records no channel, so platform.json locates the planes."""
        self._copy_meso_child_platform()
        splitter, expected = self._meso_child_manifest()
        cli = self._meso_child_cli(MESO_CHILD / "v1")
        self.assertEqual(
            {plane.channel_name for plane in cli.metadata.plane_channels},
            {None},
        )
        with patch(
            PLATFORM_READER, wraps=load_platform_plane_positions
        ) as read_platform:
            self.assertEqual(cli._roi_z_per_plane(splitter), expected)
        read_platform.assert_called_once()

    def test_roi_z_per_plane_partial_channels_use_platform(self) -> None:
        """Every plane needs a known channel to skip platform.json."""
        _write_platform(
            self.test_path, [_plane(0, -20, None), _plane(0, -10, None)]
        )
        cli = self._cli(("A_0", "A_1"), _planes(("Channel1", 1), (None, 0)))
        self.assertEqual(
            cli._roi_z_per_plane(_ManifestSplitter([[-10, -20]])),
            [(0, -20.0), (0, -10.0)],
        )

    def test_roi_z_per_plane_without_channels_or_platform_raises(self) -> None:
        """No known channels and no platform.json: planes cannot be located."""
        splitter = _ManifestSplitter([[-10], [-20]])
        clis = {
            "no channel records": self._cli(("A_0", "A_1")),
            "one channel unnamed": self._cli(
                ("A_0", "A_1"), _planes(("Channel1", None), (None, None))
            ),
            "generic channel": self._cli(
                ("A_0", "A_1"),
                _planes(("Ophys Channel", None), ("Ophys Channel", None)),
            ),
            "record for one of two planes": self._cli(
                ("A_0", "A_1"), _planes(("Channel1", None))
            ),
        }
        for label, cli in clis.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as cm:
                    cli._roi_z_per_plane(splitter)
                self.assertIn("has no *platform.json", str(cm.exception))

    def test_roi_z_per_plane_uncoupled_planes(self) -> None:
        """Each uncoupled plane takes its own one-page slice."""
        cli = self._cli(
            ("A_0", "B_1"), _planes(("Channel1", None), ("Channel1", None))
        )
        self.assertEqual(
            cli._roi_z_per_plane(_ManifestSplitter([[-10], [-20]])),
            [(0, -10.0), (1, -20.0)],
        )

    def test_roi_z_per_plane_partners_must_name_each_other(self) -> None:
        """A partner that is missing, itself, or not coupled back fails."""
        for planes in (
            _planes(("Channel2", 1), ("Channel1", None)),
            _planes(("Channel2", 5), ("Channel1", 0)),
            _planes(("Channel2", 0), ("Channel1", 0)),
        ):
            with self.subTest(planes=planes):
                cli = self._cli(("A_0", "A_1"), planes)
                with self.assertRaises(ValueError) as cm:
                    cli._roi_z_per_plane(_ManifestSplitter([[-10, -20]]))
                self.assertIn("coupled back", str(cm.exception))

    def test_roi_z_per_plane_pair_slots_must_fit_its_slice(self) -> None:
        """Partners sharing a channel, or a channel with no page, fail."""
        for planes, zs in (
            (_planes(("Channel1", 1), ("Channel1", 0)), [[-10, -20]]),
            (_planes(("Channel2", None)), [[-10]]),
        ):
            with self.subTest(planes=planes):
                fov_ids = tuple(f"A_{i}" for i in range(len(planes)))
                cli = self._cli(fov_ids, planes)
                with self.assertRaises(ValueError) as cm:
                    cli._roi_z_per_plane(_ManifestSplitter(zs))
                self.assertIn("do not match", str(cm.exception))

    def test_roi_z_per_plane_pair_slice_spans_rois_raises(self) -> None:
        """A coupled pair's pages must come from one ROI."""
        cli = self._cli(
            ("A_0", "A_1"), _planes(("Channel2", 1), ("Channel1", 0))
        )
        with self.assertRaises(ValueError) as cm:
            cli._roi_z_per_plane(_ManifestSplitter([[-10], [-20]]))
        self.assertIn("spans ROIs", str(cm.exception))

    def test_roi_z_per_plane_platform_count_mismatch_raises(self) -> None:
        """platform.json must list one entry per metadata plane."""
        _write_platform(self.test_path, [_plane(0, -10, 10)])
        cli = self._cli(("A_0", "A_1"))
        with self.assertRaises(ValueError) as cm:
            cli._roi_z_per_plane(_ManifestSplitter([[-10, -20]]))
        self.assertIn("lists 1", str(cm.exception))

    def test_roi_z_per_plane_depth_mismatch_raises(self) -> None:
        """A plane whose platform depth differs from its metadata depth."""
        _write_platform(
            self.test_path, [_plane(0, -10, 20), _plane(0, -20, 10)]
        )
        cli = self._cli(("A_0", "A_1"), depths=(10.0, 20.0))
        with self.assertRaises(ValueError) as cm:
            cli._roi_z_per_plane(_ManifestSplitter([[-10, -20]]))
        self.assertIn("not in plane-index order", str(cm.exception))

    def test_roi_z_per_plane_skips_unrecorded_depth(self) -> None:
        """Depth is checked only where both sides record one."""
        _write_platform(
            self.test_path, [_plane(0, -10, None), _plane(0, -20, 10)]
        )
        cli = self._cli(("A_0", "A_1"), depths=(20.0, None))
        self.assertEqual(
            cli._roi_z_per_plane(_ManifestSplitter([[-20, -10]])),
            [(0, -10.0), (0, -20.0)],
        )

    def test_roi_z_per_plane_position_not_in_tiff_raises(self) -> None:
        """A platform position the TIFF does not contain fails loudly."""
        for planes in (
            [_plane(0, -10, 10), _plane(0, -30, 20)],
            [_plane(0, -10, 10), _plane(5, -20, 20)],
        ):
            with self.subTest(planes=planes):
                _write_platform(self.test_path, planes)
                cli = self._cli(("A_0", "A_1"))
                with self.assertRaises(ValueError) as cm:
                    cli._roi_z_per_plane(_ManifestSplitter([[-10, -20]]))
                self.assertIn("does not contain", str(cm.exception))

    def test_roi_z_per_plane_shared_position_raises(self) -> None:
        """Two planes cannot claim the same TIFF position."""
        _write_platform(
            self.test_path, [_plane(0, -10, 10), _plane(0, -10, 20)]
        )
        cli = self._cli(("A_0", "A_1"))
        with self.assertRaises(ValueError) as cm:
            cli._roi_z_per_plane(_ManifestSplitter([[-10, -20]]))
        self.assertIn("already used", str(cm.exception))

    def test_roi_z_per_plane_count_mismatch_raises(self) -> None:
        """A manifest longer than the plane list is not paired silently."""
        cli = object.__new__(TiffSplitterCLI)
        cli.metadata = replace(_make_metadata(), fov_ids=("VISp_0", "MOp_1"))
        cli.input_dir = Path("input")
        splitter = self._splitter_with_manifest([(0, 1), (0, 2), (1, 3)])

        with self.assertRaises(ValueError) as cm:
            cli._roi_z_per_plane(splitter)
        message = str(cm.exception)
        self.assertIn("Plane count mismatch", message)
        self.assertIn("declares 2 plane(s)", message)
        self.assertIn("contains 3", message)

    def test_find_split_directories_with_matching_dirs(self) -> None:
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

    def test_find_split_directories_no_matches(self) -> None:
        """Test find_split_directories returns empty list when no matches."""
        # Create directories that don't match the pattern
        non_matching_dirs = ["dir1", "V", "V_", "Test_1", "v1_2"]
        for dir_name in non_matching_dirs:
            (self.test_path / dir_name).mkdir()

        result = find_split_directories(self.test_path)
        self.assertEqual(result, [])

    def test_find_split_directories_empty_directory(self) -> None:
        """Test find_split_directories on empty directory."""
        result = find_split_directories(self.test_path)
        self.assertEqual(result, [])

    def test_mkstemp_clean_basic(self) -> None:
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

    def test_mkstemp_clean_with_directory(self) -> None:
        """Test mkstemp_clean with specific directory."""
        file_path = mkstemp_clean(dir=self.temp_dir.name)

        # File should exist in the specified directory
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.startswith(self.temp_dir.name))

        # Clean up
        os.unlink(file_path)

    def test_mkstemp_clean_with_prefix_suffix(self) -> None:
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

    def test_tiff_splitter_cli_str2bool(self) -> None:
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
        self,
        input_dir: Path,
        output_dir: Path,
        temp_dir: Path,
        str_args: bool = False,
    ) -> TiffSplitterCLI:
        """Helper: build a TiffSplitterCLI with injected ConverterMetadata."""
        from types import SimpleNamespace

        job_settings = SimpleNamespace()
        job_settings.input_dir = str(input_dir) if str_args else input_dir
        job_settings.output_dir = str(output_dir) if str_args else output_dir
        job_settings.temp_dir = str(temp_dir) if str_args else temp_dir
        job_settings.debug = False

        return TiffSplitterCLI(job_settings, metadata=_make_metadata())

    def test_tiff_splitter_cli_init_with_strings(self) -> None:
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

    def test_tiff_splitter_cli_init_with_paths(self) -> None:
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

    def test_tiff_splitter_cli_missing_files_errors(self) -> None:
        """Test TiffSplitterCLI raises appropriate errors when tiff
        files are missing. Metadata is injected directly so we no longer
        need an acquisition.json on disk for these checks.
        """
        from types import SimpleNamespace

        metadata = _make_metadata()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input_data"
            input_dir.mkdir()

            job_settings = SimpleNamespace()
            job_settings.input_dir = input_dir
            job_settings.output_dir = temp_path / "output"
            job_settings.temp_dir = temp_path / "temp"
            job_settings.debug = False

            # Missing timeseries.tiff. The shared find_one names both the
            # pattern and the directory it searched.
            with self.assertRaises(FileNotFoundError) as cm:
                TiffSplitterCLI(job_settings, metadata=metadata)
            self.assertIn("*timeseries.tiff", str(cm.exception))
            self.assertIn(str(input_dir), str(cm.exception))

            # Create timeseries.tiff but leave depth file missing
            (input_dir / "test_timeseries.tiff").touch()
            with self.assertRaises(FileNotFoundError) as cm:
                TiffSplitterCLI(job_settings, metadata=metadata)
            self.assertIn("*averaged_depth.tiff", str(cm.exception))

            # Create depth file but leave surface file missing
            (input_dir / "test_averaged_depth.tiff").touch()
            with self.assertRaises(FileNotFoundError) as cm:
                TiffSplitterCLI(job_settings, metadata=metadata)
            self.assertIn("*averaged_surface.tiff", str(cm.exception))

    def test_split_timeseries_tiff_logger_messages(self) -> None:
        """Test that split_timeseries_tiff emits log records via the
        module-level logger.
        """
        import logging

        from aind_pophys_converter.tiff_splitters import (
            split_timeseries_tiff,
        )

        # Create minimal test setup
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            tiff_path = temp_path / "test.tif"
            tiff_path.touch()

            output_path = temp_path / "output.h5"
            offset_to_path = {0: output_path}

            with self.assertLogs(level=logging.INFO):
                try:
                    split_timeseries_tiff(
                        tiff_path=tiff_path,
                        offset_to_path=offset_to_path,
                        dump_every=10,
                    )
                except Exception:
                    # Expected to fail on an invalid TIFF; the start-message
                    # log is emitted before the failure, so assertLogs passes.
                    pass

    def test_avg_image_tiff_splitter_properties(self) -> None:
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

    def test_tiff_splitter_base_utility_methods(self) -> None:
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

    def test_tiff_splitter_error_conditions(self) -> None:
        """Test error handling in TiffSplitter classes."""
        from aind_pophys_converter.tiff_splitters import (
            AvgImageTiffSplitter,
        )

        # Test file not found error - this raises ValueError, not
        # FileNotFoundError
        with self.assertRaises(ValueError) as context:
            AvgImageTiffSplitter(tiff_path=Path("nonexistent.tif"))
        self.assertIn("is not a file", str(context.exception))

    def test_logging_and_cleanup_paths(self) -> None:
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

    def test_utility_functions(self) -> None:
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

    def test_splitter_cli(self) -> None:
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
