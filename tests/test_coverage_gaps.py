"""Focused tests for converter branches not covered by integration fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import h5py
import numpy as np
import tifffile

from aind_pophys_converter import job
from aind_pophys_converter import mesoscope_splitter as mesoscope
from aind_pophys_converter import qc
from aind_pophys_converter.tiff_splitters import (
    AvgImageTiffSplitter,
    TiffSplitterBase,
    TimeSeriesSplitter,
    ZStackSplitter,
    _cleanup_temp_files,
    _gather_timeseries_caches,
    build_roi_remap,
)
from aind_pophys_converter.utils import full_field_utils
from aind_pophys_converter.utils.tiff_metadata import ScanImageMetadata

from .test_tiff_splitter import _create_image_tiff, _create_z_stack_tiffs


class TestTiffMetadataCoverage(unittest.TestCase):
    """Exercise defensive ScanImage metadata validation branches."""

    def _metadata_object(self, roi: dict) -> ScanImageMetadata:
        """Create metadata object backed by one supplied ROI."""
        path = Path(tempfile.mkstemp(suffix=".tiff")[1])
        metadata = [
            {"SI.hStackManager.zs": [0]},
            {"RoiGroups": {"imagingRoiGroup": {"rois": [roi]}}},
        ]
        patcher = patch(
            "tifffile.read_scanimage_metadata", return_value=metadata
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(path.unlink, missing_ok=True)
        return ScanImageMetadata(path)

    def test_invalid_roi_and_scanfield_shapes(self) -> None:
        """Raise clear errors for invalid ROI indexes and scanfield data."""
        metadata = self._metadata_object(
            {
                "scanfields": [
                    {"centerXY": [0.0, 0.0]},
                    {"centerXY": [1.0, 0.0]},
                ]
            }
        )
        with self.assertRaises(ValueError):
            metadata.roi_center(1)
        with self.assertRaises(RuntimeError):
            metadata.roi_center(0)

        invalid = self._metadata_object({"scanfields": "invalid"})
        with self.assertRaises(RuntimeError):
            invalid.roi_center(0)
        with self.assertRaises(RuntimeError):
            invalid.roi_size(0)
        with self.assertRaises(RuntimeError):
            invalid.roi_resolution(0)

        missing_size = self._metadata_object({"scanfields": []})
        with self.assertRaises(ValueError):
            missing_size.roi_size(0)
        with self.assertRaises(ValueError):
            missing_size.roi_resolution(0)
        with self.assertRaises(ValueError):
            missing_size.roi_resolution(1)

    def test_inconsistent_center_dimensions_raise(self) -> None:
        """Reject inconsistent X and Y center coordinates independently."""
        for centers in (
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ):
            with self.subTest(centers=centers):
                metadata = self._metadata_object(
                    {
                        "scanfields": [
                            {"centerXY": centers[0]},
                            {"centerXY": centers[1]},
                        ]
                    }
                )
                with self.assertRaises(RuntimeError):
                    metadata.roi_center(0)

    def test_missing_z_metadata_raises(self) -> None:
        """Raise when neither ScanImage z metadata key is present."""
        path = Path(tempfile.mkstemp(suffix=".tiff")[1])
        metadata = [{}, {"RoiGroups": {"imagingRoiGroup": {"rois": []}}}]
        with patch("tifffile.read_scanimage_metadata", return_value=metadata):
            obj = ScanImageMetadata(path)
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaises(ValueError) as error:
            obj.all_zs()
        self.assertIn("Neither", str(error.exception))


class TestFullFieldCoverage(unittest.TestCase):
    """Exercise full-field path discovery and string-path conversion."""

    def test_full_field_path_variants(self) -> None:
        """Find storage files and report absent full-field files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            platform = root / "platform.json"
            storage = root / "storage"
            upload = root / "upload"
            storage.mkdir()
            upload.mkdir()
            image = storage / "fullfield.tiff"
            image.touch()
            platform.write_text(json.dumps({"fullfield_2p_image": image.name}))

            args = {
                "platform_json_path": platform,
                "storage_directory": storage,
                "data_upload_dir": None,
            }
            self.assertEqual(full_field_utils.get_full_field_path(args), image)

            args["data_upload_dir"] = upload
            image.unlink()
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(full_field_utils.get_full_field_path(args))

            platform.write_text(json.dumps({}))
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(full_field_utils.get_full_field_path(args))

    def test_stitch_tiff_with_string_paths(self) -> None:
        """Convert string inputs to paths before delegating to stitchers."""
        splitter = MagicMock()
        metadata = MagicMock()
        with (
            patch.object(
                full_field_utils,
                "stitch_full_field_tiff",
                return_value=np.zeros((2, 2)),
            ) as stitch,
            patch.object(
                full_field_utils,
                "ScanImageMetadata",
                return_value=metadata,
            ),
            patch.object(
                full_field_utils,
                "AvgImageTiffSplitter",
                return_value=splitter,
            ),
            patch.object(
                full_field_utils,
                "_insert_rois_into_surface_img",
                return_value=np.ones((2, 2)),
            ) as insert,
        ):
            result = full_field_utils.stitch_tiff_with_rois(
                "full.tiff", "surface.tiff"
            )
        self.assertEqual(result.shape, (2, 2))
        stitch.assert_called_once_with(Path("full.tiff"))
        insert.assert_called_once()


class TestQualityControlCoverage(unittest.TestCase):
    """Exercise QC pairing warning and normalization branches."""

    def _write_platform(self, path: Path, planes: list) -> None:
        """Write platform JSON containing imaging planes."""
        path.write_text(
            json.dumps({"imaging_plane_groups": [{"imaging_planes": planes}]})
        )

    def test_pairing_warns_on_duplicate_parents(self) -> None:
        """Use the first parent when several match, and warn."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pophys = root / "pophys"
            avg = root / "avg"
            output = root / "output"
            for directory in (pophys, avg, output):
                directory.mkdir()
            platform = root / "platform.json"
            self._write_platform(
                platform,
                [
                    {
                        "intended_depth": 200,
                        "targeted_structure_id": "MOp",
                        "scanimage_scanfield_z": -20.0,
                    },
                ],
            )
            tifffile.imwrite(avg / "MOp_200.tif", np.zeros((4, 4)))
            for name in (
                "first_200_MOp_depth.tif",
                "second_200_MOp_depth.tif",
            ):
                tifffile.imwrite(pophys / name, np.ones((4, 4)))
            with self.assertLogs(level="WARNING") as logged:
                result = qc.pair_depth_tifs_with_avg_depth_pngs(
                    pophys, avg, platform, output, root
                )
            self.assertEqual(len(result), 1)
            self.assertTrue(list(output.glob("*.png")))
            self.assertTrue(
                any("Multiple parent" in m for m in logged.output)
            )

    def test_pairing_skips_missing_parent(self) -> None:
        """A child plane with no parent snapshot is skipped with a
        warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pophys = root / "pophys"
            avg = root / "avg"
            output = root / "output"
            for directory in (pophys, avg, output):
                directory.mkdir()
            platform = root / "platform.json"
            self._write_platform(
                platform,
                [
                    {
                        "intended_depth": 100,
                        "targeted_structure_id": "VISp",
                        "scanimage_scanfield_z": -10.0,
                    },
                ],
            )
            tifffile.imwrite(avg / "VISp_100.tif", np.zeros((4, 4)))
            with self.assertLogs(level="WARNING") as logged:
                result = qc.pair_depth_tifs_with_avg_depth_pngs(
                    pophys, avg, platform, output, root
                )
            self.assertEqual(result, [])
            self.assertTrue(
                any("No parent depth TIF" in m for m in logged.output)
            )

    def test_pairing_handles_constant_images_and_unlink_failure(self) -> None:
        """Use zero normalization and keep a child file when unlink fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pophys = root / "pophys"
            avg = root / "avg"
            output = root / "output"
            for directory in (pophys, avg, output):
                directory.mkdir()
            platform = root / "platform.json"
            self._write_platform(
                platform,
                [
                    {
                        "intended_depth": 100,
                        "targeted_structure_id": "VISp",
                        "scanimage_scanfield_z": -10.0,
                    }
                ],
            )
            parent = pophys / "one_100_VISp_depth.tif"
            child = avg / "VISp_100.tif"
            tifffile.imwrite(parent, np.ones((4, 4)))
            tifffile.imwrite(child, np.ones((4, 4)))
            with patch.object(Path, "unlink", side_effect=OSError("locked")):
                result = qc.pair_depth_tifs_with_avg_depth_pngs(
                    pophys, avg, platform, output, root
                )
            self.assertEqual(len(result), 1)
            self.assertTrue(child.exists())

    def test_qc_type_checking_import_branch(self) -> None:
        """Execute the TYPE_CHECKING-only import for literal coverage."""
        import importlib
        import typing

        original = typing.TYPE_CHECKING
        try:
            typing.TYPE_CHECKING = True
            importlib.reload(qc)
        finally:
            typing.TYPE_CHECKING = original
            importlib.reload(qc)


class TestTiffSplitterCoverage(unittest.TestCase):
    """Exercise low-level splitter validation and cleanup branches."""

    def _avg_splitter(
        self, root: Path, z_values: list, n_rois: int, surface: bool = False
    ) -> tuple:
        """Create an average-image splitter and its fixture metadata."""
        fixture = _create_image_tiff(
            root,
            z_values,
            n_rois,
            use_zs=True,
            is_surface=surface,
        )
        patcher = patch(
            "tifffile.read_scanimage_metadata", return_value=fixture[4]
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return AvgImageTiffSplitter(fixture[0]), fixture

    def test_average_splitter_validation_branches(self) -> None:
        """Cover invalid ROI, z, shape, offset, and output suffix paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            splitter, _ = self._avg_splitter(root, [0, 1, 2, 3], 2)
            with self.assertRaises(ValueError):
                splitter._get_offset(9, 0)
            with self.assertRaises(ValueError):
                splitter._get_pages(9, 0)
            with self.assertRaises(ValueError):
                splitter._get_pages(0, 99)
            splitter._frame_shape[(0, 0)] = (1, 1)
            with self.assertRaises(RuntimeError):
                splitter._get_pages(0, 0)
            with self.assertRaises(RuntimeError):
                splitter._get_z_value(0)

            timeseries = TimeSeriesSplitter(splitter._file_path)
            complete_map = {
                (0, 0.0): root / "zero.h5",
                (0, 1.0): root / "one.h5",
                (1, 2.0): root / "two.h5",
                (1, 99.0): root / "missing.h5",
            }
            with self.assertRaises(ValueError):
                timeseries._validate_complete_coverage(complete_map)
            with self.assertRaises(RuntimeError):
                timeseries._create_offset_to_path_map(
                    {
                        (0, 0.0): root / "zero.h5",
                        (0, 0.0000005): root / "near.h5",
                    }
                )

            surface, _ = self._avg_splitter(root, [0, 1], 2, surface=True)
            self.assertEqual(surface.frame_shape(0, None), (24, 24))
            with patch("tifffile.imwrite") as write:
                surface.write_output_file(0, None, root / "output.bad")
            write.assert_called_once()

            invalid = object.__new__(TiffSplitterBase)
            invalid._metadata = SimpleNamespace(channelSave=2)
            invalid._file_path = root / "invalid.tiff"
            with self.assertRaises(RuntimeError):
                invalid._filter_z_values_for_single_channel(np.array([1, 2]))
            invalid._metadata.channelSave = [1]
            with self.assertRaises(RuntimeError):
                invalid._validate_dual_channel()

    def test_timeseries_cache_cleanup_and_chunk_limit(self) -> None:
        """Cover temporary cleanup, inconsistent shapes, and chunk limits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            temp_file = root / "temp.h5"
            temp_file.touch()
            temp_subdir = root / "temp_dir"
            temp_subdir.mkdir()
            _cleanup_temp_files({0: [temp_file]}, [temp_subdir])
            self.assertFalse(temp_file.exists())
            self.assertFalse(temp_subdir.exists())

            first = root / "first.h5"
            second = root / "second.h5"
            with h5py.File(first, "w") as file:
                file.create_dataset("data", data=np.zeros((1, 2, 2)))
            with h5py.File(second, "w") as file:
                file.create_dataset("data", data=np.zeros((1, 3, 2)))
            with self.assertRaises(RuntimeError):
                _gather_timeseries_caches([first, second], root / "bad.h5")

            first = root / "first_good.h5"
            second = root / "second_good.h5"
            for path in (first, second):
                with h5py.File(path, "w") as file:
                    file.create_dataset("data", data=np.zeros((1, 2, 2)))
            with patch(
                "aind_pophys_converter.tiff_splitters.np.floor",
                return_value=np.array(1),
            ):
                _gather_timeseries_caches(
                    [first, second], root / "good.h5", metadata={"x": 1}
                )
            with h5py.File(root / "good.h5", "r") as file:
                self.assertEqual(file["data"].shape[0], 2)

    def test_z_stack_and_roi_remap_errors(self) -> None:
        """Cover z-stack properties, shape checks, and remap exhaustion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = _create_z_stack_tiffs(root, [[0, 1]], use_zs=True)
            patcher = patch(
                "aind_pophys_converter.utils.tiff_metadata._read_metadata",
                side_effect=lambda path: dataset[0][str(path.resolve())],
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            splitter = ZStackSplitter(
                [Path(path) for path in dataset[0].keys()]
            )
            self.assertEqual(len(splitter.input_path), 1)
            splitter.frame_shape = Mock(return_value=(1, 1))
            with self.assertRaises(RuntimeError):
                splitter._get_pages(0, 0)
            with self.assertRaises(ValueError):
                splitter.write_output_file(0, 0, root / "bad.tif")

            centers = {
                (0, 0): "one.tiff",
                (0, 1): "two.tiff",
            }
            first = SimpleNamespace(roi_center=Mock(return_value=(0.0, 0.0)))
            second = SimpleNamespace(roi_center=Mock(return_value=(2.0, 0.0)))
            fake = object.__new__(ZStackSplitter)
            fake._roi_z_int_to_path = centers
            fake._path_to_metadata = {
                "one.tiff": first,
                "two.tiff": second,
                "other.tiff": first,
            }
            fake._roi_z_int_to_path[(1, 0)] = "other.tiff"
            fake._int_from_z_lookup = {0.0: 0, 1.0: 1}
            fake._z_from_int_lookup = {0: 0.0, 1: 1.0}
            self.assertEqual(fake.roi_center(0), (0.0, 0.0))

            empty = SimpleNamespace(n_rois=0)
            with self.assertRaises(RuntimeError):
                build_roi_remap(empty, [(0.0, 0.0)])


class TestJobCoverage(unittest.TestCase):
    """Exercise top-level routing, provenance, and QC orchestration."""

    def _settings(self, root: Path) -> MagicMock:
        """Build mocked JobSettings for a temporary run."""
        settings = MagicMock()
        settings.input_dir = root / "input"
        settings.output_dir = root / "output"
        settings.debug = False
        settings.model_dump.return_value = {
            "input_dir": str(settings.input_dir)
        }
        settings.output_dir.mkdir()
        settings.input_dir.mkdir()
        return settings

    def _metadata(self, raw_format: Optional[str]) -> SimpleNamespace:
        """Build minimal metadata for routing tests."""
        return SimpleNamespace(
            raw_format=raw_format,
            instrument_id="rig",
            dataset_name="dataset",
            subject_id="subject",
        )

    def _stage_guard(self) -> tuple:
        """Build a stage-guard context and its mutable context object."""
        context = SimpleNamespace(upstream_names=None)
        manager = MagicMock()
        manager.__enter__.return_value = context
        manager.__exit__.return_value = False
        return manager, context

    def test_run_missing_pophys_uses_asset_fallback(self) -> None:
        """Raise a useful error when no pophys directory is mounted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            guard, context = self._stage_guard()
            metadata = self._metadata(None)
            with (
                patch.object(job, "JobSettings", return_value=settings),
                patch.object(
                    job, "load_converter_metadata", return_value=metadata
                ),
                patch.object(job, "setup_logging"),
                patch.object(job, "find_one", return_value=None),
                patch.object(
                    job, "_mounted_input_assets", return_value=["asset"]
                ),
                patch.object(
                    job, "build_converter_code", return_value=Mock()
                ) as build_code,
                patch.object(job, "stage_guard", return_value=guard),
                patch.object(
                    job, "collect_upstream_process_names", return_value=[]
                ),
            ):
                with self.assertRaises(FileNotFoundError):
                    job.run()
            build_code.assert_called_once()
            self.assertEqual(context.upstream_names, [])

    def test_run_routes_bergamo(self) -> None:
        """Route TIFF stacks to the stitcher."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            pophys = settings.input_dir / "pophys"
            pophys.mkdir()
            guard, _ = self._stage_guard()
            metadata = self._metadata("tiff_stack")
            stitcher = MagicMock()
            with (
                patch.object(job, "JobSettings", return_value=settings),
                patch.object(
                    job, "load_converter_metadata", return_value=metadata
                ),
                patch.object(job, "setup_logging"),
                patch.object(job, "find_one", return_value=pophys),
                patch.object(
                    job, "raw_input_data", return_value=["pophys/raw.tif"]
                ),
                patch.object(job, "build_converter_code", return_value=Mock()),
                patch.object(job, "stage_guard", return_value=guard),
                patch.object(
                    job, "collect_upstream_process_names", return_value=[]
                ),
                patch.object(job, "BergamoSettings") as bergamo_settings,
                patch.object(
                    job, "BergamoTiffStitcher", return_value=stitcher
                ),
            ):
                job.run()
            bergamo_settings.assert_called_once()
            stitcher.run_converter.assert_called_once_with()

    def test_run_routes_mesoscope_without_splits(self) -> None:
        """Run mesoscope splitter and child-session depth QC branches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            pophys = settings.input_dir / "pophys"
            pophys.mkdir()
            platform = pophys / "session_platform.json"
            platform.touch()
            avg_depth = pophys / "session_averaged_depth.tiff"
            avg_depth.touch()
            guard, _ = self._stage_guard()
            metadata = self._metadata("mesoscope_aind")
            runner = MagicMock()
            metrics = [Mock()]
            find_values = [pophys, platform, avg_depth]
            with (
                patch.object(job, "JobSettings", return_value=settings),
                patch.object(
                    job, "load_converter_metadata", return_value=metadata
                ),
                patch.object(job, "setup_logging"),
                patch.object(job, "find_one", side_effect=find_values),
                patch.object(job, "raw_input_data", return_value=["raw"]),
                patch.object(job, "build_converter_code", return_value=Mock()),
                patch.object(job, "stage_guard", return_value=guard),
                patch.object(
                    job, "collect_upstream_process_names", return_value=[]
                ),
                patch.object(job, "find_split_directories", return_value=[]),
                patch.object(job, "TiffSplitterCLI", return_value=runner),
                patch.object(
                    job,
                    "is_child_session_via_platform_json",
                    return_value=True,
                ),
                patch.object(job, "AvgImageTiffSplitter") as splitter,
                patch.object(job, "write_avg_depth_slices") as write_slices,
                patch.object(
                    job,
                    "pair_depth_tifs_with_avg_depth_pngs",
                    return_value=metrics,
                ) as pair,
                patch.object(job, "finalize_quality_control") as finalize,
            ):
                job.run()
            runner.run_job.assert_called_once_with()
            splitter.assert_called_once_with(avg_depth)
            write_slices.assert_called_once()
            pair.assert_called_once()
            finalize.assert_called_once_with(
                metrics, pophys, settings.output_dir
            )

    def test_run_writes_split_directory_markers(self) -> None:
        """Write one marker file per pre-split mesoscope directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            pophys = settings.input_dir / "pophys"
            pophys.mkdir()
            platform = pophys / "session_platform.json"
            platform.touch()
            find_values = [pophys, platform, None]
            guard, _ = self._stage_guard()
            metadata = self._metadata("mesoscope_aind")
            with (
                patch.object(job, "JobSettings", return_value=settings),
                patch.object(
                    job, "load_converter_metadata", return_value=metadata
                ),
                patch.object(job, "setup_logging"),
                patch.object(job, "find_one", side_effect=find_values),
                patch.object(job, "raw_input_data", return_value=["raw"]),
                patch.object(job, "build_converter_code", return_value=Mock()),
                patch.object(job, "stage_guard", return_value=guard),
                patch.object(
                    job, "collect_upstream_process_names", return_value=[]
                ),
                patch.object(
                    job, "find_split_directories", return_value=["V1"]
                ),
                patch.object(job, "finalize_quality_control"),
            ):
                job.run()
            marker = settings.output_dir / "V1" / "V1.txt"
            self.assertEqual(marker.read_text(), "V1.h5")

    def test_run_unknown_raw_format_is_noop(self) -> None:
        """Convert nothing when no raw format is known."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            pophys = settings.input_dir / "pophys"
            pophys.mkdir()
            guard, _ = self._stage_guard()
            metadata = self._metadata(None)
            with (
                patch.object(job, "JobSettings", return_value=settings),
                patch.object(
                    job, "load_converter_metadata", return_value=metadata
                ),
                patch.object(job, "setup_logging"),
                patch.object(job, "find_one", return_value=pophys),
                patch.object(job, "raw_input_data", return_value=["raw"]),
                patch.object(job, "build_converter_code", return_value=Mock()),
                patch.object(job, "stage_guard", return_value=guard),
                patch.object(
                    job, "collect_upstream_process_names", return_value=[]
                ),
                patch.object(job, "BergamoTiffStitcher") as bergamo,
                patch.object(job, "TiffSplitterCLI") as meso,
                patch.object(job, "finalize_quality_control") as finalize,
            ):
                job.run()
            bergamo.assert_not_called()
            meso.assert_not_called()
            finalize.assert_not_called()

    def test_finalize_quality_control_adds_vasculature(self) -> None:
        """Append vasculature QC and write one complete QC document."""
        existing = [Mock()]
        vasculature = Mock()
        with (
            patch.object(qc, "create_vasculature", return_value=vasculature),
            patch.object(
                qc, "build_quality_control", return_value=Mock()
            ) as build,
            patch.object(qc, "write_quality_control_json") as write,
        ):
            qc.finalize_quality_control(
                existing, Path("pophys"), Path("out")
            )
        self.assertEqual(existing[-1], vasculature)
        build.assert_called_once_with(existing)
        write.assert_called_once()

    def test_mounted_input_assets_fallbacks(self) -> None:
        """Return sorted mounted names or None on empty/unreadable input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "z_asset").mkdir()
            (root / "a_asset").mkdir()
            self.assertEqual(
                job._mounted_input_assets(root), ["a_asset", "z_asset"]
            )
            empty = root / "empty"
            empty.mkdir()
            self.assertIsNone(job._mounted_input_assets(empty))
            with patch.object(Path, "iterdir", side_effect=OSError("denied")):
                self.assertIsNone(job._mounted_input_assets(root))


class TestMesoscopeCoverage(unittest.TestCase):
    """Exercise mesoscope splitter orchestration with mocked I/O."""

    def _cli(self, root: Path) -> mesoscope.TiffSplitterCLI:
        """Build an uninitialized CLI with paths needed by a method."""
        cli = object.__new__(mesoscope.TiffSplitterCLI)
        cli.input_dir = root / "input"
        cli.output_dir = root / "output"
        cli.input_dir.mkdir(exist_ok=True)
        cli.output_dir.mkdir(exist_ok=True)
        cli.timeseries_tif = cli.input_dir / "timeseries.tiff"
        cli.depths_tif = cli.input_dir / "averaged_depth.tiff"
        cli.surface_tif = cli.input_dir / "averaged_surface.tiff"
        cli.temp_dir = None
        cli.debug = False
        cli.metadata = SimpleNamespace(fov_ids=("plane_0",))
        return cli

    def test_initialize_and_deprecated_file_check(self) -> None:
        """Initialize all splitter types and warn for deprecated TIFFs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cli = self._cli(root)
            local_z = cli.input_dir / "local_z_0.tiff"
            local_z.touch()
            constructors = [MagicMock() for _ in range(4)]
            with (
                patch.object(
                    mesoscope,
                    "TimeSeriesSplitter",
                    return_value=constructors[0],
                ),
                patch.object(
                    mesoscope,
                    "AvgImageTiffSplitter",
                    side_effect=constructors[1:3],
                ),
                patch.object(
                    mesoscope, "ZStackSplitter", return_value=constructors[3]
                ),
            ):
                result = cli._initialize_splitters()
            self.assertEqual(result[4], [local_z])
            self.assertEqual(len(result[5]), 4)

            deprecated = cli.input_dir / "cortical_z_stack.tiff"
            deprecated.touch()
            with (
                patch.object(mesoscope, "find_one", return_value=deprecated),
                patch(
                    "aind_pophys_converter.mesoscope_splitter.logging.warning"
                ) as warning,
            ):
                cli._check_deprecated_files()
            warning.assert_called_once()

    def test_process_single_fov_and_timeseries(self) -> None:
        """Write per-FOV outputs and update matching timeseries metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cli = self._cli(root)
            splitters = [MagicMock() for _ in range(4)]
            for splitter in splitters:
                splitter.frame_shape.return_value = (2, 3)
            splitters[0].roi_center.return_value = (1.0, 2.0)
            with patch.object(
                mesoscope, "get_nearest_roi_center", return_value=0
            ):
                metadata = cli._process_single_fov(
                    "plane_0",
                    0,
                    -100.0,
                    *splitters,
                    [(1.0, 2.0)],
                    {0: 1},
                    {0: 0},
                    {0: 0},
                    0.0,
                )
            self.assertEqual(metadata["timeseries"], {})
            self.assertEqual(metadata["depth_2p"]["height"], 2)
            self.assertEqual(splitters[1].write_output_file.call_count, 1)
            self.assertEqual(splitters[2].write_output_file.call_count, 1)
            self.assertEqual(splitters[3].write_output_file.call_count, 1)

            cli.metadata.fov_ids = ("plane_0", "plane_1")
            cli._roi_z_per_plane = Mock(return_value=[(0, 1.0), (1, 2.0)])
            splitter = MagicMock()
            splitter.frame_shape.return_value = (4, 5)
            experiments = [
                cli._create_experiment_metadata("plane_0"),
                cli._create_experiment_metadata("plane_1"),
            ]
            lookup = cli._process_timeseries_data(splitter, experiments)
            self.assertEqual(len(lookup), 2)
            self.assertEqual(experiments[1]["timeseries"]["width"], 5)

    def test_full_field_and_file_metadata(self) -> None:
        """Process an available full-field TIFF and record input metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cli = self._cli(root)
            full_field = cli.input_dir / "fullfield.tiff"
            full_field.touch()
            with (
                patch.object(mesoscope, "find_one", return_value=full_field),
                patch.object(
                    mesoscope,
                    "write_out_stitched_full_field_image",
                    side_effect=lambda **kwargs: kwargs["output_path"].touch(),
                ) as write,
            ):
                cli._process_full_field_image()
            write.assert_called_once()
            self.assertTrue(
                (cli.output_dir / "stitched_full_field_img.h5").exists()
            )

            source = cli.input_dir / "source.tiff"
            source.touch()
            fake_metadata = MagicMock()
            fake_metadata._metadata = ({"scan": 1}, {"roi": 2})
            with patch.object(
                mesoscope, "ScanImageMetadata", return_value=fake_metadata
            ):
                result = cli._create_file_metadata([source])
            self.assertEqual(result[0]["scanimage_metadata"], {"scan": 1})

    def test_run_job_success_and_failure_logging(self) -> None:
        """Run orchestration and re-raise initialization failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cli = self._cli(root)
            zstack_path = cli.input_dir / "local_z.tiff"
            zstack_path.touch()
            splitters = [MagicMock() for _ in range(4)]
            splitters[3]._roi_to_path = {0: []}
            cli._initialize_splitters = Mock(
                return_value=(
                    splitters[0],
                    splitters[1],
                    splitters[2],
                    splitters[3],
                    [zstack_path],
                    [cli.timeseries_tif],
                )
            )
            cli._check_deprecated_files = Mock()
            cli._get_imaging_planes = Mock(return_value=("plane_0",))
            cli._roi_z_per_plane = Mock(return_value=[(0, 0.0)])
            cli._process_single_fov = Mock(return_value={"fov": "plane_0"})
            cli._process_timeseries_data = Mock(return_value={})
            cli._process_full_field_image = Mock()
            cli._create_file_metadata = Mock(return_value=[])
            with (
                patch.object(
                    mesoscope, "get_valid_roi_centers", return_value=[(0, 0)]
                ),
                patch.object(mesoscope, "validate_roi_pairing"),
                patch.object(
                    mesoscope, "build_roi_remap", return_value={0: 0}
                ),
            ):
                cli.run_job()
            cli._process_single_fov.assert_called_once()
            splitters[0].write_output_files.assert_called_once()

            failing = self._cli(root)
            failing._initialize_splitters = Mock(
                side_effect=RuntimeError("broken")
            )
            with self.assertLogs(level="ERROR"):
                with self.assertRaises(RuntimeError):
                    failing.run_job()

    def test_from_args_builds_settings(self) -> None:
        """Parse input, output, temp, and debug command-line arguments."""
        settings = SimpleNamespace()
        with patch.object(
            mesoscope, "JobSettings", return_value=settings
        ) as build:
            result = mesoscope.from_args(
                [
                    "--input-dir",
                    "/input",
                    "--output-dir",
                    "/output",
                    "--temp-dir",
                    "/temp",
                    "--debug",
                    "true",
                ]
            )
        self.assertIs(result, settings)
        build.assert_called_once_with(
            input_dir="/input",
            temp_dir="/temp",
            output_dir="/output",
            debug=True,
        )
