"""Tests for aind_pophys_converter.utils.metadata_utils."""

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from aind_data_schema.core.processing import Processing
from aind_data_schema.core.quality_control import Stage, Status

from aind_pophys_converter.utils.metadata_utils import (
    ConverterMetadata,
    PlaneInfo,
    _detect_major_version,
    _find_acquisition_file,
    _find_data_description_file,
    _map_v1,
    _map_v2,
    build_fov_qc_metric,
    build_processing,
    build_vasculature_qc_metric,
    fov_id_from_plane,
    load_converter_metadata,
    pending_qc_status,
    write_processing_json,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixture_dir(version: str, dest_root: Path) -> Path:
    """Copy a fixture dir into a temp location to keep tests isolated."""
    src = FIXTURES / version
    dest = dest_root / version
    shutil.copytree(src, dest)
    return dest


class TestDetectMajorVersion(unittest.TestCase):
    """Tests for _detect_major_version."""

    def test_v1_string(self):
        """A '1.x.y' schema_version yields major 1."""
        self.assertEqual(
            _detect_major_version({"schema_version": "1.1.2"}, Path("a.json")),
            1,
        )

    def test_v2_string(self):
        """A '2.x.y' schema_version yields major 2."""
        self.assertEqual(
            _detect_major_version({"schema_version": "2.4.0"}, Path("a.json")),
            2,
        )

    def test_missing_raises(self):
        """No schema_version field raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            _detect_major_version({}, Path("a.json"))
        self.assertIn("schema_version", str(cm.exception))

    def test_malformed_raises(self):
        """Non-numeric major raises ValueError."""
        with self.assertRaises(ValueError):
            _detect_major_version({"schema_version": "abc"}, Path("a.json"))


class TestFindAcquisitionFile(unittest.TestCase):
    """Tests for _find_acquisition_file."""

    def setUp(self):
        """Per-test scratch directory."""
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        """Remove scratch directory."""
        self.tmp.cleanup()

    def test_prefers_v2(self):
        """When only acquisition.json is present, return it."""
        (self.root / "acquisition.json").write_text("{}")
        self.assertEqual(
            _find_acquisition_file(self.root),
            self.root / "acquisition.json",
        )

    def test_falls_back_to_v1(self):
        """When only session.json is present, return it."""
        (self.root / "session.json").write_text("{}")
        self.assertEqual(
            _find_acquisition_file(self.root),
            self.root / "session.json",
        )

    def test_ambiguous_raises(self):
        """Both files present raises ValueError."""
        (self.root / "acquisition.json").write_text("{}")
        (self.root / "session.json").write_text("{}")
        with self.assertRaises(ValueError) as cm:
            _find_acquisition_file(self.root)
        self.assertIn("Ambiguous", str(cm.exception))

    def test_missing_raises(self):
        """Neither file present raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _find_acquisition_file(self.root)


class TestFindDataDescriptionFile(unittest.TestCase):
    """Tests for _find_data_description_file."""

    def test_missing_raises(self):
        """Absent data_description.json raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _find_data_description_file(Path(tmp))


class TestMapV1(unittest.TestCase):
    """Tests for _map_v1."""

    def test_round_trip_from_fixture(self):
        """v1 fixture maps to expected ConverterMetadata."""
        acq_path = FIXTURES / "v1" / "session.json"
        dd_path = FIXTURES / "v1" / "data_description.json"
        acq_raw = json.loads(acq_path.read_text())
        dd_raw = json.loads(dd_path.read_text())
        meta = _map_v1(acq_raw, dd_raw, acq_path, dd_path, "1.1.2")
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.subject_id, "000000")
        self.assertEqual(
            meta.dataset_name,
            "multiplane-ophys_000000_2025-05-17_10-00-00",
        )
        self.assertEqual(len(meta.planes), 2)
        # Sorted by plane_index ascending
        self.assertEqual(meta.planes[0].plane_index, 0)
        self.assertEqual(meta.planes[0].targeted_structure_acronym, "MOp")
        self.assertEqual(meta.planes[1].plane_index, 1)
        self.assertEqual(meta.planes[1].targeted_structure_acronym, "VISp")

    def test_missing_rig_id_raises(self):
        """Missing rig_id raises KeyError naming the field."""
        with self.assertRaises(KeyError) as cm:
            _map_v1({}, {"name": "x"}, Path("session.json"),
                    Path("data_description.json"), "1.1.2")
        self.assertIn("rig_id", str(cm.exception))


class TestMapV2(unittest.TestCase):
    """Tests for _map_v2."""

    def test_round_trip_from_fixture(self):
        """v2 fixture maps to expected ConverterMetadata."""
        acq_path = FIXTURES / "v2" / "acquisition.json"
        dd_path = FIXTURES / "v2" / "data_description.json"
        acq_raw = json.loads(acq_path.read_text())
        dd_raw = json.loads(dd_path.read_text())
        meta = _map_v2(acq_raw, dd_raw, acq_path, dd_path, "2.4.0")
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.subject_id, "000000")
        self.assertEqual(len(meta.planes), 2)
        self.assertEqual(meta.planes[0].plane_index, 0)
        self.assertEqual(meta.planes[0].targeted_structure_acronym, "MOp")
        self.assertEqual(meta.planes[1].plane_index, 1)
        self.assertEqual(meta.planes[1].targeted_structure_acronym, "VISp")

    def test_skips_non_imaging_configs(self):
        """configurations with other object_type values are skipped."""
        acq = {
            "instrument_id": "x",
            "subject_id": "x",
            "data_streams": [
                {"configurations": [{"object_type": "Other config"}]}
            ],
        }
        dd = {"name": "x"}
        meta = _map_v2(acq, dd, Path("a.json"), Path("d.json"), "2.4.0")
        self.assertEqual(meta.planes, ())

    def test_missing_instrument_id_raises(self):
        """Missing instrument_id raises KeyError."""
        with self.assertRaises(KeyError) as cm:
            _map_v2({}, {"name": "x"}, Path("acquisition.json"),
                    Path("data_description.json"), "2.4.0")
        self.assertIn("instrument_id", str(cm.exception))


class TestEquivalence(unittest.TestCase):
    """v1 and v2 semantically-equivalent fixtures must produce equal
    ConverterMetadata after stripping the source provenance fields."""

    def test_v1_and_v2_match(self):
        """The two fixture sets describe the same session; they should
        map to equal ConverterMetadata objects."""
        v1 = load_converter_metadata(FIXTURES / "v1")
        v2 = load_converter_metadata(FIXTURES / "v2")
        stripped_v1 = replace(
            v1,
            source_schema_version="",
            source_acquisition_file=Path(""),
            source_data_description_file=Path(""),
        )
        stripped_v2 = replace(
            v2,
            source_schema_version="",
            source_acquisition_file=Path(""),
            source_data_description_file=Path(""),
        )
        self.assertEqual(stripped_v1, stripped_v2)


class TestLoadConverterMetadata(unittest.TestCase):
    """End-to-end loader tests."""

    def test_v1_fixture(self):
        """Loader handles v1 input directory."""
        meta = load_converter_metadata(FIXTURES / "v1")
        self.assertEqual(meta.source_schema_version, "1.1.2")
        self.assertTrue(
            meta.source_acquisition_file.name == "session.json"
        )

    def test_v2_fixture(self):
        """Loader handles v2 input directory."""
        meta = load_converter_metadata(FIXTURES / "v2")
        self.assertEqual(meta.source_schema_version, "2.4.0")
        self.assertTrue(
            meta.source_acquisition_file.name == "acquisition.json"
        )

    def test_unsupported_major_raises(self):
        """A schema_version of 3.x.x raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "acquisition.json").write_text(
                json.dumps({"schema_version": "3.0.0"})
            )
            (root / "data_description.json").write_text(
                json.dumps({"schema_version": "3.0.0"})
            )
            with self.assertRaises(ValueError) as cm:
                load_converter_metadata(root)
            self.assertIn("Unsupported", str(cm.exception))


class TestFovIdFromPlane(unittest.TestCase):
    """Tests for fov_id_from_plane."""

    def test_format(self):
        """Returns '{acronym}_{plane_index}'."""
        plane = PlaneInfo(plane_index=3, targeted_structure_acronym="VISp")
        self.assertEqual(fov_id_from_plane(plane), "VISp_3")

    def test_zero_index(self):
        """plane_index of 0 still renders."""
        plane = PlaneInfo(plane_index=0, targeted_structure_acronym="MO")
        self.assertEqual(fov_id_from_plane(plane), "MO_0")


class TestPendingQcStatus(unittest.TestCase):
    """Tests for pending_qc_status."""

    def test_returns_pending_status(self):
        """pending_qc_status returns Status.PENDING."""
        self.assertEqual(pending_qc_status().status, Status.PENDING)

    def test_evaluator_is_automated(self):
        """pending_qc_status sets evaluator to 'Automated'."""
        self.assertEqual(pending_qc_status().evaluator, "Automated")


class TestBuildQcMetrics(unittest.TestCase):
    """Tests for build_fov_qc_metric and build_vasculature_qc_metric."""

    def test_fov_metric_stage(self):
        """build_fov_qc_metric returns Stage.RAW."""
        metric = build_fov_qc_metric(
            unique_id="VISp_200",
            raw_tif_path=Path("/fake/parent.tif"),
            intended_depth=200,
            targeted_structure_id="VISp",
            closest_z=-200.0,
            merged_path=Path("/fake/merged.png"),
        )
        self.assertEqual(metric.stage, Stage.RAW)

    def test_fov_metric_name_contains_unique_id(self):
        """build_fov_qc_metric name includes the unique_id."""
        metric = build_fov_qc_metric(
            unique_id="MOp_100",
            raw_tif_path=Path("/fake/parent.tif"),
            intended_depth=100,
            targeted_structure_id="MOp",
            closest_z=-100.0,
            merged_path=Path("/fake/merged.png"),
        )
        self.assertIn("MOp_100", metric.name)

    def test_vasculature_metric_stage(self):
        """build_vasculature_qc_metric returns Stage.RAW."""
        metric = build_vasculature_qc_metric(Path("/fake/vasculature.png"))
        self.assertEqual(metric.stage, Stage.RAW)

    def test_vasculature_metric_reference(self):
        """build_vasculature_qc_metric reference matches input path."""
        fp = Path("/fake/vasculature.png")
        self.assertEqual(
            build_vasculature_qc_metric(fp).reference, str(fp)
        )


class TestBuildProcessing(unittest.TestCase):
    """Tests for build_processing."""

    def _params(self) -> dict:
        return {
            "input_dir": "/data",
            "output_dir": "/results",
            "temp_dir": None,
            "debug": False,
        }

    def _times(self):
        start = datetime(2025, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 5, 17, 11, 0, 0, tzinfo=timezone.utc)
        return start, end

    def test_success_path(self):
        """build_processing with error=None yields notes=None."""
        start, end = self._times()
        proc = build_processing(
            job_parameters=self._params(),
            library_version="0.0.1",
            start_time=start,
            end_time=end,
            error=None,
        )
        self.assertEqual(len(proc.data_processes), 1)
        self.assertIsNone(proc.data_processes[0].notes)

    def test_failure_path(self):
        """build_processing with an error populates notes."""
        start, end = self._times()
        proc = build_processing(
            job_parameters=self._params(),
            library_version="0.0.1",
            start_time=start,
            end_time=end,
            error=ValueError("boom"),
        )
        self.assertIsNotNone(proc.data_processes[0].notes)
        self.assertIn("ValueError", proc.data_processes[0].notes)
        self.assertIn("boom", proc.data_processes[0].notes)

    def test_code_parameters_are_only_job_parameters(self):
        """code.parameters reflects exactly the JobSettings dict."""
        start, end = self._times()
        params = self._params()
        proc = build_processing(
            job_parameters=params,
            library_version="0.0.1",
            start_time=start,
            end_time=end,
        )
        # Parameters in v2 are GenericModel (BaseModel) — convert to dict
        recorded = proc.data_processes[0].code.parameters
        self.assertEqual(
            recorded.model_dump() if hasattr(recorded, "model_dump") else recorded,
            params,
        )

    def test_round_trip_json(self):
        """Processing object survives a JSON round-trip."""
        start, end = self._times()
        proc = build_processing(
            job_parameters=self._params(),
            library_version="0.0.1",
            start_time=start,
            end_time=end,
        )
        Processing.model_validate_json(proc.model_dump_json())


class TestWriteProcessingJson(unittest.TestCase):
    """Tests for write_processing_json."""

    def test_writes_file(self):
        """write_processing_json creates a parseable processing.json."""
        start = datetime(2025, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 5, 17, 11, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            proc = build_processing(
                job_parameters={
                    "input_dir": "/data",
                    "output_dir": str(out),
                    "temp_dir": None,
                    "debug": False,
                },
                library_version="0.0.1",
                start_time=start,
                end_time=end,
            )
            written = write_processing_json(proc, out)
            self.assertTrue(written.exists())
            content = json.loads(written.read_text())
            self.assertIn("schema_version", content)


class TestConverterMetadataEquality(unittest.TestCase):
    """Sanity: ConverterMetadata equality works for the equivalence test."""

    def test_equal_when_fields_match(self):
        """Two identically-populated ConverterMetadata compare equal."""
        a = ConverterMetadata(
            source_schema_version="x",
            source_acquisition_file=Path("a"),
            source_data_description_file=Path("b"),
            instrument_id="x",
            dataset_name="y",
            subject_id="z",
            planes=(PlaneInfo(0, "VISp"),),
        )
        b = ConverterMetadata(
            source_schema_version="x",
            source_acquisition_file=Path("a"),
            source_data_description_file=Path("b"),
            instrument_id="x",
            dataset_name="y",
            subject_id="z",
            planes=(PlaneInfo(0, "VISp"),),
        )
        self.assertEqual(a, b)


# Silence "patch is unused" warnings in environments where this module is
# only imported, not executed. The import keeps test_metadata_utils
# self-contained when refactoring further.
_ = patch  # noqa: F841


if __name__ == "__main__":
    unittest.main()
