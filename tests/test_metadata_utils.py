"""Tests for aind_pophys_converter.utils.metadata_utils."""

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from aind_data_schema.core.processing import Processing

from aind_pophys_converter.utils.metadata_utils import (
    ConverterMetadata,
    _build_fov_ids,
    _find_acquisition_file,
    _find_data_description_file,
    _fov_id,
    _map_v1,
    _map_v2,
    build_processing,
    load_converter_metadata,
    write_processing_json,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixture_dir(version: str, dest_root: Path) -> Path:
    """Copy a fixture dir into a temp location to keep tests isolated."""
    src = FIXTURES / version
    dest = dest_root / version
    shutil.copytree(src, dest)
    return dest


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

    def test_missing_returns_none(self):
        """Absent data_description.json returns None (it is optional)."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_find_data_description_file(Path(tmp)))


class TestMapV1(unittest.TestCase):
    """Tests for _map_v1."""

    def test_round_trip_from_fixture(self):
        """v1 fixture maps to expected ConverterMetadata."""
        acq_path = FIXTURES / "v1" / "session.json"
        dd_path = FIXTURES / "v1" / "data_description.json"
        acq_raw = json.loads(acq_path.read_text())
        dd_raw = json.loads(dd_path.read_text())
        meta = _map_v1(acq_raw, dd_raw, acq_path)
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.subject_id, "000000")
        self.assertEqual(
            meta.dataset_name,
            "multiplane-ophys_000000_2025-05-17_10-00-00",
        )
        # Two planes (0=MOp, 1=VISp) -> multi-plane '{acronym}_{index}' ids.
        self.assertEqual(meta.fov_ids, ("MOp_0", "VISp_1"))

    def test_missing_rig_id_raises(self):
        """Missing rig_id raises KeyError naming the field."""
        with self.assertRaises(KeyError) as cm:
            _map_v1({}, {"name": "x"}, Path("session.json"))
        self.assertIn("rig_id", str(cm.exception))

    def test_single_plane_minimal(self):
        """Minimal single-plane v1: no FOV index, bare-string structure,
        no data_description. See single_plane_support.md."""
        acq = {
            "data_streams": [
                {"ophys_fovs": [{"targeted_structure": "DLS"}]}
            ],
            "subject_id": "NY129",
            "rig_id": "Bruker",
        }
        meta = _map_v1(acq, {}, Path("session.json"))
        self.assertEqual(meta.instrument_id, "Bruker")
        self.assertEqual(meta.subject_id, "NY129")
        self.assertIsNone(meta.dataset_name)
        # Single plane -> plane_0 id (bare-string structure ignored).
        self.assertEqual(meta.fov_ids, ("plane_0",))

    def test_subject_id_optional(self):
        """subject_id is optional (logging label only) -> None when absent."""
        acq = {
            "data_streams": [{"ophys_fovs": [{"index": 0}]}],
            "rig_id": "Bruker",
        }
        meta = _map_v1(acq, {}, Path("session.json"))
        self.assertIsNone(meta.subject_id)

    def test_multiplane_bare_string_structures(self):
        """Real multi-plane v1 uses bare-string targeted_structure and no
        data_description; ids are '{structure}_{index}'."""
        meta = load_converter_metadata(FIXTURES / "multiplane_v1")
        self.assertEqual(meta.instrument_id, "MESO.2")
        self.assertIsNone(meta.dataset_name)
        self.assertEqual(len(meta.fov_ids), 8)
        self.assertEqual(meta.fov_ids[0], "VISp_0")
        self.assertEqual(meta.fov_ids[7], "VISp_7")


class TestMapV2(unittest.TestCase):
    """Tests for _map_v2."""

    def test_round_trip_from_fixture(self):
        """v2 fixture maps to expected ConverterMetadata."""
        acq_path = FIXTURES / "v2" / "acquisition.json"
        dd_path = FIXTURES / "v2" / "data_description.json"
        acq_raw = json.loads(acq_path.read_text())
        dd_raw = json.loads(dd_path.read_text())
        meta = _map_v2(acq_raw, dd_raw, acq_path)
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.subject_id, "000000")
        self.assertEqual(meta.fov_ids, ("MOp_0", "VISp_1"))

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
        meta = _map_v2(acq, dd, Path("a.json"))
        self.assertEqual(meta.fov_ids, ())

    def test_missing_instrument_id_raises(self):
        """Missing instrument_id raises KeyError."""
        with self.assertRaises(KeyError) as cm:
            _map_v2({}, {"name": "x"}, Path("acquisition.json"))
        self.assertIn("instrument_id", str(cm.exception))


class TestEquivalence(unittest.TestCase):
    """v1 and v2 semantically-equivalent fixtures must produce equal
    ConverterMetadata (the object holds no source-provenance fields)."""

    def test_v1_and_v2_match(self):
        """The two fixture sets describe the same session; they should
        map to equal ConverterMetadata objects."""
        v1 = load_converter_metadata(FIXTURES / "v1")
        v2 = load_converter_metadata(FIXTURES / "v2")
        self.assertEqual(v1, v2)


class TestLoadConverterMetadata(unittest.TestCase):
    """End-to-end loader tests (file-presence dispatch)."""

    def test_v1_fixture(self):
        """Loader parses a v1 (session.json) input directory."""
        meta = load_converter_metadata(FIXTURES / "v1")
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.fov_ids, ("MOp_0", "VISp_1"))

    def test_v2_fixture(self):
        """Loader parses a v2 (acquisition.json) input directory."""
        meta = load_converter_metadata(FIXTURES / "v2")
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.fov_ids, ("MOp_0", "VISp_1"))

    def test_single_plane_fixture_no_schema_version_no_dd(self):
        """Loader handles a minimal single-plane session.json with no
        schema_version and no data_description (dispatch by filename)."""
        meta = load_converter_metadata(FIXTURES / "singleplane_v1")
        self.assertIsNone(meta.dataset_name)
        self.assertEqual(meta.fov_ids, ("plane_0",))


class TestFovId(unittest.TestCase):
    """Tests for the FOV/plane naming scheme (_fov_id + fov_id)."""

    def test_multi_plane_uses_structure_index(self):
        """Multi-plane -> '{acronym}_{plane_index}'."""
        self.assertEqual(
            _fov_id(3, "VISp", single_plane=False), "VISp_3"
        )

    def test_single_plane_uses_plane_index(self):
        """Single-plane -> 'plane_{plane_index}' regardless of structure."""
        self.assertEqual(_fov_id(0, "DLS", single_plane=True), "plane_0")

    def test_missing_acronym_falls_back_to_plane_index(self):
        """A multi-plane plane with no acronym never yields 'None_0'."""
        self.assertEqual(_fov_id(2, None, single_plane=False), "plane_2")

    def test_build_fov_ids_single_vs_multi(self):
        """_build_fov_ids: lone plane -> plane_0; multiple -> structure ids;
        sorted by plane index; empty -> ()."""
        self.assertEqual(_build_fov_ids([(0, "DLS")]), ("plane_0",))
        self.assertEqual(
            _build_fov_ids([(1, "MOp"), (0, "VISp")]),
            ("VISp_0", "MOp_1"),
        )
        self.assertEqual(_build_fov_ids([]), ())

    def test_build_fov_ids_multi_missing_acronym_falls_back(self):
        """A multi-plane plane with no acronym falls back to plane_{index}."""
        self.assertEqual(
            _build_fov_ids([(0, "VISp"), (1, None)]),
            ("VISp_0", "plane_1"),
        )


class TestBuildProcessing(unittest.TestCase):
    """Tests for build_processing."""

    def _params(self) -> dict:
        """Return a sample JobSettings parameter dict."""
        return {
            "input_dir": "/data",
            "output_dir": "/results",
            "temp_dir": None,
            "debug": False,
        }

    def _times(self):
        """Return a (start, end) pair of tz-aware datetimes."""
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
        if hasattr(recorded, "model_dump"):
            recorded = recorded.model_dump()
        self.assertEqual(recorded, params)

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
            instrument_id="x",
            fov_ids=("VISp_0",),
            dataset_name="y",
            subject_id="z",
        )
        b = ConverterMetadata(
            instrument_id="x",
            fov_ids=("VISp_0",),
            dataset_name="y",
            subject_id="z",
        )
        self.assertEqual(a, b)


# Silence "patch is unused" warnings in environments where this module is
# only imported, not executed. The import keeps test_metadata_utils
# self-contained when refactoring further.
_ = patch  # noqa: F841


if __name__ == "__main__":
    unittest.main()
