"""Tests for aind_pophys_converter.utils.metadata_utils.

The v1/v2 field getters and FOV-naming rule now live in the shared
``aind_pophys_metadata`` library (tested there); these tests cover the
converter-specific surface: ``load_converter_metadata`` assembling a
``ConverterMetadata`` from fixtures, its require/ambiguity/missing contracts,
and the conversion ``Code`` builder.
"""

import json
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

from aind_pophys_converter.utils.metadata_utils import (
    LIBRARY_NAME,
    ConverterMetadata,
    build_converter_code,
    load_converter_metadata,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _write_input(root: Path, core_name: str, core: dict) -> Path:
    """Write a core metadata file into ``root`` and return ``root``."""
    (root / core_name).write_text(json.dumps(core))
    return root


class TestLoadConverterMetadata(unittest.TestCase):
    """End-to-end loader tests (file-presence dispatch + mapping)."""

    def test_v1_fixture(self) -> None:
        """A v1 (session.json) input maps to expected ConverterMetadata."""
        meta = load_converter_metadata(FIXTURES / "v1")
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.subject_id, "000000")
        self.assertEqual(
            meta.dataset_name,
            "multiplane-ophys_000000_2025-05-17_10-00-00",
        )
        # Two planes (0=MOp, 1=VISp) -> multi-plane '{acronym}_{index}' ids.
        self.assertEqual(meta.fov_ids, ("MOp_0", "VISp_1"))

    def test_v2_fixture(self) -> None:
        """A v2 (acquisition.json) input maps to expected metadata."""
        meta = load_converter_metadata(FIXTURES / "v2")
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.subject_id, "000000")
        self.assertEqual(meta.fov_ids, ("MOp_0", "VISp_1"))

    def test_v1_and_v2_equivalent(self) -> None:
        """Semantically-equal v1/v2 fixtures map to equal metadata."""
        self.assertEqual(
            load_converter_metadata(FIXTURES / "v1"),
            load_converter_metadata(FIXTURES / "v2"),
        )

    def test_multiplane_bare_string_structures(self) -> None:
        """Real multi-plane v1 (bare-string structures, no dd)."""
        meta = load_converter_metadata(FIXTURES / "multiplane_v1")
        self.assertEqual(meta.instrument_id, "MESO.2")
        self.assertIsNone(meta.dataset_name)
        self.assertEqual(len(meta.fov_ids), 8)
        self.assertEqual(meta.fov_ids[0], "VISp_0")
        self.assertEqual(meta.fov_ids[7], "VISp_7")

    def test_single_plane_fixture(self) -> None:
        """Minimal single-plane session.json (no schema_version, no dd)."""
        meta = load_converter_metadata(FIXTURES / "singleplane_v1")
        self.assertIsNone(meta.dataset_name)
        self.assertEqual(meta.fov_ids, ("plane_0",))


class TestLoaderContracts(unittest.TestCase):
    """require / ambiguity / missing behaviors, exercised via the loader."""

    def setUp(self) -> None:
        """Per-test scratch directory."""
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        """Remove scratch directory."""
        self.tmp.cleanup()

    def test_missing_rig_id_raises(self) -> None:
        """A v1 input missing rig_id raises KeyError naming the field."""
        _write_input(
            self.root,
            "session.json",
            {"data_streams": [{"ophys_fovs": [{"index": 0}]}]},
        )
        with self.assertRaises(KeyError) as cm:
            load_converter_metadata(self.root)
        self.assertIn("rig_id", str(cm.exception))

    def test_missing_instrument_id_raises(self) -> None:
        """A v2 input missing instrument_id raises KeyError."""
        _write_input(self.root, "acquisition.json", {"data_streams": []})
        with self.assertRaises(KeyError) as cm:
            load_converter_metadata(self.root)
        self.assertIn("instrument_id", str(cm.exception))

    def test_subject_id_optional(self) -> None:
        """subject_id is optional -> None when the core file omits it."""
        _write_input(
            self.root,
            "session.json",
            {
                "rig_id": "Bruker",
                "data_streams": [{"ophys_fovs": [{"index": 0}]}],
            },
        )
        self.assertIsNone(load_converter_metadata(self.root).subject_id)

    def test_ambiguous_input_raises(self) -> None:
        """Both core files present raises ValueError."""
        _write_input(self.root, "session.json", {"rig_id": "x"})
        _write_input(self.root, "acquisition.json", {"instrument_id": "x"})
        with self.assertRaises(ValueError):
            load_converter_metadata(self.root)

    def test_missing_input_raises(self) -> None:
        """Neither core file present raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_converter_metadata(self.root)


class TestBuildConverterCode(unittest.TestCase):
    """Tests for build_converter_code."""

    def _params(self) -> dict:
        """Return a sample JobSettings parameter dict."""
        return {
            "input_dir": "/data",
            "output_dir": "/results",
            "temp_dir": None,
            "debug": False,
        }

    def test_url_is_the_repository_landing_page(self) -> None:
        """code.url is the repo root, with no /tree/<ref>/code suffix."""
        code = build_converter_code(self._params())
        self.assertEqual(
            code.url,
            f"https://github.com/AllenNeuralDynamics/{LIBRARY_NAME}",
        )
        self.assertNotIn("/tree/", code.url)

    def test_version_comes_from_the_installed_distribution(self) -> None:
        """code.version identifies the backing library, never empty."""
        code = build_converter_code(self._params())
        self.assertEqual(code.version, version(LIBRARY_NAME))

    def test_code_parameters_are_only_job_parameters(self) -> None:
        """code.parameters reflects exactly the JobSettings dict."""
        params = self._params()
        recorded = build_converter_code(params).parameters
        if hasattr(recorded, "model_dump"):
            recorded = recorded.model_dump()
        self.assertEqual(recorded, params)

    def test_input_data_records_the_raw_sources(self) -> None:
        """Raw source paths become named input data assets."""
        code = build_converter_code(
            self._params(),
            input_data=["asset/pophys/x_timeseries.tiff"],
        )
        self.assertEqual(
            [asset.name for asset in code.input_data],
            ["asset/pophys/x_timeseries.tiff"],
        )

    def test_ephemeral_input_path_is_rejected(self) -> None:
        """A Nextflow work path in input_data fails loudly."""
        with self.assertRaises(ValueError):
            build_converter_code(
                self._params(),
                input_data=["/tmp/nxf.abc/capsule/data/x.tiff"],
            )


class TestConverterMetadataEquality(unittest.TestCase):
    """Sanity: ConverterMetadata equality works for the equivalence test."""

    def test_equal_when_fields_match(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
