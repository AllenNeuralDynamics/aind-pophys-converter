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
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from aind_pophys_converter.utils.metadata_utils import (
    ConverterMetadata,
    build_converter_code,
    load_converter_metadata,
)
from aind_pophys_converter.utils.platform_utils import (
    is_child_session_via_platform_json,
)

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_FIXTURES = FIXTURES / "synthetic"
LIBRARY = "aind-pophys-converter"


def _write_input(root: Path, core_name: str, core: dict) -> Path:
    """Write a core metadata file into ``root`` and return ``root``."""
    (root / core_name).write_text(json.dumps(core))
    return root


#: v2 fields the pipeline requires, other than the raw format.
_V2_PIPELINE_FIELDS = {
    "acquisition_start_time": "2025-05-17T10:00:00-07:00",
    "data_streams": [
        {
            "configurations": [
                {
                    "object_type": "Imaging config",
                    "sampling_strategy": {"frame_rate": 30.0},
                    "images": [
                        {
                            "object_type": "Planar image",
                            "planes": [{"plane_index": 0}],
                        }
                    ],
                }
            ]
        }
    ],
}

#: The fields a minimal document needs for an NWB file.
_MINIMAL_NWB_FIELDS = {
    "frame_rate": 30.0,
    "session_start_time": "2026-01-01T12:00:00-08:00",
    "planes": [{"plane_index": 0}],
}


class TestLoadConverterMetadata(unittest.TestCase):
    """End-to-end loader tests (file-presence dispatch + mapping)."""

    def test_v1_fixture(self) -> None:
        """A v1 (session.json) input maps to expected ConverterMetadata."""
        meta = load_converter_metadata(SYNTHETIC_FIXTURES / "v1")
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
        meta = load_converter_metadata(SYNTHETIC_FIXTURES / "v2")
        self.assertEqual(meta.instrument_id, "MESO.1")
        self.assertEqual(meta.subject_id, "000000")
        self.assertEqual(meta.fov_ids, ("MOp_0", "VISp_1"))

    def test_epoch_stems_ordered_by_start_time(self) -> None:
        """epoch_stems follows acquisition start time, not document order."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_input(
                Path(tmp),
                "acquisition.json",
                {
                    **_V2_PIPELINE_FIELDS,
                    "instrument_id": "Bergamo",
                    "stimulus_epochs": [
                        {
                            "stimulus_start_time": "2025-05-17T11:00:00-07:00",
                            "output_parameters": {"tiff_stem": "spont"},
                        },
                        {
                            "stimulus_start_time": "2025-05-17T10:00:00-07:00",
                            "output_parameters": {"tiff_stem": "spontpost"},
                        },
                    ],
                },
            )
            meta = load_converter_metadata(root)
        self.assertEqual(meta.epoch_stems, ("spontpost", "spont"))

    def test_epoch_stems_empty_without_epochs(self) -> None:
        """A v2 input with no stimulus epochs yields empty epoch_stems."""
        meta = load_converter_metadata(SYNTHETIC_FIXTURES / "v2")
        self.assertEqual(meta.epoch_stems, ())

    def test_v1_and_v2_equivalent(self) -> None:
        """Semantically-equal v1/v2 fixtures map to equal metadata.

        Only the plane channels differ: v1 sessions record no channel.
        """
        v1 = load_converter_metadata(SYNTHETIC_FIXTURES / "v1")
        v2 = load_converter_metadata(SYNTHETIC_FIXTURES / "v2")
        self.assertEqual(
            replace(v1, plane_channels=()), replace(v2, plane_channels=())
        )
        self.assertEqual(v1.plane_channels[0].channel_name, None)
        self.assertEqual(v2.plane_channels[0].channel_name, "Ch1")

    def test_multiplane_bare_string_structures(self) -> None:
        """Real multi-plane v1 (bare-string structures, no dd)."""
        meta = load_converter_metadata(SYNTHETIC_FIXTURES / "multiplane_v1")
        self.assertEqual(meta.instrument_id, "MESO.2")
        self.assertIsNone(meta.dataset_name)
        self.assertEqual(len(meta.fov_ids), 8)
        self.assertEqual(meta.fov_ids[0], "VISp_0")
        self.assertEqual(meta.fov_ids[7], "VISp_7")

    def test_single_plane_fixture_lacks_pipeline_fields(self) -> None:
        """A non-AIND v1 session without a start time fails up front."""
        with self.assertRaises(ValueError) as ctx:
            load_converter_metadata(SYNTHETIC_FIXTURES / "singleplane_v1")
        for field in ("session_start_time", "raw_format"):
            self.assertIn(field, str(ctx.exception))


class TestLoaderContracts(unittest.TestCase):
    """require / ambiguity / missing behaviors, exercised via the loader."""

    def setUp(self) -> None:
        """Per-test scratch directory."""
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        """Remove scratch directory."""
        self.tmp.cleanup()

    def test_unknown_raw_format_raises(self) -> None:
        """v1 and v2 input with no known raw format fails up front."""
        cases = (
            ("session.json", {"data_streams": [{"ophys_fovs": [{}]}]}),
            ("acquisition.json", _V2_PIPELINE_FIELDS),
        )
        for name, document in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    _write_input(Path(tmp), name, document)
                    with self.assertRaises(ValueError) as ctx:
                        load_converter_metadata(Path(tmp))
                self.assertIn("raw_format", str(ctx.exception))

    def test_minimal_document_states_its_raw_format(self) -> None:
        """An external metadata.json routes without an AIND rig id."""
        _write_input(
            self.root,
            "metadata.json",
            {
                **_MINIMAL_NWB_FIELDS,
                "instrument_id": "unknown",
                "raw_format": "tiff_stack",
                "epochs": [{"tiff_stem": "spont", "reference": True}],
            },
        )
        meta = load_converter_metadata(self.root)
        self.assertEqual(meta.raw_format, "tiff_stack")
        self.assertEqual(meta.instrument_id, "unknown")

    def test_minimal_document_missing_pipeline_fields_raises(self) -> None:
        """Missing required fields fail before any conversion."""
        _write_input(self.root, "metadata.json", {})
        with self.assertRaises(ValueError) as ctx:
            load_converter_metadata(self.root)
        fields = ("frame_rate", "session_start_time", "planes", "raw_format")
        for field in fields:
            self.assertIn(field, str(ctx.exception))

    def test_subject_id_optional(self) -> None:
        """subject_id is optional -> None when the core file omits it."""
        _write_input(
            self.root,
            "session.json",
            {
                "rig_id": "Bergamo",
                "session_start_time": "2025-05-17T10:00:00-07:00",
                "data_streams": [
                    {"ophys_fovs": [{"index": 0, "frame_rate": "30.0"}]}
                ],
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


class TestAssetFixtures(unittest.TestCase):
    """Verify converter metadata against synthetic and production assets."""

    def _load_v1_and_v2(
        self, fixture_name: str
    ) -> tuple[Path, ConverterMetadata]:
        """Load both schema views and require equivalent converter metadata."""
        root = FIXTURES / fixture_name
        v1 = load_converter_metadata(root / "v1")
        v2 = load_converter_metadata(root / "v2")
        self.assertEqual(
            replace(v1, plane_channels=()), replace(v2, plane_channels=())
        )
        self.assertTrue((root / "v2" / "acquisition.json").is_file())
        self.assertTrue((root / "v2" / "data_description.json").is_file())
        return root, v1

    def _assert_meso_fixture(
        self,
        fixture_name: str,
        instrument_id: str,
        dataset_name: str,
        session_id: int,
        parent_session: str,
        is_child: bool,
    ) -> None:
        """Check core and platform metadata from a real Meso asset."""
        root, metadata = self._load_v1_and_v2(fixture_name)
        self.assertEqual(metadata.instrument_id, instrument_id)
        self.assertEqual(metadata.raw_format, "mesoscope_aind")
        self.assertEqual(metadata.dataset_name, dataset_name)
        self.assertEqual(len(metadata.fov_ids), 8)
        self.assertEqual(metadata.fov_ids[0], "VISp_0")

        platform_paths = tuple((root / "v1" / "pophys").glob("*platform.json"))
        self.assertEqual(len(platform_paths), 1)
        platform_path = platform_paths[0]
        self.assertEqual(
            is_child_session_via_platform_json(platform_path), is_child
        )
        platform = json.loads(platform_path.read_text())
        self.assertEqual(platform["session_id"], session_id)
        self.assertEqual(platform["parent_session"], parent_session)

        self.assertTrue((root / "v1" / "session.json").is_file())
        self.assertTrue((root / "v1" / "data_description.json").is_file())

    def test_parent_meso_fixture_uses_real_metadata(self) -> None:
        """The 'parent' fixture is a true root session (no parent_session).

        It exercises the non-child converter branch: MESO.1, and
        ``is_child`` is False, so depth pairing is skipped.
        """
        self._assert_meso_fixture(
            "meso_parent",
            "MESO.1",
            "multiplane-ophys_849375_2026-05-20_12-10-09",
            1779302846,
            None,
            is_child=False,
        )

    def test_child_meso_fixture_uses_real_metadata(self) -> None:
        """Load child Meso metadata from Code Ocean data."""
        self._assert_meso_fixture(
            "meso_child",
            "MESO.2",
            "multiplane-ophys_849375_2026-06-20_13-13-04",
            1781984614,
            "multiplane-ophys_849375_2026-06-17_15-28-08",
            is_child=True,
        )

    def test_child_meso_fixture_plane_depths(self) -> None:
        """Plane depths follow plane-index order in both schema versions."""
        _, metadata = self._load_v1_and_v2("meso_child")
        self.assertEqual(
            metadata.plane_depths,
            (156.0, 172.0, 116.0, 202.0, 76.0, 230.0, 36.0, 276.0),
        )

    def test_child_meso_fixture_plane_channels(self) -> None:
        """Channels and coupled partners follow plane-index order.

        The v1 session records no channel and a coupled-FOV group index; the
        upgraded v2 acquisition records the generic channel and each plane's
        partner.
        """
        v1 = load_converter_metadata(FIXTURES / "meso_child" / "v1")
        v2 = load_converter_metadata(FIXTURES / "meso_child" / "v2")
        self.assertEqual(
            [p.plane_index for p in v2.plane_channels], list(range(8))
        )
        self.assertEqual({p.channel_name for p in v1.plane_channels}, {None})
        self.assertEqual(
            {p.channel_name for p in v2.plane_channels}, {"Ophys Channel"}
        )
        self.assertEqual(
            [p.coupled_plane_index for p in v2.plane_channels],
            [1, 0, 3, 2, 5, 4, 7, 6],
        )

    def test_bergamo_fixture_v1_and_v2(self) -> None:
        """Both views preserve fields recovered from the DocDB record."""
        _, metadata = self._load_v1_and_v2("bergamo")
        self.assertEqual(metadata.instrument_id, "442_Bergamo_2p_photostim")
        self.assertEqual(metadata.raw_format, "tiff_stack")
        self.assertEqual(
            metadata.dataset_name,
            "single-plane-ophys_772414_2025-04-11_17-41-16",
        )
        self.assertEqual(metadata.subject_id, "772414")
        self.assertEqual(metadata.fov_ids, ("plane_0",))
        self.assertEqual(
            metadata.epoch_stems,
            ("spont", "neuron8_to_17_again_again", "spontpost"),
        )


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

    def test_identity_is_the_installed_library(self) -> None:
        """name, version and url all describe this library's package."""
        code = build_converter_code(self._params())
        self.assertEqual(code.name, LIBRARY)
        self.assertEqual(code.version, version(LIBRARY))
        self.assertEqual(
            code.url, f"https://github.com/AllenNeuralDynamics/{LIBRARY}"
        )

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
