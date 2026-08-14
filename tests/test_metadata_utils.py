"""Focused tests for the converter's three supported input forms."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from aind_pophys_converter.utils.metadata_utils import (
    ConverterMetadata,
    load_converter_metadata,
)
from aind_pophys_converter.utils.platform_utils import (
    is_child_session_via_platform_json,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestMetadataInputs(unittest.TestCase):
    """Verify the adapter delegates v1/v2/minimal dispatch to the library."""

    def write_input(self, name: str, document: dict) -> Path:
        """Write one core document and return its containing directory."""
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(temp_dir))
        (temp_dir / name).write_text(json.dumps(document))
        return temp_dir

    def test_v1_session(self) -> None:
        """A v1 session remains a supported input."""
        root = self.write_input(
            "session.json",
            {
                "rig_id": "MESO.1",
                "data_streams": [
                    {
                        "ophys_fovs": [
                            {"index": 0, "targeted_structure": "VISp"}
                        ]
                    }
                ],
            },
        )
        metadata = load_converter_metadata(root)
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertEqual(metadata.fov_ids, ("plane_0",))

    def test_v2_acquisition(self) -> None:
        """A v2 acquisition is dispatched by filename."""
        root = self.write_input(
            "acquisition.json",
            {"instrument_id": "MESO.1", "data_streams": []},
        )
        metadata = load_converter_metadata(root)
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertEqual(metadata.fov_ids, ())

    def test_minimal_metadata(self) -> None:
        """A minimal metadata.json supplies the converter's core fields."""
        root = self.write_input(
            "metadata.json",
            {
                "instrument_id": "MESO.1",
                "dataset_name": "minimal-input",
                "planes": [{"plane_index": 0, "structure": "VISp"}],
            },
        )
        metadata = load_converter_metadata(root)
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertEqual(metadata.fov_ids, ("plane_0",))
        self.assertEqual(metadata.dataset_name, "minimal-input")


class TestAssetFixtures(unittest.TestCase):
    """Verify converter metadata against synthetic and production assets."""

    def _load_v1_and_v2(
        self, fixture_name: str
    ) -> tuple[Path, ConverterMetadata]:
        """Load both schema views and require equivalent converter metadata."""
        root = FIXTURES / fixture_name
        v1 = load_converter_metadata(root / "v1")
        v2 = load_converter_metadata(root / "v2")
        self.assertEqual(v1, v2)
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

    def test_bergamo_fixture_v1_and_v2(self) -> None:
        """Both views preserve fields recovered from the DocDB record."""
        _, metadata = self._load_v1_and_v2("bergamo")
        self.assertEqual(metadata.instrument_id, "442_Bergamo_2p_photostim")
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
