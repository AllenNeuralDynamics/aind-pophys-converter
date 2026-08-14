"""Tests for the converter's settings and logging identity helpers."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

from aind_pophys_converter.settings import (
    DATA_DESCRIPTION_FILE,
    PIPELINE_NAME_ENV,
    PROCESS_NAME,
    JobSettings,
    acquisition_name_from,
    build_log_fields,
)


class TestAcquisitionName(unittest.TestCase):
    """Reading the acquisition name from the raw asset directory."""

    def test_reads_data_description_name(self):
        """data_description.json is canonical."""
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp) / "multiplane-ophys_1_2026-01-01_00-00-00"
            asset_dir.mkdir()
            (asset_dir / DATA_DESCRIPTION_FILE).write_text(
                json.dumps({"name": "multiplane-ophys_1_2026-01-01_00-00-00"})
            )
            self.assertEqual(
                acquisition_name_from(asset_dir),
                "multiplane-ophys_1_2026-01-01_00-00-00",
            )

    def test_falls_back_to_the_directory_name(self):
        """A raw asset directory is named after its acquisition."""
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp) / "bergamo_2_2026-02-02_00-00-00"
            asset_dir.mkdir()
            self.assertEqual(
                acquisition_name_from(asset_dir),
                "bergamo_2_2026-02-02_00-00-00",
            )

    def test_unreadable_document_falls_back(self):
        """Malformed JSON is not fatal to logging setup."""
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp) / "asset"
            asset_dir.mkdir()
            (asset_dir / DATA_DESCRIPTION_FILE).write_text("{not json")
            self.assertEqual(acquisition_name_from(asset_dir), "asset")

    def test_empty_name_falls_back(self):
        """An empty name is treated as absent."""
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp) / "asset"
            asset_dir.mkdir()
            (asset_dir / DATA_DESCRIPTION_FILE).write_text(
                json.dumps({"name": ""})
            )
            self.assertEqual(acquisition_name_from(asset_dir), "asset")


class TestLogFields(unittest.TestCase):
    """Static fields attached to every structured log line."""

    @patch.dict(os.environ, {}, clear=True)
    def test_outside_a_pipeline_omits_pipeline_name(self):
        """A standalone capsule run has no pipeline to name."""
        fields = build_log_fields("acq")
        self.assertEqual(
            fields, {"acquisition_name": "acq", "process_name": PROCESS_NAME}
        )

    @patch.dict(
        os.environ, {PIPELINE_NAME_ENV: "aind-pophys-pipeline"}, clear=True
    )
    def test_inside_a_pipeline_records_the_pipeline_name(self):
        """The pipeline runtime exports the name; logs must carry it."""
        self.assertEqual(
            build_log_fields("acq"),
            {
                "acquisition_name": "acq",
                "process_name": PROCESS_NAME,
                "pipeline_name": "aind-pophys-pipeline",
            },
        )


class TestJobSettings(unittest.TestCase):
    """Tests for JobSettings."""

    def _make_settings(
        self, extra_argv: Optional[List[str]] = None
    ) -> "JobSettings":
        """Instantiate JobSettings with a controlled sys.argv."""
        argv = ["prog", "--input_dir", "/in", "--output_dir", "/out"]
        if extra_argv:
            argv.extend(extra_argv)
        with patch.object(sys, "argv", argv):
            return JobSettings()

    def test_required_fields(self) -> None:
        """input_dir and output_dir are set from argv."""
        s = self._make_settings()
        self.assertEqual(str(s.input_dir), "/in")
        self.assertEqual(str(s.output_dir), "/out")

    def test_default_temp_dir_is_none(self) -> None:
        """temp_dir defaults to None."""
        s = self._make_settings()
        self.assertIsNone(s.temp_dir)

    def test_default_debug_is_false(self) -> None:
        """debug defaults to False."""
        s = self._make_settings()
        self.assertFalse(s.debug)

    def test_debug_flag(self) -> None:
        """debug is set from argv."""
        s = self._make_settings(["--debug", "True"])
        self.assertTrue(s.debug)

    def test_temp_dir_set(self) -> None:
        """temp_dir is converted to Path when provided."""
        s = self._make_settings(["--temp_dir", "/tmp/mytemp"])
        self.assertEqual(s.temp_dir, Path("/tmp/mytemp"))

    def test_input_dir_mutation(self) -> None:
        """input_dir can be reassigned (used in run())."""
        s = self._make_settings()
        new_path = Path("/scratch/data/pophys")
        s.input_dir = new_path
        self.assertEqual(s.input_dir, new_path)


if __name__ == "__main__":
    unittest.main()
