"""Tests for the converter's logging identity helpers."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aind_pophys_converter.settings import (
    DATA_DESCRIPTION_FILE,
    PIPELINE_NAME_ENV,
    PROCESS_NAME,
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


if __name__ == "__main__":
    unittest.main()
