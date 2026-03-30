"""Tests for qc module."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_data_schema_models.modalities import Modality

from aind_pophys_converter.qc import (
    PendingStatus,
    write_fov_metric,
    write_vasculature_metric,
)


class TestPendingStatus(unittest.TestCase):
    """Tests for PendingStatus factory."""

    def test_returns_qcstatus(self):
        """PendingStatus returns a QCStatus instance."""
        s = PendingStatus()
        self.assertIsInstance(s, QCStatus)

    def test_status_is_pending(self):
        """PendingStatus status field is PENDING."""
        s = PendingStatus()
        self.assertEqual(s.status, Status.PENDING)

    def test_evaluator_is_automated(self):
        """PendingStatus evaluator is 'Automated'."""
        s = PendingStatus()
        self.assertEqual(s.evaluator, "Automated")

    def test_timestamp_is_datetime(self):
        """PendingStatus timestamp is a timezone-aware datetime."""
        s = PendingStatus()
        self.assertIsInstance(s.timestamp, datetime)
        self.assertIsNotNone(s.timestamp.tzinfo)


class TestWriteFovMetric(unittest.TestCase):
    """Tests for write_fov_metric."""

    def setUp(self):
        """Set up temporary directory."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up."""
        self.tmpdir.cleanup()

    def _write(self, unique_id="VISp_200"):
        """Helper to call write_fov_metric with minimal args."""
        raw_tif_path = self.output_dir / "1234_200_VISp_depth.tif"
        merged_path = self.output_dir / "merged.png"
        write_fov_metric(
            unique_id=unique_id,
            raw_tif_path=raw_tif_path,
            intended_depth=200,
            targeted_structure_id="VISp",
            closest_z=276.0,
            merged_path=merged_path,
            output_dir=self.output_dir,
        )
        return self.output_dir / f"{unique_id}_fov_metric.json"

    def test_file_is_created(self):
        """write_fov_metric creates a JSON file."""
        path = self._write()
        self.assertTrue(path.exists())

    def test_file_is_valid_json(self):
        """The written file deserializes without error."""
        path = self._write()
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_file_validates_as_qcmetric(self):
        """The written JSON round-trips through QCMetric.model_validate."""
        path = self._write()
        with open(path) as f:
            data = json.load(f)
        metric = QCMetric.model_validate(data)
        self.assertIsInstance(metric, QCMetric)

    def test_metric_name_contains_unique_id(self):
        """Metric name includes the unique_id."""
        path = self._write("VISl_300")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("VISl_300", data["name"])

    def test_metric_modality_is_pophys(self):
        """Metric modality is POPHYS."""
        path = self._write()
        with open(path) as f:
            data = json.load(f)
        metric = QCMetric.model_validate(data)
        self.assertEqual(metric.modality, Modality.POPHYS)

    def test_metric_tags(self):
        """Metric tags include evaluation and type keys."""
        path = self._write()
        with open(path) as f:
            data = json.load(f)
        self.assertIn("evaluation", data["tags"])
        self.assertIn("type", data["tags"])

    def test_filename_uses_unique_id(self):
        """Output filename is {unique_id}_fov_metric.json."""
        self._write("VISp_50")
        self.assertTrue((self.output_dir / "VISp_50_fov_metric.json").exists())


class TestWriteVasculatureMetric(unittest.TestCase):
    """Tests for write_vasculature_metric."""

    def setUp(self):
        """Set up temporary directory."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
        self.vasculature_fp = self.output_dir / "vasculature.png"
        self.vasculature_fp.touch()

    def tearDown(self):
        """Clean up."""
        self.tmpdir.cleanup()

    def test_file_is_created(self):
        """write_vasculature_metric creates vasculature_metric.json."""
        write_vasculature_metric(self.vasculature_fp, self.output_dir)
        self.assertTrue((self.output_dir / "vasculature_metric.json").exists())

    def test_file_validates_as_qcmetric(self):
        """The written JSON round-trips through QCMetric.model_validate."""
        write_vasculature_metric(self.vasculature_fp, self.output_dir)
        with open(self.output_dir / "vasculature_metric.json") as f:
            data = json.load(f)
        metric = QCMetric.model_validate(data)
        self.assertIsInstance(metric, QCMetric)

    def test_metric_name(self):
        """Metric name is 'Vasculature Image'."""
        write_vasculature_metric(self.vasculature_fp, self.output_dir)
        with open(self.output_dir / "vasculature_metric.json") as f:
            data = json.load(f)
        self.assertEqual(data["name"], "Vasculature Image")

    def test_metric_modality_is_pophys(self):
        """Metric modality is POPHYS."""
        write_vasculature_metric(self.vasculature_fp, self.output_dir)
        with open(self.output_dir / "vasculature_metric.json") as f:
            data = json.load(f)
        metric = QCMetric.model_validate(data)
        self.assertEqual(metric.modality, Modality.POPHYS)

    def test_pass_fail_option_count(self):
        """Dropdown has 8 options and 8 statuses."""
        write_vasculature_metric(self.vasculature_fp, self.output_dir)
        with open(self.output_dir / "vasculature_metric.json") as f:
            data = json.load(f)
        value = data["value"]
        self.assertEqual(len(value["options"]), 8)
        self.assertEqual(len(value["status"]), 8)


if __name__ == "__main__":
    unittest.main()
