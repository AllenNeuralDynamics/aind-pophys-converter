"""Focused tests for the schema-v2 QC contract."""

import tempfile
import unittest
from pathlib import Path

from aind_data_schema.core.quality_control import Status

from aind_pophys_converter.qc import (
    build_quality_control,
    build_vasculature_qc_metric,
    write_quality_control_json,
)


class TestQualityControl(unittest.TestCase):
    """The converter emits one full QC document with pending metrics."""

    def test_single_document_contract(self) -> None:
        """A metric is grouped and serialized as one standard file."""
        metric = build_vasculature_qc_metric(Path("vasculature.png"))
        self.assertEqual(metric.status_history[0].status, Status.PENDING)
        self.assertEqual(metric.value.value, "")
        document = build_quality_control([metric])
        self.assertEqual(document.default_grouping, ["evaluation"])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = write_quality_control_json(document, Path(temp_dir))
            self.assertEqual(output.name, "quality_control.json")
            self.assertTrue(output.exists())
