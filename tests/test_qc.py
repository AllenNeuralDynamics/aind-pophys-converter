"""Tests for QC generation (qc.py: metrics + QualityControl assembly)."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality

from aind_pophys_converter.qc import (
    build_fov_qc_metric,
    build_quality_control,
    build_vasculature_qc_metric,
    pending_qc_status,
    write_quality_control_json,
)

FOV_DESCRIPTION = (
    "Parent session depth image beside this session's averaged image of the "
    "same structure and intended depth. Pass if the two fields of view match."
)
VASCULATURE_DESCRIPTION = (
    "Vasculature image for judging brain surface health and window clarity."
)


class TestBuildQcMetrics(unittest.TestCase):
    """Tests for build_fov_qc_metric and build_vasculature_qc_metric."""

    def test_fov_metric_stage(self) -> None:
        """build_fov_qc_metric returns Stage.RAW."""
        metric = build_fov_qc_metric(
            unique_id="VISp_200",
            raw_tif_path=Path("/fake/parent.tif"),
            intended_depth=200,
            targeted_structure_id="VISp",
            merged_path=Path("/fake/merged.png"),
        )
        self.assertEqual(metric.stage, Stage.RAW)

    def test_fov_metric_modality_is_pophys(self) -> None:
        """build_fov_qc_metric modality is POPHYS."""
        metric = build_fov_qc_metric(
            unique_id="VISp_200",
            raw_tif_path=Path("/fake/parent.tif"),
            intended_depth=200,
            targeted_structure_id="VISp",
            merged_path=Path("/fake/merged.png"),
        )
        self.assertEqual(metric.modality, Modality.POPHYS)

    def test_fov_metric_name_restores_um_unit(self) -> None:
        """build_fov_qc_metric name is the unique_id plus the um unit."""
        metric = build_fov_qc_metric(
            unique_id="MOp_100",
            raw_tif_path=Path("/fake/parent.tif"),
            intended_depth=100,
            targeted_structure_id="MOp",
            merged_path=Path("/fake/merged.png"),
        )
        self.assertEqual(metric.name, "MOp_100um Parent-Child FOV")

    def test_fov_metric_group_and_type(self) -> None:
        """build_fov_qc_metric is Operational QC in the FOV matching group."""
        metric = build_fov_qc_metric(
            unique_id="VISp_200",
            raw_tif_path=Path("/fake/parent.tif"),
            intended_depth=200,
            targeted_structure_id="VISp",
            merged_path=Path("/fake/merged.png"),
        )
        self.assertEqual(
            metric.tags,
            {"evaluation": "Field-of-view Matching", "type": "Operational QC"},
        )

    def test_fov_metric_description_is_independent_of_the_run(self) -> None:
        """build_fov_qc_metric description holds no per-plane values."""
        first = build_fov_qc_metric(
            unique_id="VISp_160",
            raw_tif_path=Path("/fake/1781731432_160_VISp_depth.tif"),
            intended_depth=160,
            targeted_structure_id="VISp",
            merged_path=Path("/fake/merged.png"),
        )
        second = build_fov_qc_metric(
            unique_id="MOp_100",
            raw_tif_path=Path("/fake/1781731433_100_MOp_depth.tif"),
            intended_depth=100,
            targeted_structure_id="MOp",
            merged_path=Path("/fake/merged.png"),
        )
        self.assertEqual(first.description, FOV_DESCRIPTION)
        self.assertEqual(second.description, FOV_DESCRIPTION)

    def test_vasculature_metric_name_group_and_type(self) -> None:
        """build_vasculature_qc_metric is Operational QC, Window Clarity."""
        metric = build_vasculature_qc_metric(Path("/fake/vasculature.png"))
        self.assertEqual(metric.name, "Vasculature Image")
        self.assertEqual(
            metric.tags,
            {"evaluation": "Window Clarity", "type": "Operational QC"},
        )

    def test_vasculature_metric_description(self) -> None:
        """build_vasculature_qc_metric carries the agreed description."""
        metric = build_vasculature_qc_metric(Path("/fake/vasculature.png"))
        self.assertEqual(metric.description, VASCULATURE_DESCRIPTION)

    def test_vasculature_metric_stage(self) -> None:
        """build_vasculature_qc_metric returns Stage.RAW."""
        metric = build_vasculature_qc_metric(Path("/fake/vasculature.png"))
        self.assertEqual(metric.stage, Stage.RAW)

    def test_vasculature_metric_reference(self) -> None:
        """build_vasculature_qc_metric reference matches input path."""
        fp = Path("/fake/vasculature.png")
        self.assertEqual(build_vasculature_qc_metric(fp).reference, str(fp))


class TestPendingQcStatus(unittest.TestCase):
    """Tests for pending_qc_status helper."""

    def test_returns_qcstatus(self) -> None:
        """pending_qc_status returns a QCStatus instance."""
        s = pending_qc_status()
        self.assertIsInstance(s, QCStatus)

    def test_status_is_pending(self) -> None:
        """pending_qc_status status field is PENDING."""
        s = pending_qc_status()
        self.assertEqual(s.status, Status.PENDING)

    def test_evaluator_is_automated(self) -> None:
        """pending_qc_status evaluator is 'Automated'."""
        s = pending_qc_status()
        self.assertEqual(s.evaluator, "Automated")

    def test_timestamp_is_datetime(self) -> None:
        """pending_qc_status timestamp is a timezone-aware datetime."""
        s = pending_qc_status()
        self.assertIsInstance(s.timestamp, datetime)
        self.assertIsNotNone(s.timestamp.tzinfo)


def _sample_metrics() -> list:
    """Build one FOV metric and one vasculature metric."""
    fov = build_fov_qc_metric(
        unique_id="VISp_200",
        raw_tif_path=Path("/fake/parent.tif"),
        intended_depth=200,
        targeted_structure_id="VISp",
        merged_path=Path("/fake/merged.png"),
    )
    vasc = build_vasculature_qc_metric(Path("/fake/vasculature.png"))
    return [fov, vasc]


class TestBuildQualityControl(unittest.TestCase):
    """Tests for build_quality_control."""

    def test_returns_quality_control(self) -> None:
        """build_quality_control returns a QualityControl object."""
        qc = build_quality_control(_sample_metrics())
        self.assertIsInstance(qc, QualityControl)

    def test_contains_all_metrics(self) -> None:
        """All supplied metrics are retained."""
        metrics = _sample_metrics()
        qc = build_quality_control(metrics)
        self.assertEqual(len(qc.metrics), len(metrics))

    def test_metrics_are_qcmetrics(self) -> None:
        """Wrapped metrics are QCMetric instances."""
        qc = build_quality_control(_sample_metrics())
        for m in qc.metrics:
            self.assertIsInstance(m, QCMetric)

    def test_default_grouping_set(self) -> None:
        """default_grouping (required in v2.8) is populated."""
        qc = build_quality_control(_sample_metrics())
        self.assertTrue(qc.default_grouping)
        self.assertIn("evaluation", qc.default_grouping)

    def test_round_trips_through_json(self) -> None:
        """QualityControl survives a full JSON round-trip + validation."""
        qc = build_quality_control(_sample_metrics())
        QualityControl.model_validate_json(qc.model_dump_json())


class TestWriteQualityControlJson(unittest.TestCase):
    """Tests for write_quality_control_json."""

    def setUp(self) -> None:
        """Per-test scratch directory."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        """Clean up."""
        self.tmpdir.cleanup()

    def test_writes_single_quality_control_file(self) -> None:
        """Writes exactly one quality_control.json (not per-metric files)."""
        qc = build_quality_control(_sample_metrics())
        written = write_quality_control_json(qc, self.output_dir)
        self.assertTrue(written.exists())
        self.assertEqual(written.name, "quality_control.json")
        # No per-metric JSON fragments
        self.assertEqual(list(self.output_dir.glob("*_metric.json")), [])

    def test_written_file_validates(self) -> None:
        """The written file validates as a QualityControl."""
        qc = build_quality_control(_sample_metrics())
        written = write_quality_control_json(qc, self.output_dir)
        data = json.loads(written.read_text())
        self.assertIn("schema_version", data)
        QualityControl.model_validate(data)


if __name__ == "__main__":
    unittest.main()
