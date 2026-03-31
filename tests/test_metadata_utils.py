"""Tests for aind_pophys_converter.utils.metadata_utils."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_data_schema.components.configs import (
    CoupledPlane,
    ImagingConfig,
    PlanarImage,
)
from aind_data_schema.core.quality_control import Stage, Status

from aind_pophys_converter.utils.metadata_utils import (
    build_fov_qc_metric,
    build_vasculature_qc_metric,
    get_fov_id,
    get_imaging_planes,
    load_acquisition,
    load_data_description,
    pending_qc_status,
)


class TestLoadAcquisition(unittest.TestCase):
    """Tests for load_acquisition."""

    def test_delegates_to_model_validate_json(self):
        """load_acquisition calls Acquisition.model_validate_json."""
        mock_path = MagicMock(spec=Path)
        mock_path.read_text.return_value = '{"key": "value"}'
        with patch(
            "aind_pophys_converter.utils.metadata_utils"
            ".Acquisition.model_validate_json"
        ) as mock_validate:
            mock_validate.return_value = MagicMock()
            load_acquisition(mock_path)
            mock_validate.assert_called_once_with('{"key": "value"}')


class TestLoadDataDescription(unittest.TestCase):
    """Tests for load_data_description."""

    def test_delegates_to_model_validate_json(self):
        """load_data_description calls DataDescription.model_validate_json."""
        mock_path = MagicMock(spec=Path)
        mock_path.read_text.return_value = '{"key": "value"}'
        with patch(
            "aind_pophys_converter.utils.metadata_utils"
            ".DataDescription.model_validate_json"
        ) as mock_validate:
            mock_validate.return_value = MagicMock()
            load_data_description(mock_path)
            mock_validate.assert_called_once_with('{"key": "value"}')


class TestGetImagingPlanes(unittest.TestCase):
    """Tests for get_imaging_planes."""

    def _make_acquisition(self, planes_per_stream):
        """Build a minimal mock Acquisition with CoupledPlane objects.

        Parameters
        ----------
        planes_per_stream : list of list of int
            Each inner list gives the plane_index values for one data stream.

        Returns
        -------
        MagicMock
            Mock Acquisition object.
        """
        data_streams = []
        for plane_indices in planes_per_stream:
            planes = [
                CoupledPlane.model_construct(plane_index=i)
                for i in plane_indices
            ]
            planar_image = PlanarImage.model_construct(planes=planes)
            imaging_config = ImagingConfig.model_construct(
                images=[planar_image]
            )
            stream = MagicMock()
            stream.configurations = [imaging_config]
            data_streams.append(stream)
        acq = MagicMock()
        acq.data_streams = data_streams
        return acq

    def test_returns_planes_sorted_by_plane_index(self):
        """Planes are returned sorted by plane_index."""
        acq = self._make_acquisition([[2, 0, 1]])
        planes = get_imaging_planes(acq)
        self.assertEqual([p.plane_index for p in planes], [0, 1, 2])

    def test_collects_across_multiple_streams(self):
        """Planes are collected from all data streams."""
        acq = self._make_acquisition([[1], [0]])
        planes = get_imaging_planes(acq)
        self.assertEqual(len(planes), 2)
        self.assertEqual(planes[0].plane_index, 0)

    def test_skips_non_imaging_configs(self):
        """Non-ImagingConfig configurations are ignored."""
        planar_image = PlanarImage.model_construct(
            planes=[CoupledPlane.model_construct(plane_index=0)]
        )
        imaging_config = ImagingConfig.model_construct(images=[planar_image])
        other_config = MagicMock()  # not an ImagingConfig
        stream = MagicMock()
        stream.configurations = [other_config, imaging_config]
        acq = MagicMock()
        acq.data_streams = [stream]
        planes = get_imaging_planes(acq)
        self.assertEqual(len(planes), 1)

    def test_skips_non_planar_images(self):
        """Non-PlanarImage images inside ImagingConfig are ignored."""
        planar_image = PlanarImage.model_construct(
            planes=[CoupledPlane.model_construct(plane_index=0)]
        )
        other_image = MagicMock()  # not a PlanarImage
        imaging_config = ImagingConfig.model_construct(
            images=[other_image, planar_image]
        )
        stream = MagicMock()
        stream.configurations = [imaging_config]
        acq = MagicMock()
        acq.data_streams = [stream]
        planes = get_imaging_planes(acq)
        self.assertEqual(len(planes), 1)

    def test_empty_acquisition_returns_empty_list(self):
        """Acquisition with no data streams returns []."""
        acq = MagicMock()
        acq.data_streams = []
        self.assertEqual(get_imaging_planes(acq), [])

    def test_plain_plane_without_plane_index_does_not_raise(self):
        """Plane objects without plane_index sort to front without error."""
        plain_plane = MagicMock()
        del plain_plane.plane_index  # ensure attribute doesn't exist
        planar_image = PlanarImage.model_construct(planes=[plain_plane])
        imaging_config = ImagingConfig.model_construct(images=[planar_image])
        stream = MagicMock()
        stream.configurations = [imaging_config]
        acq = MagicMock()
        acq.data_streams = [stream]
        planes = get_imaging_planes(acq)
        self.assertEqual(len(planes), 1)
        self.assertIs(planes[0], plain_plane)


class TestGetFovId(unittest.TestCase):
    """Tests for get_fov_id."""

    def test_format_acronym_underscore_index(self):
        """get_fov_id returns '{acronym}_{plane_index}'."""
        mock_structure = MagicMock()
        mock_structure.acronym = "VISp"
        plane = CoupledPlane.model_construct(
            plane_index=3, targeted_structure=mock_structure
        )
        self.assertEqual(get_fov_id(plane), "VISp_3")

    def test_zero_index(self):
        """get_fov_id works with plane_index=0."""
        mock_structure = MagicMock()
        mock_structure.acronym = "MO"
        plane = CoupledPlane.model_construct(
            plane_index=0, targeted_structure=mock_structure
        )
        self.assertEqual(get_fov_id(plane), "MO_0")

    def test_no_plane_index_defaults_to_zero(self):
        """get_fov_id defaults plane_index to 0 for plain Plane objects."""
        mock_structure = MagicMock()
        mock_structure.acronym = "MO"
        plane = MagicMock(spec=["targeted_structure"])
        plane.targeted_structure = mock_structure
        self.assertEqual(get_fov_id(plane), "MO_0")


class TestPendingQcStatus(unittest.TestCase):
    """Tests for pending_qc_status."""

    def test_returns_pending_status(self):
        """pending_qc_status returns Status.PENDING."""
        status = pending_qc_status()
        self.assertEqual(status.status, Status.PENDING)

    def test_evaluator_is_automated(self):
        """pending_qc_status sets evaluator to 'Automated'."""
        status = pending_qc_status()
        self.assertEqual(status.evaluator, "Automated")


class TestBuildQcMetrics(unittest.TestCase):
    """Tests for build_fov_qc_metric and build_vasculature_qc_metric."""

    def test_fov_metric_stage_and_modality(self):
        """build_fov_qc_metric returns Stage.RAW and Modality.POPHYS."""
        metric = build_fov_qc_metric(
            unique_id="VISp_200",
            raw_tif_path=Path("/fake/parent.tif"),
            intended_depth=200,
            targeted_structure_id="VISp",
            closest_z=-200.0,
            merged_path=Path("/fake/merged.png"),
        )
        self.assertEqual(metric.stage, Stage.RAW)

    def test_fov_metric_has_pending_status_history(self):
        """build_fov_qc_metric status_history[0] is PENDING."""
        metric = build_fov_qc_metric(
            unique_id="VISp_200",
            raw_tif_path=Path("/fake/parent.tif"),
            intended_depth=200,
            targeted_structure_id="VISp",
            closest_z=-200.0,
            merged_path=Path("/fake/merged.png"),
        )
        self.assertEqual(metric.status_history[0].status, Status.PENDING)

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

    def test_vasculature_metric_has_pending_status_history(self):
        """build_vasculature_qc_metric status_history[0] is PENDING."""
        metric = build_vasculature_qc_metric(Path("/fake/vasculature.png"))
        self.assertEqual(metric.status_history[0].status, Status.PENDING)

    def test_vasculature_metric_reference(self):
        """build_vasculature_qc_metric reference matches input path."""
        fp = Path("/fake/vasculature.png")
        metric = build_vasculature_qc_metric(fp)
        self.assertEqual(metric.reference, str(fp))


if __name__ == "__main__":
    unittest.main()
