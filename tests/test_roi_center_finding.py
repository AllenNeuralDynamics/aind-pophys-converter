"""Test suite for ROI center finding functions in mesoscope splitter module."""

import unittest
from typing import Tuple

import numpy as np

from aind_pophys_converter.tiff_splitters import (
    build_roi_remap,
    get_nearest_roi_center,
    get_valid_roi_centers,
    validate_roi_pairing,
)


class TestRoiCenterFinding(unittest.TestCase):
    """Test suite for ROI center finding functions."""

    def test_get_nearest_valid_roi_center(self) -> None:
        """
        Test that get_nearest_valid_roi_center correctly maps
        a candidate roi_center to its nearest valid_roi_center
        """
        valid_centers = [(1.1, 2.1), (1.1, 2.2)]

        roi_list = [(1.1, 1.9), (1.1, 2.14), (1.1, 2.16)]
        expected = [0, 0, 1]
        for this_roi, this_expected in zip(roi_list, expected):
            self.assertEqual(
                this_expected,
                get_nearest_roi_center(
                    this_roi_center=this_roi, valid_roi_centers=valid_centers
                ),
            )

        with self.assertRaises(RuntimeError) as cm:
            get_nearest_roi_center(
                this_roi_center=roi_list[0], valid_roi_centers=[]
            )
        self.assertIn("Could not find nearest ROI", str(cm.exception))

    def test_get_valid_roi_centers(self) -> None:
        """
        Test that get_valid_roi_centers returns the correct list of
        distinct ROI centers
        """

        class MockTimeSeriesSplitter(object):
            """Mock class to simulate TimeSeriesSplitter behavior."""

            def __init__(self) -> None:
                """Initialize mock data."""
                manifest = []
                center_lookup = dict()
                for i_roi, zz, xx, yy in zip(
                    (1, 2, 3, 4),
                    (5, 6, 7, 8),
                    (1.1, 1.10001, 1.1, 3.1),
                    (4.5, 4.5, 4.5001, 2.4),
                ):
                    pair = (i_roi, zz)
                    manifest.append(pair)
                    center = (xx, yy)
                    center_lookup[i_roi] = center
                self.roi_z_int_manifest = manifest
                self.center_lookup = center_lookup

            def roi_center(self, i_roi: int) -> Tuple[float, float]:
                """
                Mock method to return the center of a given ROI.
                """
                return self.center_lookup[i_roi]

        splitter = MockTimeSeriesSplitter()
        valid_centers = get_valid_roi_centers(splitter)
        self.assertEqual(len(valid_centers), 2)
        np.testing.assert_allclose(
            np.array([1.1, 4.5]), np.array(valid_centers[0])
        )
        np.testing.assert_allclose(
            np.array([3.1, 2.4]), np.array(valid_centers[1])
        )


class MockSplitter:
    """Mock splitter with configurable ROI centers."""

    def __init__(self, centers: dict) -> None:
        """Initialize with a dict of ``{roi_index: (x, y) center}``."""
        self._centers = centers
        self._n_rois = len(centers)

    @property
    def n_rois(self) -> int:
        """Return the number of ROIs."""
        return self._n_rois

    def roi_center(self, i_roi: int) -> Tuple[float, float]:
        """Return the (x, y) center for the given ROI index."""
        return self._centers[i_roi]


class TestValidateRoiPairing(unittest.TestCase):
    """Test suite for validate_roi_pairing."""

    def test_matching_centers_same_order(self) -> None:
        """Depth and surface with same centers in same order passes."""
        depth = MockSplitter({0: (1.0, 2.0), 1: (5.0, 6.0)})
        surface = MockSplitter({0: (1.0, 2.0), 1: (5.0, 6.0)})
        validate_roi_pairing(depth, surface)

    def test_matching_centers_swapped_order(self) -> None:
        """Depth and surface with centers in swapped order passes."""
        depth = MockSplitter({0: (1.0, 2.0), 1: (5.0, 6.0)})
        surface = MockSplitter({0: (5.05, 6.05), 1: (1.05, 2.05)})
        validate_roi_pairing(depth, surface, tolerance=1.0)

    def test_count_mismatch_raises(self) -> None:
        """Different number of ROIs raises RuntimeError."""
        depth = MockSplitter({0: (1.0, 2.0), 1: (5.0, 6.0)})
        surface = MockSplitter({0: (1.0, 2.0)})
        with self.assertRaises(RuntimeError) as cm:
            validate_roi_pairing(depth, surface)
        self.assertIn("ROI count mismatch", str(cm.exception))

    def test_no_match_within_tolerance_raises(self) -> None:
        """Surface ROI too far from any depth ROI raises RuntimeError."""
        depth = MockSplitter({0: (1.0, 2.0), 1: (5.0, 6.0)})
        surface = MockSplitter({0: (1.0, 2.0), 1: (50.0, 60.0)})
        with self.assertRaises(RuntimeError) as cm:
            validate_roi_pairing(depth, surface, tolerance=1.0)
        self.assertIn("no surface match", str(cm.exception))

    def test_small_drift_within_tolerance(self) -> None:
        """Small positional drift within tolerance passes."""
        depth = MockSplitter({0: (-6.706, 4.707), 1: (-4.475, -5.792)})
        surface = MockSplitter({0: (-4.599, -5.854), 1: (-6.835, 4.563)})
        validate_roi_pairing(depth, surface, tolerance=1.0)

    def test_single_roi(self) -> None:
        """Single ROI case works."""
        depth = MockSplitter({0: (3.0, 4.0)})
        surface = MockSplitter({0: (3.01, 4.01)})
        validate_roi_pairing(depth, surface, tolerance=1.0)


class TestBuildRoiRemap(unittest.TestCase):
    """Test suite for build_roi_remap."""

    def test_identity_remap(self) -> None:
        """When splitter order matches valid centers, remap is identity."""
        valid_centers = [(1.0, 2.0), (5.0, 6.0)]
        splitter = MockSplitter({0: (1.0, 2.0), 1: (5.0, 6.0)})
        remap = build_roi_remap(splitter, valid_centers)
        self.assertEqual(remap, {0: 0, 1: 1})

    def test_swapped_remap(self) -> None:
        """When splitter has swapped indices, remap corrects them."""
        valid_centers = [(1.0, 2.0), (5.0, 6.0)]
        splitter = MockSplitter({0: (5.05, 6.05), 1: (1.05, 2.05)})
        remap = build_roi_remap(splitter, valid_centers, tolerance=1.0)
        self.assertEqual(remap, {0: 1, 1: 0})

    def test_tolerance_exceeded_raises(self) -> None:
        """ROI too far from any valid center raises RuntimeError."""
        valid_centers = [(1.0, 2.0), (5.0, 6.0)]
        splitter = MockSplitter({0: (1.0, 2.0), 1: (50.0, 60.0)})
        with self.assertRaises(RuntimeError) as cm:
            build_roi_remap(splitter, valid_centers, tolerance=1.0)
        self.assertIn("exceeds", str(cm.exception))

    def test_explicit_roi_indices(self) -> None:
        """Using explicit roi_indices works for splitters without n_rois."""
        valid_centers = [(1.0, 2.0), (5.0, 6.0)]
        splitter = MockSplitter({0: (5.0, 6.0), 1: (1.0, 2.0)})
        remap = build_roi_remap(splitter, valid_centers, roi_indices=[0, 1])
        self.assertEqual(remap, {0: 1, 1: 0})

    def test_non_contiguous_roi_indices(self) -> None:
        """Works with non-contiguous roi indices (e.g. zstack)."""
        valid_centers = [(1.0, 2.0), (5.0, 6.0)]
        splitter = MockSplitter({2: (5.0, 6.0), 5: (1.0, 2.0)})
        remap = build_roi_remap(splitter, valid_centers, roi_indices=[2, 5])
        self.assertEqual(remap, {0: 5, 1: 2})

    def test_empty_valid_centers(self) -> None:
        """Empty valid centers produces empty remap."""
        splitter = MockSplitter({0: (1.0, 2.0)})
        remap = build_roi_remap(splitter, [])
        self.assertEqual(remap, {})

    def test_single_roi(self) -> None:
        """Single ROI remap works."""
        valid_centers = [(3.0, 4.0)]
        splitter = MockSplitter({0: (3.01, 4.01)})
        remap = build_roi_remap(splitter, valid_centers, tolerance=1.0)
        self.assertEqual(remap, {0: 0})

    def test_real_world_scenario(self) -> None:
        """Test with actual values from the VISp/VISl debug output."""
        valid_centers = [
            (-6.706054084, 4.706558895),
            (-4.474865807, -5.792409906),
        ]
        # Surface has swapped indices
        surface = MockSplitter(
            {
                0: (-4.599042486, -5.853638735),
                1: (-6.83482236, 4.563156318),
            }
        )
        remap = build_roi_remap(surface, valid_centers, tolerance=1.0)
        # valid[0] (VISp) should map to surface roi 1
        # valid[1] (VISl) should map to surface roi 0
        self.assertEqual(remap, {0: 1, 1: 0})


if __name__ == "__main__":
    unittest.main()
