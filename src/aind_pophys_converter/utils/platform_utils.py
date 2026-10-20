"""Platform JSON utilities (temporary — will be removed when platform.json
dependency is retired)."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple


def load_platform_plane_positions(
    platform_fp: Path,
) -> List[Tuple[int, float, Optional[float]]]:
    """Return the ScanImage position of every imaging plane in platform.json.

    Planes are returned in ``imaging_plane_groups`` order, group by group.

    Parameters
    ----------
    platform_fp : Path
        Path to the platform.json file.

    Returns
    -------
    list of (int, float, float or None)
        ``(scanimage_roi_index, scanimage_scanfield_z, targeted_depth)`` per
        plane. ``scanimage_scanfield_z`` is the z value ScanImage records for
        the plane in the TIFF.

    Raises
    ------
    ValueError
        If a plane has no ``scanimage_roi_index`` or
        ``scanimage_scanfield_z``.
    """
    with open(platform_fp) as f:
        platform_json = json.load(f)
    positions = []
    for group_index, group in enumerate(
        platform_json.get("imaging_plane_groups", [])
    ):
        for plane_index, plane in enumerate(group.get("imaging_planes", [])):
            roi_index = plane.get("scanimage_roi_index")
            z_value = plane.get("scanimage_scanfield_z")
            if roi_index is None or z_value is None:
                raise ValueError(
                    f"{platform_fp}: imaging_plane_groups[{group_index}]"
                    f".imaging_planes[{plane_index}] needs "
                    "scanimage_roi_index and scanimage_scanfield_z to "
                    "locate its TIFF pages."
                )
            depth = plane.get("targeted_depth")
            positions.append(
                (
                    int(roi_index),
                    float(z_value),
                    None if depth is None else float(depth),
                )
            )
    return positions


def is_child_session_via_platform_json(platform_fp: Path) -> bool:
    """Return True if platform.json contains a parent_session field.

    Parameters
    ----------
    platform_fp : Path
        Path to the platform.json file.

    Returns
    -------
    bool
        True if a parent_session key is present and non-null.
    """
    logging.debug(f"checking is_child_session_via_platform_json {platform_fp}")
    with open(platform_fp) as f:
        platform_json = json.load(f)
    parent_session = platform_json.get("parent_session", None)
    logging.debug(f"is_child_session returning {parent_session is not None}")
    return parent_session is not None
