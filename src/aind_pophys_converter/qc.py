"""All ``quality_control.json`` generation for aind-pophys-converter.

This module owns the full QC path: constructing individual ``QCMetric``
objects, the QC visual generation (depth pairing, vasculature PNG) which
returns the metrics it produces, and assembling + writing a single v2
``QualityControl`` object. ``job.run`` accumulates the returned metrics and
calls ``finalize_quality_control`` to emit one ``quality_control.json`` —
never per-metric fragments.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import tifffile
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_pophys_metadata import find_one
from aind_pophys_metadata import quality_control as shared_qc
from PIL import Image

from aind_pophys_converter.utils.image_utils import add_border_and_label

if TYPE_CHECKING:
    from aind_pophys_converter.tiff_splitters import AvgImageTiffSplitter

# Evaluator label recorded on every pending status this capsule emits.
_EVALUATOR = "Automated"


def pending_qc_status() -> QCStatus:
    """Return a PENDING QCStatus timestamped to US/Pacific.

    Returns
    -------
    QCStatus
        A QCStatus with status=PENDING and evaluator='Automated'.
    """
    return shared_qc.pending_qc_status(_EVALUATOR)


def build_fov_qc_metric(
    unique_id: str,
    raw_tif_path: Path,
    intended_depth: int,
    targeted_structure_id: str,
    closest_z: float,
    merged_path: Path,
) -> QCMetric:
    """Construct the FOV parent-child matching QCMetric object.

    Parameters
    ----------
    unique_id : str
        Identifier string like 'VISp_200'.
    raw_tif_path : Path
        Path to the parent depth snapshot TIFF.
    intended_depth : int
        Intended imaging depth in microns.
    targeted_structure_id : str
        CCF structure acronym.
    closest_z : float
        Closest child z-plane in microns.
    merged_path : Path
        Path to the saved merged PNG.

    Returns
    -------
    QCMetric
        Constructed QCMetric object, not yet serialized.
    """
    return shared_qc.dropdown_metric(
        name=f"{unique_id} Parent-Child FOV",
        evaluation="Op. QC: Field-of-view Matching",
        options=[
            "FOV Matches parent FOV",
            "FOV does not match parent FOV.",
        ],
        status=[Status.PASS, Status.FAIL],
        reference=str(merged_path),
        description=(
            f"Side-by-side comparison of the parent session depth snapshot "
            f"({raw_tif_path.stem}, intended_depth={intended_depth}µm, "
            f"structure={targeted_structure_id}) and the current session's "
            f"averaged-depth image at the closest matching z-plane "
            f"(child z: {closest_z}µm). "
            f"Pass if the fields of view align spatially across sessions."
        ),
        stage=Stage.RAW,
        evaluator=_EVALUATOR,
    )


def build_vasculature_qc_metric(vasculature_output_fp: Path) -> QCMetric:
    """Construct the vasculature window-clarity QCMetric object.

    Parameters
    ----------
    vasculature_output_fp : Path
        Path to the saved vasculature PNG.

    Returns
    -------
    QCMetric
        Constructed QCMetric object, not yet serialized.
    """
    return shared_qc.dropdown_metric(
        name="Vasculature Image",
        evaluation="Op. QC: Window Clarity",
        options=[
            "Quality is sufficient",
            "Poor vasculature image quality",
            "Light bruising on surface of brain",
            "Severe bruising on surface of brain",
            "Vascularization of brain surface",
            "Discoloration of brain surface (white)",
            (
                "Bubbles in objective immersion present, but do NOT"
                " impact imaging quality"
            ),
            "Bubbles in objective immersion impact imaging quality",
        ],
        status=[
            Status.PASS,
            Status.PASS,
            Status.PASS,
            Status.FAIL,
            Status.PASS,
            Status.PASS,
            Status.PASS,
            Status.FAIL,
        ],
        reference=str(vasculature_output_fp),
        description=(
            "Vasculature image captured at session start to assess brain "
            "surface health and cranial window clarity. Select the option "
            "that best describes the observed surface condition. Bruising, "
            "bubbles, or discoloration may indicate compromised tissue or "
            "optical path quality."
        ),
        stage=Stage.RAW,
        evaluator=_EVALUATOR,
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap a list of QCMetrics into a single v2 ``QualityControl``.

    The project's output contract is that each capsule emits one complete
    ``quality_control.json`` (a ``QualityControl`` object), not loose
    per-metric fragments; the QC aggregator later merges these via
    ``QualityControl.__add__``. The shared builder sets the required
    ``default_grouping`` so the aggregator and every capsule agree on the
    grouping key.

    Parameters
    ----------
    metrics : list of QCMetric
        Metrics produced during the conversion run. Must be non-empty
        (callers should skip writing QC when no metrics were produced).

    Returns
    -------
    QualityControl
        v2 QualityControl object ready for serialization.
    """
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl, output_dir: Path
) -> Path:
    """Write ``quality_control.json`` to ``output_dir`` via the schema helper.

    Uses the shared writer (``QualityControl.write_standard_file``) so the
    file format matches other v2 metadata artifacts (``describedBy`` /
    ``schema_version`` populated).

    Parameters
    ----------
    quality_control : QualityControl
        Object built by ``build_quality_control``.
    output_dir : Path
        Destination directory. Created if it does not exist.

    Returns
    -------
    Path
        Absolute path to the written ``quality_control.json``.
    """
    return shared_qc.write_quality_control_json(quality_control, output_dir)


def write_avg_depth_slices(
    splitter: "AvgImageTiffSplitter", output_dir: Path
) -> None:
    """Write per-ROI averaged-depth slices as float32 TIF files.

    Parameters
    ----------
    splitter : AvgImageTiffSplitter
        Initialized splitter with a loaded averaged-depth TIFF.
    output_dir : Path
        Directory to write the per-z-value TIF files.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for roi_idx, z_int in splitter.roi_z_int_manifest:
        z_value = splitter._z_from_int(z_int)
        tif_path = output_dir / f"{z_value:.1f}.tif"
        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp_tif:
            tmp_path = Path(tmp_tif.name)
            splitter.write_output_file(
                i_roi=roi_idx, z_value=z_value, output_path=tmp_path
            )
            img_array = tifffile.imread(tmp_path)
        tifffile.imwrite(tif_path, img_array.astype(np.float32))
        logging.info(f"Saved TIF: {tif_path}")


def pair_depth_tifs_with_avg_depth_pngs(
    pophys_dir: Path,
    avg_slice_dir: Path,
    platform_fp: Path,
    output_dir: Path,
) -> List[QCMetric]:
    """Pair parent depth TIFs with child averaged-depth PNGs; return QC.

    For each imaging plane defined in platform.json, locate the parent depth
    TIFF by intended_depth and targeted_structure_id, find the closest child
    averaged-depth TIF by abs(scanimage_scanfield_z), merge them side-by-side
    with borders and labels, and save the result. Builds a per-plane
    ``QCMetric`` and returns the collected metrics (does not write per-metric
    JSON files — the caller assembles a single ``quality_control.json``).

    Parameters
    ----------
    pophys_dir : Path
        Directory containing parent depth TIFFs named
        <timestamp>_<intended_depth>_<targeted_structure_id>_depth.tif.
    avg_slice_dir : Path
        Directory containing child averaged-depth float32 TIF files named by
        z-value (e.g. -276.0.tif), written by write_avg_depth_slices.
    platform_fp : Path
        Path to the session platform.json.
    output_dir : Path
        Directory to save merged PNGs.

    Returns
    -------
    list of QCMetric
        One metric per successfully paired plane. Empty if no planes could
        be paired.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics: List[QCMetric] = []

    with open(platform_fp) as f:
        platform_json = json.load(f)

    imaging_planes = [
        plane
        for group in platform_json.get("imaging_plane_groups", [])
        for plane in group.get("imaging_planes", [])
    ]

    if not imaging_planes:
        logging.info(
            "No imaging planes found in platform.json;"
            " skipping depth pairing."
        )
        return metrics

    child_tifs = list(avg_slice_dir.glob("*.tif"))
    z_to_tif = {abs(float(p.stem)): p for p in child_tifs}

    if not z_to_tif:
        logging.info("No averaged-depth TIFs found; skipping depth pairing.")
        return metrics

    for plane in imaging_planes:
        intended_depth = plane["intended_depth"]
        targeted_structure_id = plane["targeted_structure_id"]
        scanfield_z = plane["scanimage_scanfield_z"]

        parent_tifs = list(
            pophys_dir.glob(
                f"*_{intended_depth}_{targeted_structure_id}_depth.tif"
            )
        )
        if len(parent_tifs) == 0:
            logging.warning(
                f"No parent TIF found for "
                f"intended_depth={intended_depth}, "
                f"targeted_structure_id={targeted_structure_id}; skipping."
            )
            continue
        if len(parent_tifs) > 1:
            logging.warning(
                f"Multiple parent TIFs found for "
                f"intended_depth={intended_depth}, "
                f"targeted_structure_id={targeted_structure_id}; "
                f"using first: {parent_tifs[0].name}"
            )
        raw_tif_path = parent_tifs[0]

        closest_z = min(
            z_to_tif.keys(), key=lambda z: abs(z - abs(scanfield_z))
        )
        child_tif_path = z_to_tif[closest_z]

        with tifffile.TiffFile(raw_tif_path) as tif:
            parent_array = tif.asarray().astype(np.float64)
        with tifffile.TiffFile(child_tif_path) as tif:
            child_array = tif.asarray().astype(np.float64)

        def _to_uint8(arr: np.ndarray) -> np.ndarray:
            """Normalize array to uint8 using 5th/95th percentiles."""
            lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
            if hi > lo:
                return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(
                    np.uint8
                )
            return np.zeros_like(arr, dtype=np.uint8)

        p_lo = np.percentile(parent_array, 5)
        p_hi = np.percentile(parent_array, 95)
        c_lo = np.percentile(child_array, 5)
        c_hi = np.percentile(child_array, 95)
        logging.info(
            f"Parent normalization: p5={p_lo:.1f}, p95={p_hi:.1f}"
            f" (dtype={parent_array.dtype})"
        )
        logging.info(
            f"Child  normalization: p5={c_lo:.1f}, p95={c_hi:.1f}"
            f" (dtype={child_array.dtype})"
        )

        raw_img = Image.fromarray(_to_uint8(parent_array))
        avg_img = Image.fromarray(_to_uint8(child_array))
        raw_img = add_border_and_label(raw_img, "Parent")
        avg_img = add_border_and_label(avg_img, "Child")

        total_width = raw_img.width + avg_img.width
        max_height = max(raw_img.height, avg_img.height)
        merged = Image.new("RGB", (total_width, max_height), color="white")
        merged.paste(raw_img, (0, 0))
        merged.paste(avg_img, (raw_img.width, 0))

        merged_path = output_dir / f"{raw_tif_path.stem}_merged.png"
        merged.save(merged_path)
        logging.info(
            f"Saved merged image for {raw_tif_path.name} -> {merged_path}"
        )

        try:
            child_tif_path.unlink()
            logging.info(f"Deleted original averaged TIF -> {child_tif_path}")
        except Exception as e:
            logging.warning(f"Failed to delete {child_tif_path}: {e}")

        unique_id = f"{targeted_structure_id}_{intended_depth}"
        metrics.append(
            build_fov_qc_metric(
                unique_id=unique_id,
                raw_tif_path=raw_tif_path,
                intended_depth=intended_depth,
                targeted_structure_id=targeted_structure_id,
                closest_z=closest_z,
                merged_path=merged_path,
            )
        )

    return metrics


def create_vasculature(
    pophys_dir: Path, output_dir: Path
) -> Optional[QCMetric]:
    """Create a vasculature PNG and return a window-clarity QC metric.

    Parameters
    ----------
    pophys_dir : Path
        Directory containing the vasculature TIFF file.
    output_dir : Path
        Parent output directory; vasculature/ subdirectory is created here.

    Returns
    -------
    QCMetric or None
        The window-clarity metric, or ``None`` if no vasculature TIFF was
        found (nothing to assess).
    """
    vasculature_fp = find_one(
        pophys_dir, "*_vasculature.tif", recursive=False, required=False
    )
    if not vasculature_fp:
        logging.info("No vasculature TIFF found; skipping.")
        return None
    vasculature_output_dir = output_dir / "vasculature"
    vasculature_output_dir.mkdir(exist_ok=True, parents=True)
    vasculature_output_fp = vasculature_output_dir / "vasculature.png"
    with Image.open(vasculature_fp) as im:
        im.save(vasculature_output_fp)
    logging.info(f"Saved vasculature image -> {vasculature_output_fp}")
    return build_vasculature_qc_metric(vasculature_output_fp)


def finalize_quality_control(
    qc_metrics: List[QCMetric],
    pophys_dir: Path,
    output_dir: Path,
) -> None:
    """Add the vasculature metric and emit a single quality_control.json.

    Appends the window-clarity metric (if a vasculature TIFF exists) to the
    FOV metrics collected upstream, then writes one ``QualityControl``
    object. Writes nothing when no metrics were produced.

    Parameters
    ----------
    qc_metrics : list of QCMetric
        Metrics collected so far (e.g. from FOV depth pairing). Mutated
        in place to append the vasculature metric.
    pophys_dir : Path
        Directory searched for the vasculature TIFF.
    output_dir : Path
        Destination directory for ``quality_control.json``.
    """
    vasculature_metric = create_vasculature(pophys_dir, output_dir)
    if vasculature_metric is not None:
        qc_metrics.append(vasculature_metric)
    if qc_metrics:
        write_quality_control_json(
            build_quality_control(qc_metrics), output_dir
        )
