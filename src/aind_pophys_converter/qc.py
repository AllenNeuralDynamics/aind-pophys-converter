"""Schema-v2 quality-control construction for the converter."""

import logging
from pathlib import Path
from typing import List, Optional

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

_EVALUATOR = "Automated"


def pending_qc_status() -> QCStatus:
    """Return an automated pending QC status."""
    return shared_qc.pending_qc_status(_EVALUATOR)


def build_fov_qc_metric(
    unique_id: str,
    raw_tif_path: Path,
    intended_depth: int,
    targeted_structure_id: str,
    closest_z: float,
    merged_path: Path,
) -> QCMetric:
    """Build a pending parent-child FOV metric."""
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
            f"(child z: {closest_z}µm)."
        ),
        stage=Stage.RAW,
        evaluator=_EVALUATOR,
    )


def build_vasculature_qc_metric(vasculature_output_fp: Path) -> QCMetric:
    """Build a pending window-clarity metric."""
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
            "Bubbles in objective immersion present, but do NOT impact "
            "imaging quality",
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
            "surface health and cranial window clarity."
        ),
        stage=Stage.RAW,
        evaluator=_EVALUATOR,
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap collected metrics in one complete v2 QC document."""
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl, output_dir: Path
) -> Path:
    """Write one standard ``quality_control.json`` file."""
    return shared_qc.write_quality_control_json(quality_control, output_dir)


def create_vasculature(
    pophys_dir: Path, output_dir: Path
) -> Optional[QCMetric]:
    """Save the vasculature image and return its QC metric, if present."""
    vasculature_fp = find_one(
        pophys_dir, "*_vasculature.tif", recursive=False, required=False
    )
    if vasculature_fp is None:
        logging.info("No vasculature TIFF found; skipping.")
        return None
    output_dir = output_dir / "vasculature"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_fp = output_dir / "vasculature.png"
    with Image.open(vasculature_fp) as image:
        image.save(output_fp)
    logging.info(f"Saved vasculature image -> {output_fp}")
    return build_vasculature_qc_metric(output_fp)


def finalize_quality_control(
    qc_metrics: List[QCMetric], pophys_dir: Path, output_dir: Path
) -> None:
    """Add available metrics and emit one complete QC document."""
    vasculature_metric = create_vasculature(pophys_dir, output_dir)
    if vasculature_metric is not None:
        qc_metrics.append(vasculature_metric)
    if qc_metrics:
        write_quality_control_json(
            build_quality_control(qc_metrics), output_dir
        )
