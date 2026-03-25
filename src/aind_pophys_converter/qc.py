"""QC metric construction and serialization for pophys converter."""
import json
from datetime import datetime as dt
from pathlib import Path

import pytz
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import DropdownMetric


def PendingStatus() -> QCStatus:
    """Create a pending QCStatus timestamped to US/Pacific.

    Returns
    -------
    QCStatus
        A QCStatus with status=PENDING and evaluator='Automated'.
    """
    seattle_tz = pytz.timezone("America/Los_Angeles")
    return QCStatus(
        evaluator="Automated",
        status=Status.PENDING,
        timestamp=dt.now(seattle_tz).isoformat(),
    )


def write_fov_metric(
    unique_id: str,
    raw_tif_path: Path,
    intended_depth: int,
    targeted_structure_id: str,
    closest_z: float,
    merged_path: Path,
    output_dir: Path,
) -> None:
    """Write a single FOV parent-child matching QC metric JSON.

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
    output_dir : Path
        Directory to write the metric JSON.
    """
    metric = QCMetric(
        name=f"{unique_id} Parent-Child FOV",
        modality=Modality.POPHYS,
        stage=Stage.RAW,
        tags={
            "evaluation": "Op. QC: Field-of-view Matching",
            "type": "Operational QC",
        },
        description=(
            f"Side-by-side comparison of the parent session depth snapshot "
            f"({raw_tif_path.stem}, intended_depth={intended_depth}µm, "
            f"structure={targeted_structure_id}) and the current session's "
            f"averaged-depth image at the closest matching z-plane "
            f"(child z: {closest_z}µm). "
            f"Pass if the fields of view align spatially across sessions."
        ),
        status_history=[PendingStatus()],
        reference=str(merged_path),
        value=DropdownMetric(
            value="",
            options=[
                "FOV Matches parent FOV",
                "FOV does not match parent FOV.",
            ],
            status=[Status.PASS, Status.FAIL],
        ),
    )
    metric_out_path = output_dir / f"{unique_id}_fov_metric.json"
    with open(metric_out_path, "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)
    print(f"Saved QC metric -> {metric_out_path}")


def write_vasculature_metric(
    vasculature_output_fp: Path,
    output_dir: Path,
) -> None:
    """Write the vasculature window clarity QC metric JSON.

    Parameters
    ----------
    vasculature_output_fp : Path
        Path to the saved vasculature PNG.
    output_dir : Path
        Directory to write the metric JSON.
    """
    metric = QCMetric(
        name="Vasculature Image",
        modality=Modality.POPHYS,
        stage=Stage.RAW,
        tags={
            "evaluation": "Op. QC: Window Clarity",
            "type": "Operational QC",
        },
        description=(
            "Vasculature image captured at session start to assess brain "
            "surface health and cranial window clarity. Select the option "
            "that best describes the observed surface condition. Bruising, "
            "bubbles, or discoloration may indicate compromised tissue or "
            "optical path quality."
        ),
        status_history=[PendingStatus()],
        reference=str(vasculature_output_fp),
        value=DropdownMetric(
            value="",
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
        ),
    )
    metric_out_path = output_dir / "vasculature_metric.json"
    with open(metric_out_path, "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)
    print(f"Saved QC metric -> {metric_out_path}")
