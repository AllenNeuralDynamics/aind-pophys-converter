"""Centralised aind-data-schema interaction helpers.

All imports of ``aind-data-schema`` and ``aind-data-schema-models`` symbols
live in this module. Callers interact with schema objects only through the
functions here, so schema-version changes require edits in one place.

The module covers three concerns at the schema boundary:

1. **Input metadata** is loaded as raw JSON dicts (no Pydantic) and mapped
   into a ``ConverterMetadata`` frozen dataclass via either ``_map_v1`` or
   ``_map_v2``, dispatched by ``load_converter_metadata`` based on the
   ``schema_version`` field of the acquisition / session file. Downstream
   code consumes only ``ConverterMetadata``; it never sees a v1 or v2
   schema object.

2. **QC output** (unchanged): ``build_fov_qc_metric`` and
   ``build_vasculature_qc_metric`` construct v2 ``QCMetric`` objects.

3. **Processing output** (new): ``build_processing`` /
   ``write_processing_json`` emit a v2 ``processing.json`` describing
   the conversion run.
"""

import json
import logging
import os
import platform
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Optional, Tuple

import pytz
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import (
    DataProcess,
    Processing,
    ProcessStage,
    ResourceUsage,
)
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.process_names import ProcessName
from aind_qcportal_schema.metric_value import DropdownMetric

# ---------------------------------------------------------------------------
# Internal metadata model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaneInfo:
    """Per-plane info the converter actually consumes.

    Parameters
    ----------
    plane_index : int
        Index of the plane within the imaging session. ``0`` for
        single-plane instruments (e.g. Bergamo) that have no explicit index.
    targeted_structure_acronym : str
        CCF acronym of the targeted brain structure, e.g. ``"VISp"``.
    """

    plane_index: int
    targeted_structure_acronym: str


@dataclass(frozen=True)
class ConverterMetadata:
    """Schema-agnostic metadata object consumed by the converter.

    Populated from either v1 (``session.json``) or v2
    (``acquisition.json``) input by ``load_converter_metadata``. The
    converter pipeline reads only this object; it never sees a v1 or v2
    aind-data-schema Pydantic class.

    Attributes
    ----------
    source_schema_version : str
        Full schema_version string from the input acquisition / session
        file, e.g. ``"1.1.2"`` or ``"2.4.0"``. Used for provenance and
        error messages.
    source_acquisition_file : Path
        Path to the input ``acquisition.json`` (v2) or ``session.json``
        (v1) that this metadata was derived from.
    source_data_description_file : Path
        Path to the input ``data_description.json``.
    instrument_id : str
        Hardware identifier; drives Bergamo vs mesoscope routing.
    dataset_name : str
        Dataset name from ``data_description.json``; drives multiplane
        detection.
    subject_id : str
        Subject identifier.
    planes : tuple of PlaneInfo
        Imaging planes sorted by ``plane_index``. Tuple for immutability.
    """

    source_schema_version: str
    source_acquisition_file: Path
    source_data_description_file: Path
    instrument_id: str
    dataset_name: str
    subject_id: str
    planes: Tuple[PlaneInfo, ...]


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_converter_metadata(input_dir: Path) -> ConverterMetadata:
    """Discover, detect, and parse v1 or v2 input metadata.

    Searches ``input_dir`` recursively for an acquisition / session JSON
    file and a data_description JSON file, detects the schema major
    version from the acquisition / session ``schema_version`` field, and
    dispatches to the appropriate mapper.

    Parameters
    ----------
    input_dir : Path
        Directory containing input metadata files (searched recursively).

    Returns
    -------
    ConverterMetadata
        Schema-agnostic metadata object populated from input JSON.

    Raises
    ------
    FileNotFoundError
        If neither ``acquisition.json`` nor ``session.json`` is found,
        or if ``data_description.json`` is not found.
    ValueError
        If both ``acquisition.json`` and ``session.json`` are present
        (ambiguous input), or if the detected schema major version is
        not supported.
    KeyError
        If a required field is missing from the input metadata.
    """
    acq_path = _find_acquisition_file(input_dir)
    dd_path = _find_data_description_file(input_dir)
    acq_raw = json.loads(acq_path.read_text())
    dd_raw = json.loads(dd_path.read_text())
    schema_version = str(acq_raw.get("schema_version", ""))
    major = _detect_major_version(acq_raw, acq_path)
    if major == 1:
        return _map_v1(acq_raw, dd_raw, acq_path, dd_path, schema_version)
    if major == 2:
        return _map_v2(acq_raw, dd_raw, acq_path, dd_path, schema_version)
    raise ValueError(
        f"Unsupported schema_version major {major} in {acq_path} "
        f"(schema_version={schema_version!r})"
    )


def _find_acquisition_file(input_dir: Path) -> Path:
    """Locate the input acquisition or session JSON file.

    Prefers v2 ``acquisition.json``. Falls back to v1 ``session.json``.
    Raises if both are present (ambiguous) or neither is present.

    Parameters
    ----------
    input_dir : Path
        Directory to search recursively.

    Returns
    -------
    Path
        Path to the located file.

    Raises
    ------
    ValueError
        If both ``acquisition.json`` and ``session.json`` are present.
    FileNotFoundError
        If neither file is present.
    """
    acq_matches = list(input_dir.rglob("acquisition.json"))
    session_matches = list(input_dir.rglob("session.json"))
    if acq_matches and session_matches:
        raise ValueError(
            f"Ambiguous input: found both v2 acquisition.json "
            f"({acq_matches[0]}) and v1 session.json "
            f"({session_matches[0]}) under {input_dir}. "
            f"Remove one to disambiguate."
        )
    if acq_matches:
        return acq_matches[0]
    if session_matches:
        return session_matches[0]
    raise FileNotFoundError(
        f"No acquisition.json (v2) or session.json (v1) found under "
        f"{input_dir}"
    )


def _find_data_description_file(input_dir: Path) -> Path:
    """Locate ``data_description.json`` under ``input_dir`` (recursive)."""
    matches = list(input_dir.rglob("data_description.json"))
    if not matches:
        raise FileNotFoundError(
            f"No data_description.json found under {input_dir}"
        )
    return matches[0]


def _detect_major_version(blob: dict, file_path: Path) -> int:
    """Parse ``blob['schema_version']`` and return the integer major.

    Parameters
    ----------
    blob : dict
        Raw JSON dict from the acquisition / session file.
    file_path : Path
        Source file path, included in error messages.

    Returns
    -------
    int
        Major version (1 or 2).

    Raises
    ------
    ValueError
        If ``schema_version`` is missing or malformed.
    """
    version_str = blob.get("schema_version")
    if not version_str:
        raise ValueError(
            f"Missing 'schema_version' field in {file_path}; "
            f"cannot detect v1 vs v2 schema."
        )
    try:
        major_str = str(version_str).split(".", 1)[0]
        return int(major_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Malformed schema_version {version_str!r} in {file_path}: "
            f"{exc}"
        ) from exc


def _require(
    blob: dict, key: str, source_version: str, file_path: Path
) -> Any:
    """Return ``blob[key]`` or raise a descriptive KeyError.

    Parameters
    ----------
    blob : dict
        The raw JSON dict.
    key : str
        Required key.
    source_version : str
        Schema version string, for the error message.
    file_path : Path
        Source file path, for the error message.

    Returns
    -------
    Any
        The value at ``blob[key]``.

    Raises
    ------
    KeyError
        If ``key`` is not in ``blob``.
    """
    if key not in blob:
        raise KeyError(
            f"Required field '{key}' missing from {file_path.name} "
            f"(schema v{source_version})"
        )
    return blob[key]


def _map_v1(
    acq_raw: dict,
    dd_raw: dict,
    acq_path: Path,
    dd_path: Path,
    schema_version: str,
) -> ConverterMetadata:
    """Map a v1 ``session.json`` + ``data_description.json`` pair to
    a ``ConverterMetadata`` instance.

    v1 layout:
    ``session["data_streams"][*]["ophys_fovs"][*]`` →
    each FOV has ``index`` and ``targeted_structure`` (a CCFStructure
    dict with an ``acronym`` key).
    """
    instrument_id = _require(acq_raw, "rig_id", schema_version, acq_path)
    subject_id = _require(acq_raw, "subject_id", schema_version, acq_path)
    dataset_name = _require(dd_raw, "name", schema_version, dd_path)

    planes = []
    data_streams = _require(
        acq_raw, "data_streams", schema_version, acq_path
    )
    for stream in data_streams:
        for fov in stream.get("ophys_fovs", []):
            plane_index = _require(fov, "index", schema_version, acq_path)
            targeted_structure = _require(
                fov, "targeted_structure", schema_version, acq_path
            )
            acronym = _require(
                targeted_structure, "acronym", schema_version, acq_path
            )
            planes.append(
                PlaneInfo(
                    plane_index=int(plane_index),
                    targeted_structure_acronym=str(acronym),
                )
            )
    planes.sort(key=lambda p: p.plane_index)
    return ConverterMetadata(
        source_schema_version=schema_version,
        source_acquisition_file=acq_path,
        source_data_description_file=dd_path,
        instrument_id=str(instrument_id),
        dataset_name=str(dataset_name),
        subject_id=str(subject_id),
        planes=tuple(planes),
    )


_V2_IMAGING_CONFIG_DISCRIMINATOR = "Imaging config"
_V2_PLANAR_IMAGE_DISCRIMINATORS = {"Planar image", "Planar image stack"}


def _map_v2(
    acq_raw: dict,
    dd_raw: dict,
    acq_path: Path,
    dd_path: Path,
    schema_version: str,
) -> ConverterMetadata:
    """Map a v2 ``acquisition.json`` + ``data_description.json`` pair
    to a ``ConverterMetadata`` instance.

    v2 layout:
    ``acquisition["data_streams"][*]["configurations"][*]`` (filter by
    ``object_type == "Imaging config"``) →
    ``images[*]`` (filter by ``object_type in {"Planar image",
    "Planar image stack"}``) →
    ``planes[*]`` — each plane has ``plane_index`` (optional; defaults
    to 0 for single-plane instruments) and ``targeted_structure``
    (a BrainStructureModel dict with an ``acronym`` key).
    """
    instrument_id = _require(
        acq_raw, "instrument_id", schema_version, acq_path
    )
    subject_id = _require(acq_raw, "subject_id", schema_version, acq_path)
    dataset_name = _require(dd_raw, "name", schema_version, dd_path)

    planes = []
    data_streams = _require(
        acq_raw, "data_streams", schema_version, acq_path
    )
    for stream in data_streams:
        configurations = stream.get("configurations", [])
        for config in configurations:
            if (
                config.get("object_type")
                != _V2_IMAGING_CONFIG_DISCRIMINATOR
            ):
                continue
            for image in config.get("images", []):
                if (
                    image.get("object_type")
                    not in _V2_PLANAR_IMAGE_DISCRIMINATORS
                ):
                    continue
                for plane in image.get("planes", []):
                    targeted_structure = _require(
                        plane,
                        "targeted_structure",
                        schema_version,
                        acq_path,
                    )
                    acronym = _require(
                        targeted_structure,
                        "acronym",
                        schema_version,
                        acq_path,
                    )
                    planes.append(
                        PlaneInfo(
                            plane_index=int(plane.get("plane_index", 0)),
                            targeted_structure_acronym=str(acronym),
                        )
                    )
    planes.sort(key=lambda p: p.plane_index)
    return ConverterMetadata(
        source_schema_version=schema_version,
        source_acquisition_file=acq_path,
        source_data_description_file=dd_path,
        instrument_id=str(instrument_id),
        dataset_name=str(dataset_name),
        subject_id=str(subject_id),
        planes=tuple(planes),
    )


# ---------------------------------------------------------------------------
# FOV helper
# ---------------------------------------------------------------------------


def fov_id_from_plane(plane: PlaneInfo) -> str:
    """Return the FOV identifier ``"{acronym}_{plane_index}"`` for a plane.

    Parameters
    ----------
    plane : PlaneInfo
        Imaging plane info.

    Returns
    -------
    str
        FOV identifier string, e.g. ``"VISp_3"`` or ``"MO_0"``.
    """
    return f"{plane.targeted_structure_acronym}_{plane.plane_index}"


# ---------------------------------------------------------------------------
# QC schema helpers
# ---------------------------------------------------------------------------


def pending_qc_status() -> QCStatus:
    """Return a PENDING QCStatus timestamped to US/Pacific.

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
    return QCMetric(
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
        status_history=[pending_qc_status()],
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
    return QCMetric(
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
        status_history=[pending_qc_status()],
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


# ---------------------------------------------------------------------------
# Processing writer
# ---------------------------------------------------------------------------

_REPO_URL = (
    "https://github.com/AllenNeuralDynamics/aind-pophys-converter"
)
_CODE_NAME = "AIND Pophys Converter"
_PIPELINE_NAME = "aind-pophys-converter"


def build_processing(
    job_parameters: dict,
    library_version: str,
    start_time: dt,
    end_time: dt,
    error: Optional[BaseException] = None,
) -> Processing:
    """Build a v2 aind-data-schema Processing object for this run.

    Produces a single ``DataProcess`` describing the conversion. The
    process is annotated with the CLI-driven ``JobSettings`` parameters
    on ``code.parameters`` — nothing else. Runtime status (success vs
    failure + the error message on failure) goes into ``notes``.

    Parameters
    ----------
    job_parameters : dict
        Dict of CLI-driven JobSettings fields (input_dir, output_dir,
        temp_dir, debug). Caller produces this via
        ``job_settings.model_dump(mode="json")``.
    library_version : str
        Library version string (e.g. ``aind_pophys_converter.__version__``).
    start_time : datetime.datetime
        Timezone-aware start time of the run.
    end_time : datetime.datetime
        Timezone-aware end time of the run.
    error : Exception, optional
        The exception that terminated the run, or ``None`` on success.

    Returns
    -------
    Processing
        v2 Processing object ready for serialization.
    """
    status_text = "success" if error is None else "failed"
    notes = (
        None
        if error is None
        else f"Run failed with {type(error).__name__}: {error}"
    )
    code = Code(
        url=_REPO_URL,
        name=_CODE_NAME,
        version=library_version,
        language="Python",
        language_version=platform.python_version(),
        parameters=job_parameters,
    )
    resources = ResourceUsage(
        os=platform.system() or "unknown",
        architecture=platform.machine() or "unknown",
        cpu_cores=os.cpu_count(),
    )
    data_process = DataProcess(
        process_type=ProcessName.FILE_FORMAT_CONVERSION,
        stage=ProcessStage.PROCESSING,
        code=code,
        experimenters=["Allen Institute for Neural Dynamics"],
        pipeline_name=_PIPELINE_NAME,
        start_date_time=start_time,
        end_date_time=end_time,
        notes=notes,
        resources=resources,
    )
    return Processing(
        data_processes=[data_process],
        pipelines=[
            Code(
                url=_REPO_URL,
                name=_PIPELINE_NAME,
                version=library_version,
            )
        ],
        notes=(
            None
            if status_text == "success"
            else f"Conversion finished with status={status_text}."
        ),
    )


def write_processing_json(
    processing: Processing, output_dir: Path
) -> Path:
    """Write ``processing.json`` to ``output_dir`` via the schema helper.

    Uses ``Processing.write_standard_file`` so the file format matches
    other v2 metadata artifacts (3-space indent, ``describedBy`` /
    ``schema_version`` populated).

    Parameters
    ----------
    processing : Processing
        Object built by ``build_processing``.
    output_dir : Path
        Destination directory. Created if it does not exist.

    Returns
    -------
    Path
        Absolute path to the written ``processing.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processing.write_standard_file(output_directory=str(output_dir))
    written = output_dir / "processing.json"
    logging.info(f"processing.json written to {written}")
    return written
