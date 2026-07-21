"""Input-metadata reading + processing.json generation.

This module covers two concerns at the aind-data-schema boundary:

1. **Input metadata** is loaded as raw JSON dicts (no Pydantic) and mapped
   into a ``ConverterMetadata`` frozen dataclass via either ``_map_v1`` or
   ``_map_v2``, dispatched by ``load_converter_metadata`` based on **which
   core file is present** (``session.json`` -> v1, ``acquisition.json`` ->
   v2). Downstream code consumes only ``ConverterMetadata``; it never sees a
   v1 or v2 schema object.

2. **Processing output**: ``build_processing`` / ``write_processing_json``
   emit a v2 ``processing.json`` describing the conversion run.

All ``quality_control.json`` generation lives in ``qc.py``.
"""

import json
import logging
import os
import platform
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Optional, Tuple

from aind_data_schema.components.configs import (
    ImagingConfig,
    PlanarImage,
    PlanarImageStack,
)
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import (
    DataProcess,
    Processing,
    ProcessStage,
    ResourceUsage,
)
from aind_data_schema_models.process_names import ProcessName

# ---------------------------------------------------------------------------
# Internal metadata model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConverterMetadata:
    """Minimal, schema-agnostic metadata the converter actually uses.

    Holds only the fields the conversion consumes (the "fields as used by
    the code", not a faithful copy of the input file). Populated from either
    v1 (``session.json``) or v2 (``acquisition.json``) input by
    ``load_converter_metadata`` via a factory (``_map_v1``/``_map_v2``); the
    canonical FOV ids are resolved once, at construction. Downstream code
    reads only this object — never a v1/v2 aind-data-schema class, and never
    a raw targeted-structure acronym.

    Attributes
    ----------
    instrument_id : str
        Hardware identifier; drives Bergamo vs mesoscope routing.
    fov_ids : tuple of str
        Canonical FOV id per imaging plane, ordered by plane index (e.g.
        ``("plane_0",)`` single-plane, or ``("MOp_0", "VISp_1")``
        multi-plane). Precomputed via the shared naming rule (``_fov_id``)
        so the ids are identical across capsules. Paired with the TIFF's
        scan-order planes **by position** (``fov_ids[i]`` is the i-th
        scanned plane). Empty only for malformed input with no planes
        (consumers fail-fast on that).
    dataset_name : str or None
        Dataset name from ``data_description.json``; used as the logging
        label. ``None`` when no data_description.
    subject_id : str or None
        Subject identifier; used only as the logging-label fallback.
    """

    instrument_id: str
    fov_ids: Tuple[str, ...]
    dataset_name: Optional[str] = None
    subject_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_converter_metadata(input_dir: Path) -> ConverterMetadata:
    """Discover and parse v1 or v2 input metadata.

    Searches ``input_dir`` recursively for the core acquisition file and an
    optional data_description file. The schema version is dispatched by
    **which core file is present** — ``session.json`` (v1) vs
    ``acquisition.json`` (v2), the v1->v2 rename being the authoritative
    version signal — so no ``schema_version`` field is required.

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
        If neither ``acquisition.json`` nor ``session.json`` is found.
        (``data_description.json`` is optional.)
    ValueError
        If both ``acquisition.json`` and ``session.json`` are present
        (ambiguous input).
    KeyError
        If a required field is missing from the input metadata.
    """
    acq_path = _find_acquisition_file(input_dir)
    dd_path = _find_data_description_file(input_dir)
    acq_raw = json.loads(acq_path.read_text())
    dd_raw = json.loads(dd_path.read_text()) if dd_path else {}
    # Dispatch on which core file is present: the v1 -> v2 rename
    # (session.json -> acquisition.json) IS the authoritative version signal,
    # so no schema_version parsing is needed. Minimal inputs that omit
    # schema_version (and inputs with no data_description) still load.
    if acq_path.name == "session.json":
        return _map_v1(acq_raw, dd_raw, acq_path)
    return _map_v2(acq_raw, dd_raw, acq_path)


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


def _find_data_description_file(input_dir: Path) -> Optional[Path]:
    """Locate ``data_description.json`` under ``input_dir`` (recursive).

    Returns ``None`` when absent — a minimal input (e.g. a single-plane
    session) may ship only the session/acquisition file.
    """
    matches = list(input_dir.rglob("data_description.json"))
    return matches[0] if matches else None


def _require(blob: dict, key: str, file_path: Path) -> Any:
    """Return ``blob[key]`` or raise a descriptive KeyError.

    Parameters
    ----------
    blob : dict
        The raw JSON dict.
    key : str
        Required key.
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
            f"Required field '{key}' missing from {file_path.name}"
        )
    return blob[key]


def _acronym_from_targeted_structure(value: Any) -> Optional[str]:
    """Extract a structure acronym from a v1/v2 ``targeted_structure`` value.

    The field type changed across schema versions, so tolerate all shapes:
    a **bare acronym string** (``"DLS"`` — early v1, where
    ``FieldOfView.targeted_structure: str``), a **dict** carrying an
    ``"acronym"`` key (``{"acronym": "VISp", ...}`` — later v1's
    ``CCFStructure`` and v2's structure model), or **absent** (``None``).
    Returns ``None`` when no acronym is available (fine for single-plane,
    which is named ``plane_{index}`` regardless).
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        acronym = value.get("acronym")
        return str(acronym) if acronym is not None else None
    return None


def _map_v1(
    acq_raw: dict,
    dd_raw: dict,
    acq_path: Path,
) -> ConverterMetadata:
    """Map a v1 ``session.json`` (+ optional ``data_description.json``) to
    a ``ConverterMetadata`` instance.

    v1 layout: ``session["data_streams"][*]["ophys_fovs"][*]``. Each FOV
    *may* have an ``index`` (defaults to ``0`` for single-plane) and a
    ``targeted_structure`` that is either a bare acronym string
    (single-plane, e.g. ``"DLS"``) or a CCFStructure dict with an
    ``"acronym"`` key (multi-plane). ``subject_id`` and
    ``data_description.json``/``name`` are optional (logging labels only).
    See ``single_plane_support.md``.
    """
    instrument_id = _require(acq_raw, "rig_id", acq_path)
    subject_id = acq_raw.get("subject_id")
    dataset_name = dd_raw.get("name")

    pairs = []
    data_streams = _require(acq_raw, "data_streams", acq_path)
    for stream in data_streams:
        for fov in stream.get("ophys_fovs", []):
            plane_index = int(fov.get("index", 0))
            acronym = _acronym_from_targeted_structure(
                fov.get("targeted_structure")
            )
            pairs.append((plane_index, acronym))
    return ConverterMetadata(
        instrument_id=str(instrument_id),
        fov_ids=_build_fov_ids(pairs),
        dataset_name=dataset_name,
        subject_id=None if subject_id is None else str(subject_id),
    )


def _object_type_value(cls) -> str:
    """Return a schema class's auto-derived ``object_type`` discriminator.

    In aind-data-schema v2 the ``object_type`` field is a ``Literal``
    whose value is derived from the class name (e.g. ``ImagingConfig`` →
    ``"Imaging config"``) and used as the discriminator in serialized
    JSON. Reading it from ``model_fields`` instead of hardcoding the
    string keeps the v2 mapper correct if the schema ever re-words a
    discriminator.

    Parameters
    ----------
    cls : type
        An aind-data-schema v2 model class with an ``object_type`` field.

    Returns
    -------
    str
        The discriminator string the class serializes to.
    """
    return cls.model_fields["object_type"].default


# Discriminator values are read from the schema classes (not hardcoded) so
# they track the installed aind-data-schema version. See _object_type_value.
_V2_IMAGING_CONFIG_DISCRIMINATOR = _object_type_value(ImagingConfig)
_V2_PLANAR_IMAGE_DISCRIMINATORS = {
    _object_type_value(PlanarImage),
    _object_type_value(PlanarImageStack),
}


def _map_v2(
    acq_raw: dict,
    dd_raw: dict,
    acq_path: Path,
) -> ConverterMetadata:
    """Map a v2 ``acquisition.json`` (+ optional ``data_description.json``)
    to a ``ConverterMetadata`` instance.

    v2 layout:
    ``acquisition["data_streams"][*]["configurations"][*]`` (filter by
    ``object_type == "Imaging config"``) →
    ``images[*]`` (filter by ``object_type in {"Planar image",
    "Planar image stack"}``) →
    ``planes[*]`` — each plane has ``plane_index`` (optional; defaults
    to 0 for single-plane) and ``targeted_structure`` (a bare acronym
    string or a BrainStructureModel dict with an ``"acronym"`` key;
    optional). ``subject_id`` and ``data_description.json``/``name`` are
    optional (logging labels only).
    """
    instrument_id = _require(acq_raw, "instrument_id", acq_path)
    subject_id = acq_raw.get("subject_id")
    dataset_name = dd_raw.get("name")

    pairs = []
    data_streams = _require(acq_raw, "data_streams", acq_path)
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
                    acronym = _acronym_from_targeted_structure(
                        plane.get("targeted_structure")
                    )
                    pairs.append(
                        (int(plane.get("plane_index", 0)), acronym)
                    )
    return ConverterMetadata(
        instrument_id=str(instrument_id),
        fov_ids=_build_fov_ids(pairs),
        dataset_name=dataset_name,
        subject_id=None if subject_id is None else str(subject_id),
    )


# ---------------------------------------------------------------------------
# FOV / plane naming
# ---------------------------------------------------------------------------


def _build_fov_ids(pairs: list) -> Tuple[str, ...]:
    """Resolve canonical FOV ids for ``(plane_index, acronym)`` pairs.

    Sorts by plane index and applies the shared naming rule (``_fov_id``)
    once per plane, so the returned ids are stable and identical to what
    other capsules produce for the same input. Single- vs multi-plane is
    decided here (exactly one plane -> single); the acronym never escapes
    this module. The returned order matches the TIFF's scan order, which
    downstream splitting pairs with by position.

    Parameters
    ----------
    pairs : list of (int, str or None)
        Per-plane ``(plane_index, targeted_structure_acronym)`` pairs.

    Returns
    -------
    tuple of str
        Canonical FOV ids, ordered by plane index. Empty when ``pairs``
        is empty.
    """
    pairs = sorted(pairs, key=lambda p: p[0])
    single_plane = len(pairs) == 1
    return tuple(
        _fov_id(idx, acronym, single_plane=single_plane)
        for idx, acronym in pairs
    )


def _fov_id(
    plane_index: int, acronym: Optional[str], *, single_plane: bool
) -> str:
    """Return the FOV/plane id for a plane under the active naming scheme.

    Called once per plane by ``_build_fov_ids`` at construction time; the
    result is cached on ``ConverterMetadata.fov_ids``. The id flows
    downstream (capsules key their outputs/folders on it), so it must be
    identical across every capsule for the same input.

    Naming schemes (add new ones here as needed):

    - **single-plane** → ``"plane_{plane_index}"`` (e.g. ``"plane_0"``):
      stable and independent of targeted-structure metadata, which
      single-plane acquisitions may omit.
    - **multi-plane** → ``"{acronym}_{plane_index}"`` (e.g. ``"VISp_0"``).
      Falls back to ``"plane_{plane_index}"`` if the acronym is missing, so
      a malformed plane never yields ``"None_0"``.

    Parameters
    ----------
    plane_index : int
        Index of the plane within the imaging session.
    acronym : str or None
        CCF acronym of the plane's targeted structure, or ``None``.
    single_plane : bool
        Whether the acquisition has exactly one plane.

    Returns
    -------
    str
        FOV/plane identifier string.
    """
    if single_plane or not acronym:
        return f"plane_{plane_index}"
    return f"{acronym}_{plane_index}"


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
