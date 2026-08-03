"""Input-metadata reading + processing.json generation for the converter.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, discriminators, the
FOV-naming rule, and the ``Processing`` builders all live in the shared
library; this module only:

1. assembles the converter's schema-agnostic ``ConverterMetadata`` view from
   the shared field getters (``load_converter_metadata``), and
2. builds the conversion-specific ``Processing`` document
   (``build_processing`` / ``write_processing_json``).

All ``quality_control.json`` generation lives in ``qc.py``.
"""

from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple

from aind_data_schema.core.processing import Processing
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing

# ---------------------------------------------------------------------------
# Internal metadata model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConverterMetadata:
    """Minimal, schema-agnostic metadata the converter actually uses.

    Holds only the fields the conversion consumes (the "fields as used by
    the code", not a faithful copy of the input file). Populated from either
    v1 (``session.json``) or v2 (``acquisition.json``) input by
    ``load_converter_metadata`` via the shared field getters. Downstream code
    reads only this object — never a v1/v2 aind-data-schema class, and never
    a raw targeted-structure acronym.

    Attributes
    ----------
    instrument_id : str
        Hardware identifier; drives Bergamo vs mesoscope routing.
    fov_ids : tuple of str
        Canonical FOV id per imaging plane, ordered by plane index (e.g.
        ``("plane_0",)`` single-plane, or ``("MOp_0", "VISp_1")``
        multi-plane). Produced by the shared naming rule
        (``aind_pophys_metadata.fields.get_fov_ids``) so the ids are
        identical across capsules. Paired with the TIFF's scan-order planes
        by position (``fov_ids[i]`` is the i-th scanned plane). Empty only for
        malformed input with no planes (consumers fail-fast on that).
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
    """Discover and parse v1 or v2 input metadata into ``ConverterMetadata``.

    Searches ``input_dir`` recursively for the core acquisition file and an
    optional data_description file. Schema version is dispatched by which
    core file is present (``session.json`` -> v1, ``acquisition.json`` ->
    v2). ``instrument_id`` is required; the remaining fields are optional
    (logging labels / fail-fast-on-empty FOVs downstream).

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
    ValueError
        If both core files are present (ambiguous input).
    KeyError
        If the required ``instrument_id`` (v1 ``rig_id``) is missing.
    """
    core_path = io.find_acquisition_file(input_dir)
    version = io.detect_schema_version(core_path)
    core_raw = io.load_json(core_path)

    dd_path = io.find(input_dir, "data_description.json")
    dd_raw = io.load_json(dd_path) if dd_path else {}

    return ConverterMetadata(
        instrument_id=fields.get_instrument_id(
            version, core_raw, required=True, file_path=core_path
        ),
        fov_ids=fields.get_fov_ids(version, core_raw),
        dataset_name=fields.get_dataset_name(dd_raw),
        subject_id=fields.get_subject_id(core_raw),
    )


# ---------------------------------------------------------------------------
# Processing writer
# ---------------------------------------------------------------------------

_REPO_URL = "https://github.com/AllenNeuralDynamics/aind-pophys-converter"
_CODE_NAME = "AIND Pophys Converter"

#: ``DataProcess.name``. Set explicitly because ``ophys_nwb`` also uses
#: ``ProcessName.FILE_FORMAT_CONVERSION``, so both would otherwise derive the
#: same default name ("File format conversion") and the run-level aggregator
#: rejects a merged document with duplicate process names. Both capsules run
#: once per session, so a plane id cannot separate them.
_DATA_PROCESS_NAME = "Raw movie conversion to HDF5"


def build_processing(
    job_parameters: dict,
    library_version: str,
    start_time: dt,
    end_time: dt,
    error: Optional[BaseException] = None,
    input_data: Optional[List[str]] = None,
) -> Processing:
    """Build a v2 aind-data-schema Processing object for this run.

    Produces a single ``DataProcess`` describing the conversion, annotated
    with the CLI-driven ``JobSettings`` parameters on ``code.parameters``.
    Runtime status (success vs failure + the error message on failure) goes
    into ``notes``.

    Pipeline identity (``Processing.pipelines`` and
    ``DataProcess.pipeline_name``) is supplied by the shared builders from
    the ``PIPELINE_*`` environment variables the pipeline exports — it
    describes the pipeline, never this capsule, and is omitted entirely on
    a standalone run.

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
    input_data : list of str, optional
        Names of the input data assets consumed, for provenance.

    Returns
    -------
    Processing
        v2 Processing object ready for serialization.
    """
    notes = (
        None
        if error is None
        else f"Run failed with {type(error).__name__}: {error}"
    )
    code = shared_processing.build_code(
        url=_REPO_URL,
        name=_CODE_NAME,
        version=library_version,
        parameters=job_parameters,
        input_data=input_data,
    )
    data_process = shared_processing.build_data_process(
        process_type=ProcessName.FILE_FORMAT_CONVERSION,
        name=_DATA_PROCESS_NAME,
        code=code,
        start_time=start_time,
        end_time=end_time,
        notes=notes,
        resources=shared_processing.resource_usage(),
    )
    return shared_processing.build_processing(
        [data_process],
        notes=(
            None
            if error is None
            else "Conversion finished with status=failed."
        ),
    )


def write_processing_json(processing: Processing, output_dir: Path) -> Path:
    """Write ``processing.json`` to ``output_dir`` via the shared writer.

    Parameters
    ----------
    processing : Processing
        Object built by ``build_processing``.
    output_dir : Path
        Destination directory. Created if it does not exist.

    Returns
    -------
    Path
        Path to the written ``processing.json``.
    """
    return shared_processing.write_processing_json(processing, output_dir)
