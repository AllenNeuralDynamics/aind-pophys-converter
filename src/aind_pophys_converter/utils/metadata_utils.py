"""Input-metadata reading + processing.json generation for the converter.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, discriminators, the
FOV-naming rule, and the ``Processing`` builders all live in the shared
library; this module only:

1. assembles the converter's schema-agnostic ``ConverterMetadata`` view from
   the shared field getters (``load_converter_metadata``), and
2. builds the conversion-specific ``Code`` block (``build_converter_code``)
   that ``aind_pophys_metadata.runner.stage_guard`` writes out.

All ``quality_control.json`` generation lives in ``qc.py``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from aind_data_schema.components.identifiers import Code
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
    common = io.load_common(input_dir)
    return ConverterMetadata(
        instrument_id=fields.get_instrument_id(
            common.version,
            common.core_raw,
            required=True,
            file_path=common.core_path,
        ),
        fov_ids=fields.get_fov_ids(common.version, common.core_raw),
        dataset_name=fields.get_dataset_name(
            common.data_description_raw or {}
        ),
        subject_id=fields.get_subject_id(common.core_raw),
    )


# ---------------------------------------------------------------------------
# Code block
# ---------------------------------------------------------------------------

#: Distribution name. The shared builder derives both ``code.url`` and
#: ``code.version`` from it; this repo is its own backing library.
LIBRARY_NAME = "aind-pophys-converter"

_CODE_NAME = "AIND Pophys Converter"

#: ``DataProcess.name``. Set explicitly because ``ophys_nwb`` also uses
#: ``ProcessName.FILE_FORMAT_CONVERSION``, so both would otherwise derive the
#: same default name ("File format conversion") and the run-level aggregator
#: rejects a merged document with duplicate process names. Both capsules run
#: once per session, so a plane id cannot separate them.
DATA_PROCESS_NAME = "Raw movie conversion to HDF5"


def build_converter_code(
    job_parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Build the v2 ``Code`` block describing this conversion.

    The ``DataProcess`` and ``Processing`` documents around it are assembled
    by ``aind_pophys_metadata.runner.stage_guard``, which also supplies the
    timings, the static resource capture and the pipeline identity from the
    ``PIPELINE_*`` environment variables.

    Parameters
    ----------
    job_parameters : dict
        Dict of CLI-driven JobSettings fields (input_dir, output_dir,
        temp_dir, debug). Caller produces this via
        ``job_settings.model_dump(mode="json")``.
    input_data : list of str, optional
        Raw source files this run read, as paths relative to the mounted
        data root (see ``job.raw_input_data``).

    Returns
    -------
    Code
        The populated code block.

    Raises
    ------
    ValueError
        If ``job_parameters`` or ``input_data`` carry an ephemeral Nextflow
        task scratch path.
    """
    return shared_processing.build_code(
        name=_CODE_NAME,
        library_name=LIBRARY_NAME,
        parameters=job_parameters,
        input_data=input_data,
    )
