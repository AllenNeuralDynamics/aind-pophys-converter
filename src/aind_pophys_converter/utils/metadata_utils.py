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
from aind_pophys_metadata import CoreMetadata
from aind_pophys_metadata import processing as shared_processing

# ---------------------------------------------------------------------------
# Internal metadata model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaneChannel:
    """One plane's imaging channel and coupled partner, as recorded.

    Attributes
    ----------
    plane_index : int
        The plane's index.
    channel_name : str or None
        Imaging channel of the plane's image in ``acquisition.json``;
        ``None`` when the input records none (v1 sessions).
    coupled_plane_index : int or None
        The plane's ``coupled_plane_index`` record field; in
        ``acquisition.json`` the index of its coupled partner.
    """

    plane_index: int
    channel_name: Optional[str]
    coupled_plane_index: Optional[int]


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
        (``aind_pophys_metadata.CoreMetadata.get_fov_ids``) so the
        ids are identical across capsules. Empty only for malformed input
        with no planes (consumers fail-fast on that).
    plane_channels : tuple of PlaneChannel
        Imaging channel and coupled partner of each plane, in ``fov_ids``
        order.
    plane_depths : tuple of (float or None)
        Imaging depth of each plane, in ``fov_ids`` order; ``None`` where
        the input records no depth.
    dataset_name : str or None
        Dataset name from ``data_description.json``; used as the logging
        label. ``None`` when no data_description.
    subject_id : str or None
        Subject identifier; used only as the logging-label fallback.
    epoch_stems : tuple of str
        TIFF stems of the stimulus epochs, ordered by acquisition start
        time. The Bergamo stitcher concatenates epochs in this order so the
        HDF5 stack is chronological; empty when the input declares no
        stimulus epochs.
    """

    instrument_id: str
    fov_ids: Tuple[str, ...]
    plane_channels: Tuple[PlaneChannel, ...] = ()
    plane_depths: Tuple[Optional[float], ...] = ()
    dataset_name: Optional[str] = None
    subject_id: Optional[str] = None
    epoch_stems: Tuple[str, ...] = ()


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
        If no core metadata file is found.
    ValueError
        If both v1 and v2 core files are present (ambiguous input), or if the
        required ``instrument_id`` is missing.
    """
    core = CoreMetadata.load(input_dir)
    records = sorted(core.get_plane_records(), key=lambda r: r["plane_index"])
    return ConverterMetadata(
        instrument_id=core.get_instrument_id(required=True),
        fov_ids=core.get_fov_ids(),
        plane_channels=tuple(
            PlaneChannel(
                plane_index=record["plane_index"],
                channel_name=record["channel_name"],
                coupled_plane_index=record["coupled_plane_index"],
            )
            for record in records
        ),
        plane_depths=tuple(record["depth"] for record in records),
        dataset_name=core.get_dataset_name(),
        subject_id=core.get_subject_id(),
        epoch_stems=_epoch_stems_by_start_time(core),
    )


def _epoch_stems_by_start_time(core: CoreMetadata) -> Tuple[str, ...]:
    """Return stimulus-epoch TIFF stems ordered by acquisition start time.

    Epochs without a resolved TIFF stem are dropped; epochs without a start
    time sort last (their order among themselves is left as recorded).

    Parameters
    ----------
    core : CoreMetadata
        Loaded input metadata.

    Returns
    -------
    tuple of str
        TIFF stems in chronological order.
    """
    epochs = [e for e in core.get_epoch_records() if e["tiff_stem"]]
    epochs.sort(key=lambda e: (e["start_time"] is None, e["start_time"] or ""))
    return tuple(e["tiff_stem"] for e in epochs)


# ---------------------------------------------------------------------------
# Code block
# ---------------------------------------------------------------------------

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
        The populated code block; name, version and url are this library's
        installed package metadata.
    """
    return shared_processing.build_code(
        __name__,
        parameters=job_parameters,
        input_data=input_data,
    )
