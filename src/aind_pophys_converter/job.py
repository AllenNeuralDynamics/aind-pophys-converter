"""Top-level job orchestration for aind-pophys-converter."""

import logging
from pathlib import Path
from typing import List, Optional

from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import find_one, relative_to_root
from aind_pophys_metadata.processing import collect_upstream_process_names
from aind_pophys_metadata.runner import stage_guard
from log_schema.main import setup_logging

from aind_pophys_converter.bergamo_stitcher import (
    BergamoSettings,
    BergamoTiffStitcher,
)
from aind_pophys_converter.mesoscope_splitter import (
    TiffSplitterCLI,
    find_split_directories,
)
from aind_pophys_converter.qc import (
    finalize_quality_control,
    pair_depth_tifs_with_avg_depth_pngs,
    write_avg_depth_slices,
)
from aind_pophys_converter.settings import JobSettings, build_log_fields
from aind_pophys_converter.tiff_splitters import AvgImageTiffSplitter
from aind_pophys_converter.utils.metadata_utils import (
    DATA_PROCESS_NAME,
    build_converter_code,
    load_converter_metadata,
)
from aind_pophys_converter.utils.platform_utils import (
    is_child_session_via_platform_json,
)

# instrument_id routing signals: Bergamo matched as a substring, mesoscope
# rigs matched by the MESO.* prefix.
_BERGAMO_INSTRUMENT = "Bergamo"
_MESOSCOPE_INSTRUMENT_PREFIX = "MESO"

_POPHYS_DIR_PATTERN = "pophys/"

# Raw source files each conversion route reads out of the pophys directory.
# This is the converter's ``code.input_data`` provenance: it is the only
# process in the pipeline that consumes the raw asset directly, so nothing
# else records what it read.
_BERGAMO_RAW_PATTERNS = ("*.tif",)
_MESOSCOPE_RAW_PATTERNS = (
    "*platform.json",
    "*timeseries.tiff",
    "*averaged_depth.tiff",
    "*averaged_surface.tiff",
    "*local_z*.tiff",
    "*fullfield*.tiff",
    "*_vasculature.tif",
)


def raw_input_data(
    input_dir: Path,
    pophys_dir: Optional[Path],
    instrument_id: str,
) -> Optional[List[str]]:
    """List the raw source files this run reads, for ``code.input_data``.

    Paths are returned relative to the mounted data root, so they name the
    file inside its data asset rather than inside the task container: an
    absolute container path is meaningless once the task is gone.

    Never raises. Discovery here is provenance only; the conversion itself
    re-resolves the files it needs and fails loudly there, inside the stage
    guard, so a malformed input still produces a ``processing.json``.

    Parameters
    ----------
    input_dir : Path
        Mounted data root; every returned path is relative to it.
    pophys_dir : Path or None
        The located ``pophys`` directory, or ``None`` when absent.
    instrument_id : str
        Instrument identifier, used to pick the route's file patterns.

    Returns
    -------
    list of str or None
        Sorted data-root-relative paths, or ``None`` when nothing matched
        (the caller then falls back to the mounted asset names).
    """
    if pophys_dir is None:
        return None
    patterns = (
        _BERGAMO_RAW_PATTERNS
        if _BERGAMO_INSTRUMENT in instrument_id
        else _MESOSCOPE_RAW_PATTERNS
    )
    found = {
        relative_to_root(input_dir, match)
        for pattern in patterns
        for match in Path(pophys_dir).glob(pattern)
    }
    return sorted(found) or None


def run() -> None:
    """Entry point for the pophys converter capsule."""
    job_settings = JobSettings()
    input_dir = Path(job_settings.input_dir)
    output_dir = Path(job_settings.output_dir)

    metadata = load_converter_metadata(input_dir)

    setup_logging(
        use_cloudwatch=True,
        model=build_log_fields(
            metadata.dataset_name or metadata.subject_id
        ),
    )

    pophys_dir = find_one(input_dir, _POPHYS_DIR_PATTERN, required=False)
    code = build_converter_code(
        job_parameters=job_settings.model_dump(mode="json"),
        input_data=(
            raw_input_data(input_dir, pophys_dir, metadata.instrument_id)
            or _mounted_input_assets(input_dir)
        ),
    )

    with stage_guard(
        output_dir,
        ProcessName.FILE_FORMAT_CONVERSION,
        code,
        name=DATA_PROCESS_NAME,
    ) as ctx:
        # The converter is the pipeline's first step, so in a normal run no
        # upstream document exists and the graph is legitimately empty.
        ctx.upstream_names = collect_upstream_process_names(
            input_dir, exclude=DATA_PROCESS_NAME
        )
        if pophys_dir is None:
            raise FileNotFoundError(
                f"No {_POPHYS_DIR_PATTERN} directory found under {input_dir}"
            )
        if _BERGAMO_INSTRUMENT in metadata.instrument_id:
            bergamo_settings = BergamoSettings(
                input_dir=pophys_dir,
                output_dir=output_dir,
                debug=job_settings.debug,
            )
            bergamo_stitcher = BergamoTiffStitcher(
                bergamo_settings, metadata=metadata
            )
            bergamo_stitcher.run_converter()
        elif metadata.instrument_id.startswith(_MESOSCOPE_INSTRUMENT_PREFIX):
            # recursive=False: rglob does not follow symlinks in Python
            # <=3.12 and pophys/ is a symlink in dev.
            platform_fp = find_one(
                pophys_dir, "*platform.json", recursive=False
            )
            job_settings.input_dir = pophys_dir
            split_directories = find_split_directories(pophys_dir)
            if len(split_directories) == 0:
                runner = TiffSplitterCLI(job_settings, metadata=metadata)
                runner.run_job()
            else:
                for split_dir in split_directories:
                    new_directory = output_dir / split_dir
                    new_directory.mkdir(parents=True, exist_ok=True)
                    with open(new_directory / f"{split_dir}.txt", "w") as f:
                        f.write(f"{split_dir}.h5")
            qc_metrics = []
            avg_depth_path = find_one(
                pophys_dir,
                "*_averaged_depth.tiff",
                recursive=False,
                required=False,
            )
            if avg_depth_path is not None:
                logging.info(
                    f"Processing averaged depth TIFF:"
                    f" {avg_depth_path} -> {output_dir}"
                )
                if is_child_session_via_platform_json(platform_fp):
                    splitter = AvgImageTiffSplitter(avg_depth_path)
                    # Averaged-depth slices are scratch: they are matched to
                    # parent snapshots and consumed. Write them to temp_dir
                    # (the scratch mount) when available so they never land in
                    # the published results, falling back to output_dir only
                    # when no scratch dir was provided.
                    avg_slice_dir = (
                        Path(job_settings.temp_dir) / "averaged_depths"
                        if job_settings.temp_dir
                        else output_dir
                    )
                    write_avg_depth_slices(splitter, avg_slice_dir)
                    qc_metrics.extend(
                        pair_depth_tifs_with_avg_depth_pngs(
                            pophys_dir,
                            avg_slice_dir,
                            platform_fp,
                            output_dir / "matched_tiff_vals",
                        )
                    )
            finalize_quality_control(qc_metrics, pophys_dir, output_dir)


def _mounted_input_assets(input_dir: Path) -> Optional[list]:
    """Return the names of the data assets mounted under ``input_dir``.

    Each immediate subdirectory of the capsule's data directory is one
    attached asset, so their names are a coarse input-data provenance. Used
    only as the fallback for :func:`raw_input_data`, which names the
    individual source files. Returns ``None`` when nothing is mounted or the
    directory is unreadable.
    """
    try:
        names = sorted(p.name for p in Path(input_dir).iterdir() if p.is_dir())
    except OSError:
        return None
    return names or None
