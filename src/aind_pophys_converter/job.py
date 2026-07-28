"""Top-level job orchestration for aind-pophys-converter."""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from log_schema.main import setup_logging

from aind_pophys_converter import __version__
from aind_pophys_converter.bergamo_stitcher import (
    BergamoSettings,
    BergamoTiffStitcher,
)
from aind_pophys_converter.mesoscope_splitter import (
    TiffSplitterCLI,
    find_split_directories,
)
from aind_pophys_converter.qc import (
    build_quality_control,
    create_vasculature,
    pair_depth_tifs_with_avg_depth_pngs,
    write_avg_depth_slices,
    write_quality_control_json,
)
from aind_pophys_converter.settings import JobSettings
from aind_pophys_converter.tiff_splitters import AvgImageTiffSplitter
from aind_pophys_converter.utils.metadata_utils import (
    build_processing,
    load_converter_metadata,
    write_processing_json,
)
from aind_pophys_converter.utils.platform_utils import (
    is_child_session_via_platform_json,
)

# instrument_id routing signals: Bergamo matched as a substring, mesoscope
# rigs matched by the MESO.* prefix.
_BERGAMO_INSTRUMENT = "Bergamo"
_MESOSCOPE_INSTRUMENT_PREFIX = "MESO"


def run() -> None:
    """Entry point for the pophys converter capsule."""
    job_settings = JobSettings()
    input_dir = Path(job_settings.input_dir)
    output_dir = Path(job_settings.output_dir)

    metadata = load_converter_metadata(input_dir)

    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": "aind-pophys-converter",
        },
    )

    started_at = datetime.now(timezone.utc)
    run_error: Optional[BaseException] = None
    logging.info(
        "Starting pophys converter job",
        extra={"event_type": "stage_start"},
    )
    try:
        pophys_dir = next(input_dir.rglob("pophys/"))
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
            # pophys_dir.glob (not input_dir.rglob): rglob does not follow
            # symlinks in Python <=3.12 and pophys/ is a symlink in dev.
            platform_fp = next(pophys_dir.glob("*platform.json"), None)
            if platform_fp is None:
                raise FileNotFoundError(
                    f"No platform.json file found in {input_dir}"
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
            avg_depth_files = list(pophys_dir.glob("*_averaged_depth.tiff"))
            if avg_depth_files:
                avg_depth_path = avg_depth_files[0]
                logging.info(
                    f"Processing averaged depth TIFF:"
                    f" {avg_depth_path} -> {output_dir}"
                )
                if is_child_session_via_platform_json(platform_fp):
                    splitter = AvgImageTiffSplitter(avg_depth_path)
                    write_avg_depth_slices(splitter, output_dir)
                    qc_metrics.extend(
                        pair_depth_tifs_with_avg_depth_pngs(
                            pophys_dir,
                            output_dir,
                            platform_fp,
                            Path("/results/matched_tiff_vals"),
                        )
                    )
            _finalize_quality_control(qc_metrics, pophys_dir, output_dir)
        logging.info(
            "Pophys converter job complete",
            extra={"event_type": "stage_complete"},
        )
    except BaseException as exc:
        run_error = exc
        logging.error(
            "Pophys converter job failed",
            extra={"event_type": "stage_error"},
        )
        logging.exception("Unhandled exception")
        raise
    finally:
        _write_processing_metadata(
            job_settings=job_settings,
            start_time=started_at,
            error=run_error,
            output_dir=output_dir,
        )


def _finalize_quality_control(
    qc_metrics: list,
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


def _mounted_input_assets(input_dir: Path) -> Optional[list]:
    """Return the names of the data assets mounted under ``input_dir``.

    Each immediate subdirectory of the capsule's data directory is one
    attached asset, so their names are the input-data provenance. Returns
    ``None`` when nothing is mounted or the directory is unreadable.
    """
    try:
        names = sorted(p.name for p in Path(input_dir).iterdir() if p.is_dir())
    except OSError:
        return None
    return names or None


def _write_processing_metadata(
    job_settings: JobSettings,
    start_time: datetime,
    error: Optional[BaseException],
    output_dir: Path,
) -> None:
    """Best-effort writer for processing.json. Never raises.

    Failures here must not mask a conversion failure that was already
    propagating out of ``run()``.
    """
    try:
        processing = build_processing(
            job_parameters=job_settings.model_dump(mode="json"),
            library_version=__version__,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            error=error,
            input_data=_mounted_input_assets(job_settings.input_dir),
        )
        write_processing_json(processing, output_dir)
    except Exception as proc_exc:
        logging.error(
            f"processing.json write failed (non-fatal): {proc_exc}"
        )
