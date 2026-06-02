"""Top-level job orchestration for aind-pophys-converter."""
import logging
from datetime import datetime, timezone
from importlib import resources
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
    create_vasculature,
    pair_depth_tifs_with_avg_depth_pngs,
    write_avg_depth_slices,
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

_LOGGING_CONFIG = str(
    resources.files("aind_pophys_converter.utils") / "logging.yml"
)


def run() -> None:
    """Entry point for the pophys converter capsule."""
    job_settings = JobSettings()
    input_dir = Path(job_settings.input_dir)
    output_dir = Path(job_settings.output_dir)

    metadata = load_converter_metadata(input_dir)

    setup_logging(
        path_to_config=_LOGGING_CONFIG,
        model={
            "acquisition_name": metadata.dataset_name,
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
        if "Bergamo" in metadata.instrument_id:
            bergamo_settings = BergamoSettings(
                input_dir=pophys_dir,
                output_dir=output_dir,
                debug=job_settings.debug,
            )
            bergamo_stitcher = BergamoTiffStitcher(
                bergamo_settings, metadata=metadata
            )
            bergamo_stitcher.run_converter()
        elif "multiplane" in metadata.dataset_name:
            # Use pophys_dir.glob (not input_dir.rglob) because rglob does
            # not follow symlinks in Python <=3.12 and pophys/ is a symlink
            # in dev.
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
                    pair_depth_tifs_with_avg_depth_pngs(
                        pophys_dir,
                        output_dir,
                        platform_fp,
                        Path("/results/matched_tiff_vals"),
                    )
            create_vasculature(pophys_dir, output_dir)
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
        )
        write_processing_json(processing, output_dir)
    except Exception as proc_exc:
        logging.error(
            f"processing.json write failed (non-fatal): {proc_exc}"
        )
