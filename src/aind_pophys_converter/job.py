"""Top-level job orchestration for aind-pophys-converter."""
import logging
from pathlib import Path

from aind_data_schema.core.acquisition import Acquisition
from aind_data_schema.core.data_description import DataDescription

from aind_pophys_converter.bergamo_stitcher import (
    BergamoSettings,
    BergamoTiffStitcher,
)
from aind_pophys_converter.mesoscope_splitter import (
    AvgImageTiffSplitter,
    TiffSplitterCLI,
    find_split_directories,
)
from aind_pophys_converter.qc import (
    create_vasculature,
    pair_depth_tifs_with_avg_depth_pngs,
    write_avg_depth_slices,
)
from aind_pophys_converter.settings import JobSettings
from aind_pophys_converter.utils.platform_utils import (
    is_child_session_via_platform_json,
)


def run() -> None:
    """Entry point for the pophys converter capsule."""
    job_settings = JobSettings()
    input_dir = Path(job_settings.input_dir)
    output_dir = Path(job_settings.output_dir)
    acquisition_fp = next(input_dir.rglob("acquisition.json"))
    data_description_fp = next(input_dir.rglob("data_description.json"))

    acquisition = Acquisition.model_validate_json(acquisition_fp.read_text())
    data_description = DataDescription.model_validate_json(
        data_description_fp.read_text()
    )
    pophys_dir = next(input_dir.rglob("pophys/"))
    if "Bergamo" in acquisition.instrument_id:
        # pull this from targetted structure in acquisition

        unique_id = "MOp2_3_0"  # TODO: read from CCF when available
        output_dir = output_dir / unique_id
        output_dir.mkdir(exist_ok=True)
        bergamo_settings = BergamoSettings(
            input_dir=pophys_dir,
            output_dir=output_dir,
            unique_id=unique_id,
            acquisition_fp=acquisition_fp,
            debug=job_settings.debug,
        )
        bergamo_stitcher = BergamoTiffStitcher(bergamo_settings)
        bergamo_stitcher.run_converter()
    elif "multiplane" in data_description.name:
        # Use pophys_dir.glob (not input_dir.rglob) because rglob does not
        # follow symlinks in Python <=3.12 and pophys/ is a symlink in dev.
        platform_fp = next(pophys_dir.glob("*platform.json"), None)
        if platform_fp is None:
            raise FileNotFoundError(
                f"No platform.json file found in {input_dir}"
            )
        job_settings.input_dir = pophys_dir
        split_directories = find_split_directories(pophys_dir)
        if len(split_directories) == 0:
            runner = TiffSplitterCLI(job_settings)
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
            print(
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
