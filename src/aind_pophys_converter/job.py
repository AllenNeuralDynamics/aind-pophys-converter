"""Top-level job entrypoint for aind-pophys-converter."""

import json
from pathlib import Path

from aind_pophys_converter.bergamo_stitcher import (
    BergamoSettings,
    BergamoTiffStitcher,
)
from aind_pophys_converter.mesoscope_splitter import (
    TiffSplitterCLI,
    find_split_directories,
)
from aind_pophys_converter.settings import JobSettings


def run() -> None:
    """Route one conversion job through the reusable library entrypoint.

    This initial library boundary deliberately retains the v1/session input
    behavior. Schema-version dispatch and output metadata are introduced by
    later stack layers.
    """
    job_settings = JobSettings()
    input_dir = Path(job_settings.input_dir)
    output_dir = Path(job_settings.output_dir)

    session_fp = next(input_dir.rglob("*session.json"), None)
    if session_fp is None:
        raise FileNotFoundError(f"No session.json file found under {input_dir}")
    with open(session_fp) as f:
        session_data = json.load(f)

    pophys_dir = next(input_dir.rglob("pophys"), None)
    if pophys_dir is None:
        raise FileNotFoundError(f"No pophys directory found under {input_dir}")

    instrument_id = session_data.get("instrument_id", "")
    if "Bergamo" in instrument_id:
        settings = BergamoSettings(
            input_dir=pophys_dir,
            output_dir=output_dir,
            unique_id=input_dir.name,
            session_fp=session_fp,
            debug=job_settings.debug,
        )
        BergamoTiffStitcher(settings).run_converter()
        return

    job_settings.input_dir = pophys_dir
    split_directories = find_split_directories(pophys_dir)
    if not split_directories:
        TiffSplitterCLI(job_settings).run_job()
        return

    for split_dir in split_directories:
        new_directory = output_dir / split_dir
        new_directory.mkdir(parents=True, exist_ok=True)
        with open(new_directory / f"{split_dir}.txt", "w") as f:
            f.write(f"{split_dir}.h5")
