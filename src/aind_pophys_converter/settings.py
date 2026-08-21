"""Job settings for aind-pophys-converter."""

import os
from pathlib import Path
from typing import Dict, Optional, Union

from pydantic import Field
from pydantic_settings import BaseSettings

#: Process identity recorded on every structured log line.
PROCESS_NAME = "aind-pophys-converter"

#: Exported by the pipeline runtime; unset for a standalone capsule run.
PIPELINE_NAME_ENV = "PIPELINE_NAME"


def build_log_fields(acquisition_name: str) -> Dict[str, str]:
    """Build the static fields attached to every structured log line.

    Field names follow the AIND logging standard (``log_schema.models
    .LogFields``). They are passed as a plain dict because the pinned
    log-schema revision predates that model; switch to ``LogFields`` once a
    revision carries both the model and ``setup_logging(use_cloudwatch=...)``.

    Parameters
    ----------
    acquisition_name : str
        Name of the acquisition being converted.

    Returns
    -------
    dict of str to str
        ``acquisition_name`` and ``process_name`` always, plus
        ``pipeline_name`` when the pipeline runtime exported one. Absent
        values are omitted rather than emitted as null.
    """
    fields = {
        "acquisition_name": acquisition_name,
        "process_name": PROCESS_NAME,
    }
    pipeline_name = os.getenv(PIPELINE_NAME_ENV)
    if pipeline_name:
        fields["pipeline_name"] = pipeline_name
    return fields


class JobSettings(BaseSettings, cli_parse_args=True):
    """Settings shared by the converter job and splitter CLI.

    Parameters
    ----------
    input_dir : Union[Path, str]
        Directory where input files are found.
    output_dir : Union[Path, str]
        Directory where outputs are saved.
    temp_dir : Optional[Path]
        Temporary directory for intermediate files.
    debug : bool
        Clip movie processing when true.
    """

    input_dir: Union[Path, str] = Field(
        description="directory where input files are found"
    )
    output_dir: Union[Path, str] = Field(
        description="directory where outputs are saved"
    )
    temp_dir: Optional[Path] = Field(
        default=None,
        description="temporary directory for intermediate files",
    )
    debug: bool = Field(
        default=False,
        description="clip movie processing when true",
    )
