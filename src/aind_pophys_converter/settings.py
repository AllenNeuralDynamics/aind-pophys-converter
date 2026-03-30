"""Job settings for aind-pophys-converter."""
from pathlib import Path
from typing import Optional, Union

from pydantic import Field
from pydantic_settings import BaseSettings


class JobSettings(BaseSettings, cli_parse_args=True):
    """Settings for a pophys converter job.

    Parameters
    ----------
    input_dir : Union[Path, str]
        Directory where input files are found.
    output_dir : Union[Path, str]
        Directory where outputs are saved.
    temp_dir : Optional[Path]
        Temporary directory for intermediate files.
    debug : bool
        Clip movie to DEBUG_N_FRAMES per plane when True.
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
        description="clip movie to DEBUG_N_FRAMES per plane when True",
    )
