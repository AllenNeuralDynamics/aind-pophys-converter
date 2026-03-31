"""This module provides functionality to stitch together multiple TIFF files"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime as dt
from pathlib import Path
from typing import List, Tuple

import h5py as h5
import numpy as np
from pydantic import BaseModel, Field
from ScanImageTiffReader import ScanImageTiffReader

from aind_pophys_converter.utils.logging import setup_logging


class BergamoSettings(BaseModel):
    """Settings required to stitch Bergamo images"""

    input_dir: Path = Field(description="directory of tiff files")
    output_dir: Path = Field(description="where to save the hdf5 file")
    unique_id: str = Field(
        description="name for data (how it relates to the experiment)"
    )
    acquisition_fp: Path = Field(description="path to the acquisition file")
    debug: bool = Field(
        default=False,
        description=(
            "If True, process only the first tiff file per epoch "
            "instead of the full dataset."
        ),
    )


class BaseStitcher:
    """Base class for stitching TIFF files into a single HDF5 file"""

    def __init__(self, bergamo_settings: BergamoSettings) -> None:
        """Initialize the stitcher with settings"""
        self.input_dir = bergamo_settings.input_dir
        self.output_dir = bergamo_settings.output_dir
        self.unique_id = bergamo_settings.unique_id
        self.acquisition_fp = bergamo_settings.acquisition_fp
        self.debug = bergamo_settings.debug

    def write_images(
        self,
        image_data_to_cache: List[np.ndarray],
        initial_frame: int,
        cache_filepath: Path,
    ) -> None:
        """
        Cache images to disk

        Parameters
        ----------
        image_data_to_cache : List[np.ndarray]
            A list of image data to cache
        initial_frame : int
            The index of the first frame to concatenate to the array
        cache_filepath : Path
            The filepath to the cache file (should be a temporary file)

        Returns
        -------
        """
        with h5.File(cache_filepath, "a") as f:
            f["data"].resize(initial_frame + len(image_data_to_cache), axis=0)
            f["data"][
                initial_frame : initial_frame + len(image_data_to_cache)
            ] = image_data_to_cache

    def write_final_output(
        self,
        output_filepath: Path,
        **kwargs: dict,
    ) -> None:
        """Writes the final output to disk along with the metadata.
        Clears the temporary
        hdf5 data file

        Parameters
        ----------
        output_filepath : Path
            The filepath to the output file
        **kwargs : dict
            The metadata to write to disk

        Returns
        -------
        None
        """

        # b/c this seems to take a long time
        logging.info("Writing final output file")
        start_time = dt.now()
        with h5.File(output_filepath, "a") as f:
            for key, value in kwargs.items():
                meta_size = 1
                f.create_dataset(
                    key,
                    (meta_size),
                    maxshape=(meta_size,),
                    dtype=h5.special_dtype(vlen=str),
                )
                f[key].resize(meta_size, axis=0)
                f[key][:] = value
        total_time = (dt.now() - start_time).seconds
        logging.info(f"{total_time} seconds to write data")
        logging.info("Finished...")


class BergamoTiffStitcher(BaseStitcher):
    """Class to stitch together TIFF files from a Bergamo session"""

    def __init__(self, bergamo_settings: BergamoSettings) -> None:
        """Initialize the BergamoTiffStitcher with settings"""
        super().__init__(bergamo_settings)

    def _discover_tiff_epochs(self) -> dict:
        """Discover tiff epochs by grouping *.tif files in input_dir by stem.

        Groups files by their filename stem (everything before the trailing
        ``_NNNNN`` numeric suffix). Files whose stem is ``stack`` are
        excluded as z-stack calibration scans. Within each group files
        are sorted by their numeric suffix.

        Returns
        -------
        dict
            ``{stem: {"tiff_files": [sorted absolute path strings]}}``

        Raises
        ------
        ValueError
            If no eligible tiff files are found in input_dir.
        """
        _suffix = re.compile(r"^(.+)_(\d+)$")
        groups: dict = defaultdict(list)

        for tif in self.input_dir.glob("*.tif"):
            m = _suffix.match(tif.stem)
            stem = m.group(1) if m else tif.stem
            if stem == "stack":
                continue
            groups[stem].append(tif)

        if not groups:
            raise ValueError(
                f"No tiff files found in {self.input_dir} "
                "(excluding 'stack')"
            )

        def _sort_key(p: Path) -> int:
            """Extract numeric suffix for sorting."""
            m = _suffix.match(p.stem)
            return int(m.group(2)) if m else 0

        epochs = {
            stem: {
                "tiff_files": [str(p) for p in sorted(files, key=_sort_key)]
            }
            for stem, files in sorted(groups.items())
        }
        if self.debug:
            logging.info(
                "Debug mode: limiting to 1 file per epoch "
                "(was %s files total).",
                sum(len(v["tiff_files"]) for v in epochs.values()),
            )
            epochs = {
                stem: {"tiff_files": v["tiff_files"][:1]}
                for stem, v in epochs.items()
            }
        return epochs

    def _get_image_dim(self) -> Tuple[int, ...]:
        """Grab image shape from metdata"""
        file_path = next(self.input_dir.glob("*.tif"))
        with ScanImageTiffReader(str(file_path)) as reader:
            return reader.shape()[1:]

    def write_bergamo(
        self,
        epochs: dict,
        image_width: int = 800,
        image_height: int = 800,
    ) -> Path:
        """
        Reads in a list of tiff files from a specified path (initialized above)
        and converts them to a single h5 file

        Parameters
        ----------
        epochs : dict
            A dictionary containg the sorted epochs
        image_width : int, optional
            The width of the image, by default 800
        image_height : int, optional
            The height of the image, by default 800

        Returns
        -------
        Path
            converted filepath
        tuple
            image shape
        """
        start_time = dt.now()
        epoch_count = 0
        start_epoch_count = 0
        trial_counter = 0
        header_data = {}
        output_filepath = self.output_dir / f"{self.unique_id}.h5"
        with h5.File(output_filepath, "w") as f:
            f.create_dataset(
                "data",
                (0, image_width, image_height),
                chunks=True,
                maxshape=(None, image_width, image_height),
                dtype="int16",
            )
        # metadata dictionary that keeps track of the epoch name and the
        # location of the epoch image in the stack
        epoch_location = {}
        trial_locations = {}
        image_shape = None
        for epoch in epochs.keys():
            header_count = 0
            for filename in epochs[epoch]["tiff_files"]:
                epoch_name = "_".join(
                    os.path.basename(filename).split("_")[:-1]
                )
                image_data = ScanImageTiffReader(filename).data()
                if header_count == 0:
                    header_data[epoch_name] = ScanImageTiffReader(
                        filename
                    ).metadata()
                image_shape = image_data.shape
                frame_count = image_shape[0]
                self.write_images(image_data, epoch_count, output_filepath)
                epoch_count += frame_count
                trial_locations[os.path.basename(filename)] = [
                    trial_counter,
                    (trial_counter + frame_count) - 1,
                ]
                trial_counter += frame_count
            epoch_location[epoch_name] = [start_epoch_count, epoch_count - 1]
            start_epoch_count = epoch_count
        self.write_final_output(
            output_filepath,
            trial_locations=json.dumps(trial_locations),
            epoch_locations=json.dumps(epoch_location),
            epoch_filenames=json.dumps(epochs),
            metadata=json.dumps(header_data),
        )
        total_time = dt.now() - start_time
        logging.info("Time to cache %s seconds", total_time.total_seconds())
        if image_shape is None:
            raise ValueError("No tiff files were processed")
        return self.output_dir / f"{self.unique_id}.h5", image_shape

    def run_converter(self) -> Path:
        """
        Reads in a list of tiff files from a specified path (initialized above)
        and converts them to a single h5 file. Writes relevant metadata to the
        h5 file.

        Returns
        -------
        Path
            converted filepath
        """
        # Convert the file and build the final metadata structure
        epochs = self._discover_tiff_epochs()
        shape = self._get_image_dim()
        # metadata dictionary where the keys are the image filename and
        # the values are the index of the order in which the image was
        # read, which epoch it's associated with, the location of the
        # image in the h5 stack and the image shape
        output_filepath = self.write_bergamo(
            epochs=epochs,
            image_width=shape[0],
            image_height=shape[1],
        )
        return output_filepath

    @classmethod
    def from_args(cls, args: List[str]) -> "BergamoTiffStitcher":
        """
        Adds ability to construct settings from a list of arguments.
        Parameters
        ----------
        args : list
        A list of command line arguments to parse.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-i",
            "--input-dir",
            required=True,
            type=str,
            help=("""
                data-directory for bergamo settings
                """),
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            required=False,
            default=None,
            type=str,
            help=("""
                output-directory for bergamo settings
                """),
        )
        parser.add_argument(
            "-u",
            "--unique-id",
            required=False,
            default=None,
            type=str,
            help=("""
                unique name for h5 file
                """),
        )
        parser.add_argument(
            "-s",
            "--acquisition-file",
            required=True,
            type=str,
            help=("""
                path to the acquisition file
                """),
        )
        job_args = parser.parse_args(args)

        # Provide default values for optional arguments
        output_dir = (
            Path(job_args.output_dir)
            if job_args.output_dir
            else Path(job_args.input_dir)
        )
        unique_id = (
            job_args.unique_id if job_args.unique_id else "bergamo_stitched"
        )

        job_settings = BergamoSettings(
            input_dir=Path(job_args.input_dir),
            output_dir=output_dir,
            unique_id=unique_id,
            acquisition_fp=Path(job_args.acquisition_file),
        )
        return cls(
            job_settings,
        )


if __name__ == "__main__":
    setup_logging(
        "aind-pophys-converter", process_name="aind-pophys-converter"
    )
    sys_args = sys.argv[1:]
    runner = BergamoTiffStitcher.from_args(sys_args)
    runner.run_converter()
