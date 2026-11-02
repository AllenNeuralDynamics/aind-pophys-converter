"""This module provides functionality to stitch together multiple TIFF files"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime as dt
from pathlib import Path
from typing import List, Tuple

import h5py as h5
import numpy as np
from aind_pophys_metadata import find_one
from pydantic import BaseModel, Field
from ScanImageTiffReader import ScanImageTiffReader

from aind_pophys_converter.utils.metadata_utils import (
    ConverterMetadata,
    load_converter_metadata,
)


class BergamoSettings(BaseModel):
    """Settings required to stitch Bergamo images"""

    input_dir: Path = Field(description="directory of tiff files")
    output_dir: Path = Field(description="where to save the hdf5 file")
    debug: bool = Field(
        default=False,
        description=(
            "If True, process only the first tiff file per epoch "
            "instead of the full dataset."
        ),
    )


class BaseStitcher:
    """Base class for stitching TIFF files into a single HDF5 file"""

    def __init__(
        self,
        bergamo_settings: BergamoSettings,
        metadata: ConverterMetadata,
    ) -> None:
        """Initialize the stitcher with settings and converter metadata.

        Parameters
        ----------
        bergamo_settings : BergamoSettings
            CLI-driven settings (input/output dirs, debug flag).
        metadata : ConverterMetadata
            Schema-agnostic metadata loaded from v1 or v2 input.
        """
        self.input_dir = bergamo_settings.input_dir
        self.output_dir = bergamo_settings.output_dir
        self.debug = bergamo_settings.debug
        self.metadata = metadata

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

    def __init__(
        self,
        bergamo_settings: BergamoSettings,
        metadata: ConverterMetadata,
    ) -> None:
        """Initialize the BergamoTiffStitcher with settings and metadata."""
        super().__init__(bergamo_settings, metadata=metadata)

    def _discover_tiff_epochs(self) -> dict:
        """Resolve stimulus epochs to their tiff files, in acquisition order.

        Iterates the stimulus epochs recorded in the acquisition metadata,
        ordered by start time (``metadata.epoch_stems``), and resolves each
        to the ``*.tif`` files on disk whose filename stem (everything before
        the trailing ``_NNNNN`` numeric suffix) matches, sorted by numeric
        suffix. Files on disk whose stem no epoch names -- including the
        ``stack`` z-stack calibration scans -- are ignored.

        Returns
        -------
        dict
            ``{stem: {"tiff_files": [sorted absolute path strings]}}``,
            insertion-ordered by acquisition start time.

        Raises
        ------
        ValueError
            If the acquisition metadata records no stimulus epochs, or if an
            epoch's tiff files are missing from input_dir.
        """
        if not self.metadata.epoch_stems:
            raise ValueError(
                "No stimulus epochs found in the acquisition metadata for "
                f"{self.input_dir}; cannot stitch Bergamo epochs."
            )

        _suffix = re.compile(r"^(.+)_(\d+)$")
        groups: dict = defaultdict(list)

        for tif in self.input_dir.glob("*.tif"):
            m = _suffix.match(tif.stem)
            stem = m.group(1) if m else tif.stem
            groups[stem].append(tif)

        def _sort_key(p: Path) -> int:
            """Extract numeric suffix for sorting."""
            m = _suffix.match(p.stem)
            return int(m.group(2)) if m else 0

        epochs = {}
        for stem in self.metadata.epoch_stems:
            if stem not in groups:
                raise ValueError(
                    f"Stimulus epoch '{stem}' from the acquisition metadata "
                    f"has no tiff files in {self.input_dir}."
                )
            epochs[stem] = {
                "tiff_files": [
                    str(p) for p in sorted(groups[stem], key=_sort_key)
                ]
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
        file_path = find_one(self.input_dir, "*.tif", recursive=False)
        with ScanImageTiffReader(str(file_path)) as reader:
            return reader.shape()[1:]

    def write_bergamo(
        self,
        epochs: dict,
        unique_id: str,
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
        unique_id : str
            Name for the output HDF5 file (without extension).
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
        output_filepath = self.output_dir / f"{unique_id}.h5"
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
        return self.output_dir / f"{unique_id}.h5", image_shape

    def run_converter(self) -> Path:
        """
        Reads in a list of tiff files from a specified path (initialized above)
        and converts them to a single h5 file. Writes relevant metadata to the
        h5 file. The output filename is derived from the acquisition metadata.

        Returns
        -------
        Path
            converted filepath
        """
        if not self.metadata.fov_ids:
            raise ValueError(
                f"No imaging planes found in {self.input_dir}; cannot "
                "derive the Bergamo output name. Check that the "
                "acquisition/session metadata defines an imaging FOV/plane."
            )
        unique_id = self.metadata.fov_ids[0]
        self.output_dir = self.output_dir / unique_id
        self.output_dir.mkdir(exist_ok=True)
        # Convert the file and build the final metadata structure
        epochs = self._discover_tiff_epochs()
        shape = self._get_image_dim()
        # metadata dictionary where the keys are the image filename and
        # the values are the index of the order in which the image was
        # read, which epoch it's associated with, the location of the
        # image in the h5 stack and the image shape
        output_filepath = self.write_bergamo(
            epochs=epochs,
            unique_id=unique_id,
            image_width=shape[0],
            image_height=shape[1],
        )
        return output_filepath

    @classmethod
    def from_args(cls, args: List[str]) -> "BergamoTiffStitcher":
        """Construct a stitcher from command line arguments.

        Parameters
        ----------
        args : list of str
            CLI arguments. Expects ``--input-dir`` (tiff directory; its
            parent is searched for v1 ``session.json`` or v2
            ``acquisition.json``), and optionally ``--output-dir``
            (defaults to input directory) and ``--metadata-dir``
            (overrides where metadata files are looked up; defaults to
            ``input-dir.parent``).
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-i",
            "--input-dir",
            required=True,
            type=str,
            help="directory containing the tiff files to stitch",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            required=False,
            default=None,
            type=str,
            help="directory to write outputs into",
        )
        parser.add_argument(
            "-m",
            "--metadata-dir",
            required=False,
            default=None,
            type=str,
            help=(
                "directory to search for acquisition.json / session.json"
                " and data_description.json (defaults to input-dir.parent)"
            ),
        )
        job_args = parser.parse_args(args)

        input_dir = Path(job_args.input_dir)
        output_dir = (
            Path(job_args.output_dir) if job_args.output_dir else input_dir
        )
        metadata_dir = (
            Path(job_args.metadata_dir)
            if job_args.metadata_dir
            else input_dir.parent
        )
        metadata = load_converter_metadata(metadata_dir)
        job_settings = BergamoSettings(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        return cls(job_settings, metadata=metadata)
