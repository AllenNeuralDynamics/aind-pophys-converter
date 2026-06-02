"""Mesoscope TIFF splitting CLI and orchestration."""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

from aind_pophys_converter.settings import JobSettings
from aind_pophys_converter.tiff_splitters import (
    AvgImageTiffSplitter,
    TiffSplitterBase,
    TimeSeriesSplitter,
    ZStackSplitter,
    get_nearest_roi_center,
    get_valid_roi_centers,
)
from aind_pophys_converter.utils.full_field_utils import (
    write_out_stitched_full_field_image,
)
from aind_pophys_converter.utils.metadata_utils import (
    ConverterMetadata,
    PlaneInfo,
    fov_id_from_plane,
    load_converter_metadata,
)
from aind_pophys_converter.utils.tiff_metadata import ScanImageMetadata


def find_split_directories(directory: Path) -> List[Path]:
    """Determine if there are any V* directories in the specified directory.

    Parameters
    ----------
    directory: Path
        The directory to search for V* directories.

    Returns
    -------
    List[Path]
        A list of Path objects representing the V* directories found in the
        specified directory.
    """
    # Search for directories of the pattern V*_[0-9]
    pattern = re.compile(r"^V.*\d+$")

    matching_dirs = [
        d.name
        for d in directory.iterdir()
        if d.is_dir() and pattern.match(d.name)
    ]
    return matching_dirs


class TiffSplitterCLI:
    """A command line interface for splitting TIFF files"""

    @staticmethod
    def str2bool(v: str) -> bool:
        """Convert a string to a boolean value."""
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        else:
            return False

    def __init__(
        self,
        job_settings: JobSettings,
        metadata: ConverterMetadata,
    ) -> None:
        """Initialize the TiffSplitterCLI with job settings and metadata.

        Parameters
        ----------
        job_settings : JobSettings
            CLI-driven settings (input/output/temp dirs, debug flag).
        metadata : ConverterMetadata
            Schema-agnostic metadata loaded by ``load_converter_metadata``.
        """
        self.input_dir = job_settings.input_dir
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        self.output_dir = job_settings.output_dir
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.metadata = metadata
        self.timeseries_tif = next(
            self.input_dir.glob("*timeseries.tiff"), None
        )
        if not self.timeseries_tif:
            raise ValueError("No timeseries.tiff file found")
        self.depths_tif = next(
            self.input_dir.glob("*averaged_depth.tiff"), None
        )
        if not self.depths_tif:
            raise ValueError("No averaged_depth.tiff file found")
        self.surface_tif = next(
            self.input_dir.glob("*averaged_surface.tiff"), None
        )
        if not self.surface_tif:
            raise ValueError("No averaged_surface.tiff file found")
        self.temp_dir = job_settings.temp_dir
        if self.temp_dir is not None:
            # Ensure temp directory exists before mkdtemp is called downstream
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        self.debug = job_settings.debug

    def _initialize_splitters(
        self,
    ) -> Tuple[
        "TimeSeriesSplitter",
        "AvgImageTiffSplitter",
        "AvgImageTiffSplitter",
        "ZStackSplitter",
        List[Path],
        List[Path],
    ]:
        """Initialize all the splitters needed for processing."""
        files_to_record = []

        ts_path = Path(self.timeseries_tif)
        timeseries_splitter = TimeSeriesSplitter(tiff_path=ts_path)
        files_to_record.append(ts_path)

        depth_path = Path(self.depths_tif)
        depth_splitter = AvgImageTiffSplitter(tiff_path=depth_path)
        files_to_record.append(depth_path)

        surface_path = Path(self.surface_tif)
        surface_splitter = AvgImageTiffSplitter(tiff_path=surface_path)
        files_to_record.append(surface_path)

        zstack_path_list = []
        for zstack in self.input_dir.glob("*local_z*.tiff"):
            zstack_path = zstack
            zstack_path_list.append(zstack_path)
            files_to_record.append(zstack_path)

        zstack_splitter = ZStackSplitter(tiff_path_list=zstack_path_list)

        return (
            timeseries_splitter,
            depth_splitter,
            surface_splitter,
            zstack_splitter,
            zstack_path_list,
            files_to_record,
        )

    def _check_deprecated_files(self) -> None:
        """Check for deprecated file types and warn accordingly."""
        if next(self.input_dir.glob("*cortical_z_stack*.tiff"), None):
            msg = "'column_z_stack_tif' detected in 'plane_groups'; "
            msg += "the TIFF splitting code no longer handles that file."
            logging.warning(msg)

    def _create_experiment_metadata(
        self, plane: PlaneInfo
    ) -> Tuple[dict, str]:
        """Create experiment metadata structure for a given imaging plane.

        Parameters
        ----------
        plane : PlaneInfo
            Plane info from the schema-agnostic metadata model.

        Returns
        -------
        tuple[dict, str]
            Metadata dict and the fov_id string.
        """
        fov_id = fov_id_from_plane(plane)
        this_exp_metadata = {"fov": fov_id}

        for file_key in (
            "timeseries",
            "depth_2p",
            "surface_2p",
            "local_z_stack",
        ):
            this_exp_metadata[file_key] = {}

        return this_exp_metadata, fov_id

    def _get_imaging_planes(
        self,
    ) -> Tuple[PlaneInfo, ...]:
        """Return imaging planes sorted by plane_index.

        Returns
        -------
        tuple of PlaneInfo
            Plane info objects from ``self.metadata.planes``.
        """
        return self.metadata.planes

    def _process_single_fov(
        self,
        plane: PlaneInfo,
        roi_index: int,
        z_value: float,
        depth_splitter: "AvgImageTiffSplitter",
        surface_splitter: "AvgImageTiffSplitter",
        zstack_splitter: "ZStackSplitter",
        valid_roi_centers: List[Tuple[float, float]],
        t0: float,
    ) -> dict:
        """Process a single imaging plane and return its metadata."""
        this_exp_metadata, fov_id = self._create_experiment_metadata(plane)
        scanfield_z = z_value
        baseline_center = None

        for splitter, z_value, output_name, metadata_tag in zip(
            (depth_splitter, surface_splitter, zstack_splitter),
            (scanfield_z, None, scanfield_z),
            (
                f"{fov_id}_depth.tif",
                f"{fov_id}_surface.tif",
                f"{fov_id}_z_stack_local.h5",
            ),
            ("depth_2p", "surface_2p", "local_z_stack"),
        ):
            output_dir = self.output_dir / str(fov_id)
            if not output_dir.is_dir():
                output_dir.mkdir()
            output_path = output_dir / output_name

            roi_center = splitter.roi_center(i_roi=roi_index)
            nearest_valid = get_nearest_roi_center(
                this_roi_center=roi_center,
                valid_roi_centers=valid_roi_centers,
            )

            if baseline_center is None:
                baseline_center = nearest_valid

            if nearest_valid != baseline_center:
                self._raise_roi_center_error(
                    fov_id, splitter, output_path, baseline_center, roi_center
                )

            splitter.write_output_file(
                i_roi=roi_index,
                z_value=z_value,
                output_path=output_path,
            )

            self._update_metadata_with_file_info(
                this_exp_metadata,
                metadata_tag,
                output_path,
                splitter,
                roi_index,
                z_value,
            )

            elapsed_time = time.time() - t0
            logging.info(
                f"wrote {output_path.resolve().absolute()} "
                f"after {elapsed_time:.2e} seconds"
            )

        return this_exp_metadata

    def _raise_roi_center_error(
        self,
        fov_id: str,
        splitter: Union[TiffSplitterBase, ZStackSplitter],
        output_path: Path,
        baseline_center: int,
        roi_center: Tuple[float, float],
    ) -> None:
        """Raise an error when ROI centers are inconsistent."""
        msg = f"experiment {fov_id}\n"
        msg += "roi center inconsistent for "
        msg += f"input: {splitter.input_path.resolve().absolute()}\n"
        msg += f"output: {output_path.resolve().absolute()}\n"
        msg += f"{baseline_center}; {roi_center}\n"
        raise RuntimeError(msg)

    def _update_metadata_with_file_info(
        self,
        metadata: dict,
        metadata_tag: str,
        output_path: Path,
        splitter: Union[TiffSplitterBase, ZStackSplitter],
        roi_index: int,
        z_value: float,
    ) -> None:
        """Update metadata with file information."""
        str_path = str(output_path.resolve().absolute())
        metadata[metadata_tag]["filename"] = str_path

        frame_shape = splitter.frame_shape(i_roi=roi_index, z_value=z_value)
        metadata[metadata_tag]["height"] = frame_shape[0]
        metadata[metadata_tag]["width"] = frame_shape[1]

    def _process_timeseries_data(
        self,
        timeseries_splitter: "TimeSeriesSplitter",
        experiment_metadata: List[dict],
    ) -> Dict[Tuple[int, float], Path]:
        """Process timeseries data and return output path lookup."""
        output_path_lookup = {}

        for plane in self._get_imaging_planes():
            # Assumes plane_index order in the v2 acquisition schema matches
            # the sequential scan order encoded in the TIFF's zsAllActuators.
            # In v1, session.json carried scanfield_z explicitly (the raw
            # ScanImage actuator coordinate) providing a direct lookup. In v2,
            # depth is stored in physical micrometers and cannot be matched to
            # the TIFF z-values, so we rely on ordering being preserved by the
            # metadata auto-upgrader.
            roi_z = timeseries_splitter.roi_z_int_manifest[plane.plane_index]
            roi_index, z_int = roi_z
            z_value = timeseries_splitter._z_from_int(z_int)
            _, fov_id = self._create_experiment_metadata(plane)
            fname = f"{fov_id}.h5"
            output_path = self.output_dir / fov_id / fname
            output_path_lookup[(roi_index, z_value)] = output_path

            frame_shape = timeseries_splitter.frame_shape(
                i_roi=roi_index, z_value=z_value
            )

            str_path = str(output_path.resolve().absolute())

            for metadata in experiment_metadata:
                if metadata["fov"] == fov_id:
                    metadata["timeseries"]["height"] = frame_shape[0]
                    metadata["timeseries"]["width"] = frame_shape[1]
                    metadata["timeseries"]["filename"] = str_path

        return output_path_lookup

    def _process_full_field_image(self) -> None:
        """Process full field image if it exists."""
        full_field_path = next(self.input_dir.glob("*fullfield*.tiff"), None)

        if full_field_path is not None:
            avg_path = self.surface_tif
            output_name = "stitched_full_field_img.h5"
            output_path = self.output_dir / output_name
            logging.info(f"Writing {output_path.resolve().absolute()}")

            write_out_stitched_full_field_image(
                path_to_avg_tiff=Path(avg_path),
                path_to_full_field_tiff=full_field_path,
                output_path=output_path,
            )

            if output_path.is_file():
                logging.info(
                    f"Wrote full field stitched image to "
                    f"{output_path.resolve().absolute()}"
                )

    def _create_file_metadata(self, files_to_record: List[Path]) -> List[dict]:
        """Create metadata for recorded files."""
        file_metadata = []
        for file_path in files_to_record:
            tiff_metadata = ScanImageMetadata(file_path)
            this_metadata = {
                "input_tif": str(file_path.resolve().absolute()),
                "scanimage_metadata": tiff_metadata._metadata[0],
                "roi_metadata": tiff_metadata._metadata[1],
            }
            file_metadata.append(this_metadata)
        return file_metadata

    def run_job(self) -> None:
        """Run the job to split TIFF files."""
        t0 = time.time()
        output = {"column_stacks": []}
        try:
            # Initialize splitters and get file paths
            (
                timeseries_splitter,
                depth_splitter,
                surface_splitter,
                zstack_splitter,
                zstack_path_list,
                files_to_record,
            ) = self._initialize_splitters()

            # Check for deprecated file types
            self._check_deprecated_files()

            # Get valid ROI centers for validation
            valid_roi_centers = get_valid_roi_centers(
                timeseries_splitter=timeseries_splitter
            )

            # Process experiment metadata for each imaging plane
            experiment_metadata = []
            for plane in self._get_imaging_planes():
                # See ordering assumption comment in _process_timeseries_data.
                roi_z = timeseries_splitter.roi_z_int_manifest[
                    plane.plane_index
                ]
                roi_index, z_int = roi_z
                z_value = timeseries_splitter._z_from_int(z_int)
                this_exp_metadata = self._process_single_fov(
                    plane,
                    roi_index,
                    z_value,
                    depth_splitter,
                    surface_splitter,
                    zstack_splitter,
                    valid_roi_centers,
                    t0,
                )
                experiment_metadata.append(this_exp_metadata)

            # Process timeseries data separately for efficiency
            output_path_lookup = self._process_timeseries_data(
                timeseries_splitter, experiment_metadata
            )

            # dump_every controls the per-plane frame buffer size before
            # flushing to temp HDF5 files. Historically read from v1
            # session.json via session_data.get("dump_every", 1000);
            # some multiplane-ophys data assets may have carried custom
            # values there. The default of 1000 has negligible runtime
            # impact (bottleneck is TIFF read, not buffer flush) so we
            # hardcode it and rely on write_output_files()'s default.
            timeseries_splitter.write_output_files(
                output_path_map=output_path_lookup,
                tmp_dir=self.temp_dir,
                debug=self.debug,
            )

            # Set up output structure
            output["experiment_output"] = experiment_metadata

            ready_to_archive = {
                str(self.surface_tif),
                str(self.depths_tif),
                str(self.timeseries_tif),
            }
            ready_to_archive.update(
                str(zstack_path.resolve().absolute())
                for zstack_path in zstack_path_list
            )
            output["ready_to_archive"] = list(ready_to_archive)

            # Process full field image if available
            self._process_full_field_image()

            # Create file metadata
            output["file_metadata"] = self._create_file_metadata(
                files_to_record
            )

            duration = time.time() - t0
            logging.info(
                f"Mesoscope TIFF splitting complete in {duration:.2e} seconds"
            )
        except Exception:
            logging.exception(
                "Unhandled exception in mesoscope TIFF splitting"
            )
            raise


def from_args(args: List[str]) -> JobSettings:
    """
    Adds ability to construct settings from a list of arguments.
    Parameters
    ----------
    args : List[str]
    A list of command line arguments to parse.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=str,
        help=("""
            data-directory for job settings
            """),
    )
    parser.add_argument(
        "-t",
        "--temp-dir",
        required=False,
        default=None,
        type=str,
        help=("""
            temp-directory for job settings
            """),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        type=str,
        help=("""
            output-directory for job settings
            """),
    )
    parser.add_argument(
        "-d",
        "--debug",
        required=False,
        default=False,
        type=TiffSplitterCLI.str2bool,
        help="run in debug",
    )

    job_args = parser.parse_args(args)

    return JobSettings(
        input_dir=job_args.input_dir,
        temp_dir=job_args.temp_dir,
        output_dir=job_args.output_dir,
        debug=job_args.debug,
    )


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    job_settings = from_args(sys_args)
    input_dir_path = Path(job_settings.input_dir)
    split_directories = find_split_directories(input_dir_path)
    if len(split_directories) == 0:
        # Metadata lives one level up from the pophys/ tiff directory
        # when invoked standalone (matches the capsule layout).
        metadata = load_converter_metadata(input_dir_path.parent)
        runner = TiffSplitterCLI(job_settings, metadata=metadata)
        runner.run_job()
    else:
        output_dir = Path(job_settings.output_dir)
        for split_dir in split_directories:
            new_directory = output_dir / split_dir
            new_directory.mkdir(parents=True, exist_ok=True)
            with open(new_directory / f"{split_dir}.txt", "w") as f:
                f.write(f"{split_dir}.h5")
