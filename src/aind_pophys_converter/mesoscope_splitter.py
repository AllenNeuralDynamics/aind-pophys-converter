"""Mesoscope TIFF splitting CLI and orchestration."""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

from aind_pophys_metadata import find_one

from aind_pophys_converter.settings import JobSettings
from aind_pophys_converter.tiff_splitters import (
    AvgImageTiffSplitter,
    TiffSplitterBase,
    TimeSeriesSplitter,
    ZStackSplitter,
    build_roi_remap,
    build_surface_roi_remap,
    get_nearest_roi_center,
    get_valid_roi_centers,
)
from aind_pophys_converter.utils.full_field_utils import (
    write_out_stitched_full_field_image,
)
from aind_pophys_converter.utils.metadata_utils import (
    ConverterMetadata,
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
        # recursive=False throughout: the splitter's inputs sit directly in
        # input_dir, and rglob does not follow the symlink it can be.
        self.timeseries_tif = find_one(
            self.input_dir, "*timeseries.tiff", recursive=False
        )
        self.depths_tif = find_one(
            self.input_dir, "*averaged_depth.tiff", recursive=False
        )
        self.surface_tif = find_one(
            self.input_dir, "*averaged_surface.tiff", recursive=False
        )
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
        if find_one(
            self.input_dir,
            "*cortical_z_stack*.tiff",
            recursive=False,
            required=False,
        ):
            msg = "'column_z_stack_tif' detected in 'plane_groups'; "
            msg += "the TIFF splitting code no longer handles that file."
            logging.warning(msg)

    def _create_experiment_metadata(self, fov_id: str) -> dict:
        """Create the experiment metadata structure for one imaging plane.

        Parameters
        ----------
        fov_id : str
            Canonical FOV id for the plane (from
            ``self.metadata.fov_ids``).

        Returns
        -------
        dict
            Experiment metadata keyed by ``"fov"`` plus empty per-file-type
            sub-dicts to be populated by the splitters.
        """
        this_exp_metadata = {"fov": fov_id}

        for file_key in (
            "timeseries",
            "depth_2p",
            "surface_2p",
            "local_z_stack",
        ):
            this_exp_metadata[file_key] = {}

        return this_exp_metadata

    def _get_imaging_planes(self) -> Tuple[str, ...]:
        """Return the FOV ids for this acquisition (one per plane).

        Returns
        -------
        tuple of str
            ``self.metadata.fov_ids``, ordered by the metadata's declared
            plane index. Enumerate to recover each plane's index.

        Raises
        ------
        ValueError
            If no imaging planes were parsed from the input metadata.
            Splitting with zero planes would silently produce no output,
            so fail loudly instead (fail-fast).
        """
        if not self.metadata.fov_ids:
            raise ValueError(
                f"No imaging planes found in {self.input_dir}; cannot "
                "split TIFFs. Check that the acquisition/session metadata "
                "defines imaging FOVs/planes."
            )
        return self.metadata.fov_ids

    def _roi_z_per_plane(
        self, timeseries_splitter: "TimeSeriesSplitter"
    ) -> List[Tuple[int, float]]:
        """Return the ``(roi_index, z_value)`` locating each imaging plane.

        Paired by position: ``fov_ids`` are sorted by the metadata's declared
        plane index, the TIFF manifest is ROI-major (each ROI's zs in
        actuator order). Assumes the two numberings agree. There is no key to
        join on -- metadata records depth in micrometres, the TIFF raw
        actuator coordinates.

        Parameters
        ----------
        timeseries_splitter : TimeSeriesSplitter
            Splitter for the timeseries TIFF, the canonical ROI/z manifest.

        Returns
        -------
        list of (int, float)
            One pair per plane, indexable by plane index. ``z_value`` is the
            float actuator coordinate, not the manifest's int form.

        Raises
        ------
        ValueError
            If the manifest and the metadata disagree on plane count. Pairing
            by position would then select the wrong plane for every FOV and
            write correct-looking filenames over wrong data, so fail loudly.
        """
        fov_ids = self._get_imaging_planes()
        manifest = timeseries_splitter.roi_z_int_manifest
        if len(manifest) != len(fov_ids):
            raise ValueError(
                f"Plane count mismatch: metadata in {self.input_dir} "
                f"declares {len(fov_ids)} plane(s) {list(fov_ids)}, but "
                f"{timeseries_splitter.input_path} contains "
                f"{len(manifest)}. Cannot pair planes to TIFF pages."
            )
        return [
            (roi_index, timeseries_splitter._z_from_int(z_int))
            for roi_index, z_int in manifest
        ]

    def _process_single_fov(
        self,
        fov_id: str,
        roi_index: int,
        z_value: float,
        timeseries_splitter: "TimeSeriesSplitter",
        depth_splitter: "AvgImageTiffSplitter",
        surface_splitter: "AvgImageTiffSplitter",
        zstack_splitter: "ZStackSplitter",
        valid_roi_centers: List[Tuple[float, float]],
        depth_remap: Dict[int, int],
        surface_remap: Dict[int, int],
        zstack_remap: Dict[int, int],
        t0: float,
    ) -> dict:
        """Process a single imaging plane and return its metadata.

        Uses the timeseries splitter as the canonical source of ROI
        indexing; per-splitter remap dicts translate from the
        timeseries-anchored valid-center index to each splitter's own
        roi_index so depth/surface/zstack TIFFs that index ROIs in a
        different order are still addressed correctly.
        """
        this_exp_metadata = self._create_experiment_metadata(fov_id)
        scanfield_z = z_value

        ts_center = timeseries_splitter.roi_center(i_roi=roi_index)
        baseline_center = get_nearest_roi_center(
            this_roi_center=ts_center,
            valid_roi_centers=valid_roi_centers,
        )

        splitter_roi_remaps = {
            id(depth_splitter): depth_remap,
            id(surface_splitter): surface_remap,
            id(zstack_splitter): zstack_remap,
        }

        for splitter, output_z, output_name, metadata_tag in zip(
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

            remap = splitter_roi_remaps[id(splitter)]
            splitter_roi_index = remap[baseline_center]

            if splitter_roi_index != roi_index:
                logging.info(
                    f"FOV {fov_id} {metadata_tag}: remapped roi_index "
                    f"{roi_index} -> {splitter_roi_index}"
                )

            splitter.write_output_file(
                i_roi=splitter_roi_index,
                z_value=output_z,
                output_path=output_path,
            )

            self._update_metadata_with_file_info(
                this_exp_metadata,
                metadata_tag,
                output_path,
                splitter,
                splitter_roi_index,
                output_z,
            )

            elapsed_time = time.time() - t0
            logging.info(
                f"wrote {output_path.resolve().absolute()} "
                f"after {elapsed_time:.2e} seconds"
            )

        return this_exp_metadata

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
        roi_z_per_plane = self._roi_z_per_plane(timeseries_splitter)

        for plane_index, fov_id in enumerate(self._get_imaging_planes()):
            roi_index, z_value = roi_z_per_plane[plane_index]
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
        full_field_path = find_one(
            self.input_dir,
            "*fullfield*.tiff",
            recursive=False,
            required=False,
        )

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

    def run_job(self) -> List[dict]:
        """Split TIFF files and return depth/surface position measurements."""
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

            # Get valid ROI centers from the timeseries (canonical anchor)
            valid_roi_centers = get_valid_roi_centers(
                timeseries_splitter=timeseries_splitter
            )

            # Build per-splitter roi_index remaps so each TIFF is addressed
            # by its own indexing (depth, surface, and zstack TIFFs can
            # index ROIs in a different order than the timeseries).
            depth_remap = build_roi_remap(depth_splitter, valid_roi_centers)
            surface_remap, position_pairs = build_surface_roi_remap(
                depth_splitter, surface_splitter, depth_remap
            )
            zstack_remap = build_roi_remap(
                zstack_splitter,
                valid_roi_centers,
                roi_indices=list(zstack_splitter._roi_to_path.keys()),
            )

            # Process experiment metadata for each imaging plane
            experiment_metadata = []
            roi_z_per_plane = self._roi_z_per_plane(timeseries_splitter)
            for plane_index, fov_id in enumerate(self._get_imaging_planes()):
                roi_index, z_value = roi_z_per_plane[plane_index]
                this_exp_metadata = self._process_single_fov(
                    fov_id,
                    roi_index,
                    z_value,
                    timeseries_splitter,
                    depth_splitter,
                    surface_splitter,
                    zstack_splitter,
                    valid_roi_centers,
                    depth_remap,
                    surface_remap,
                    zstack_remap,
                    t0,
                )
                experiment_metadata.append(this_exp_metadata)

            # Process timeseries data separately for efficiency
            output_path_lookup = self._process_timeseries_data(
                timeseries_splitter, experiment_metadata
            )

            # dump_every (buffer-flush cadence to temp files) uses default 1000
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
            return position_pairs
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
