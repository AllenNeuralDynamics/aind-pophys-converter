# aind-pophys-converter

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-86%25-yellow?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Python library for converting raw population physiology imaging data to AIND
standard formats. Handles multi-plane mesoscope (ScanImage TIFF → split HDF5)
and single-plane Bergamo (per-epoch TIFFs → stitched HDF5) instruments,
generates QC metrics, and emits structured logs to CloudWatch.

## Installation

```bash
pip install aind-pophys-converter
```

Or in editable mode from source:

```bash
git clone https://github.com/AllenNeuralDynamics/aind-pophys-converter.git
cd aind-pophys-converter
pip install -e .
```

## Usage

`job.run()` reads its configuration from CLI arguments or environment variables
via `JobSettings`:

| Parameter | Type | Required | Description |
|---|---|---|---|
| `--input_dir` | path | yes | Directory containing the core acquisition file (v1 `session.json` or v2 `acquisition.json`), `data_description.json`, and a `pophys/` subdirectory |
| `--output_dir` | path | yes | Directory where HDF5 files, QC metrics, and images are written |
| `--temp_dir` | path | no | Scratch directory for intermediate files during mesoscope splitting |
| `--debug` | bool | no (default `False`) | Clip data volume — 100 frames/plane for mesoscope, 1 file/epoch for Bergamo |

All parameters can also be set as environment variables (e.g. `INPUT_DIR=/data/...`).

## Processing paths

The top-level `job.run()` reads the core acquisition file (v1 `session.json` or
v2 `acquisition.json`) and `data_description.json` from the input directory,
then dispatches to one of two paths based on the
instrument type.

### Bergamo (single-plane)

**Trigger:** `"Bergamo" in metadata.instrument_id`

The Bergamo microscope produces a directory of per-epoch TIFF files under
`pophys/`. `BergamoTiffStitcher.run_converter()`:

1. Derives the output filename from the first canonical FOV id
   (`metadata.fov_ids[0]`, e.g. `MO_0`).
2. Groups `*.tif` files in `pophys/` by epoch stem, sorted numerically,
   excluding z-stack calibration files.
3. Reads each TIFF with `ScanImageTiffReader`, caching frames incrementally
   into a single HDF5 file.
4. Writes epoch location, trial location, epoch filename, and ScanImage header
   metadata as datasets in the same HDF5.

Output: `<output_dir>/<fov_id>/<fov_id>.h5`

### Mesoscope (multi-plane)

**Trigger:** `instrument_id.startswith("MESO")` (mesoscope rigs are named
`MESO.*`).

The mesoscope produces multiplexed ScanImage TIFFs that interleave frames from
multiple ROIs and depths. `TiffSplitterCLI.run_job()`:

1. Reads the core acquisition file to discover the set of imaging planes — v2
   `ImagingConfig → PlanarImage → CoupledPlane`, or v1
   `data_streams → ophys_fovs` — ordered by plane index.
2. For each plane, runs `AvgImageTiffSplitter` twice (depth and surface TIFFs)
   and `ZStackSplitter` (local z-stack, if `*local_z*.tiff` files are present)
   to write per-FOV depth, surface, and z-stack outputs.
3. Processes the timeseries splitter separately across all planes, writing one
   HDF5 per plane from the multiplexed `*timeseries.tiff`.
4. After splitting, `job.py` runs the QC pipeline:
   - `create_vasculature` — converts the pre-existing `*_vasculature.tif` to
     PNG and returns a `QCMetric`. Always runs.
   - `write_avg_depth_slices` + `pair_depth_tifs_with_avg_depth_pngs` — for
     child sessions only (gated by `is_child_session_via_platform_json`):
     writes per-z averaged-depth TIFs then pairs them with parent depth TIFs
     read from `platform.json`, producing merged PNGs and a list of metrics.

   The collected metrics are written as one `quality_control.json`.

Output: per-plane HDF5s, `processing.json` and `quality_control.json`.

If pre-split directories are already present in the input, `run_job()` skips
splitting and writes only stub `.txt` pointer files.

## Module reference

### `job.py`

Top-level orchestrator. `run()` is the library entry point called by the
capsule. Loads settings, reads metadata, calls `setup_logging`, then dispatches
to the Bergamo or mesoscope path. Emits `stage_start`, `stage_complete`, and
`stage_error` log events.

### `settings.py`

`JobSettings` — a Pydantic-settings model that parses CLI arguments and
environment variables. Fields: `input_dir`, `output_dir`, `temp_dir` (optional),
`debug` (clips data volume when `True`; 100 frames/plane for mesoscope, 1
file/epoch for Bergamo).

### `bergamo_stitcher.py`

`BergamoSettings` — Pydantic model holding `input_dir`, `output_dir` and
`debug`.

`BaseStitcher` — shared HDF5 write helpers (`write_images`, `write_final_output`).

`BergamoTiffStitcher(BaseStitcher)` — concrete stitcher. Key methods:
- `_discover_tiff_epochs()` — groups and sorts input TIFFs by epoch stem.
- `write_bergamo()` — streams frames into HDF5 with epoch/trial metadata.
- `run_converter()` — takes `unique_id` from `metadata.fov_ids[0]`, sets up the
  output subdirectory, and orchestrates the full conversion.
- `from_args()` — constructs a stitcher from CLI arguments for standalone use.

### `mesoscope_splitter.py`

`TiffSplitterCLI` — orchestrates the full mesoscope split. Takes the loaded
`ConverterMetadata`, iterates over its `fov_ids` via `_get_imaging_planes()`,
and dispatches to `TimeSeriesSplitter`, `AvgImageTiffSplitter`, and
`ZStackSplitter`. Also handles optional full-field surface image stitching via
`full_field_utils` when a `*fullfield*.tiff` is present.

`find_split_directories()` — detects whether a previous split run already
produced output directories, allowing `job.py` to skip re-splitting.

### `tiff_splitters.py`

Core ScanImage TIFF splitting classes. Class hierarchy:

```
IntFromZMapperMixin          ← maps z-values ↔ integer indices
├── TiffSplitterBase         ← parses ScanImage metadata; validates ROI/z config
│   ├── AvgImageTiffSplitter ← writes per-ROI depth-averaged TIFF
│   └── TimeSeriesSplitter   ← splits multiplexed timeseries → per-plane HDF5
└── ZStackSplitter           ← extracts z-stack calibration slices → HDF5
```

`mkstemp_clean()` — thin wrapper around `tempfile.mkstemp` that closes the file
descriptor immediately, used throughout to avoid descriptor leaks.

`DEBUG_N_FRAMES = 100` — frame count used when `debug=True`.

### `qc.py`

QC visual generation and metric serialization. Functions:

- `write_avg_depth_slices` — saves averaged-depth slice images from an
  `AvgImageTiffSplitter` to the output directory.
- `pair_depth_tifs_with_avg_depth_pngs` — matches each split depth TIFF to an
  averaged-depth PNG and returns one `QCMetric` per matched pair.
- `create_vasculature` — converts the pre-existing `*_vasculature.tif` in
  `pophys/` to PNG and returns its `QCMetric`, or `None` when absent.
- `pending_qc_status` / `build_fov_qc_metric` / `build_vasculature_qc_metric` —
  build the individual `QCMetric` objects; they return metrics rather than
  writing them.
- `build_quality_control` / `write_quality_control_json` — collect the metrics
  into one `QualityControl` document and serialise it.

The collector pattern is deliberate: metric-producing functions return metrics
and the entrypoint writes a **single** `quality_control.json` for the whole run,
rather than one file per metric.

### `utils/metadata_utils.py`

The converter's only point of contact with schema-version differences. The
reads themselves live in the shared
[`aind-pophys-metadata`](https://github.com/AllenNeuralDynamics/aind-pophys-metadata)
package, so nothing else in this repo branches on v1 versus v2.

| Object | Description |
|---|---|
| `ConverterMetadata` | Frozen, schema-agnostic view holding only what the conversion uses: `instrument_id`, `fov_ids`, `dataset_name`, `subject_id` |
| `load_converter_metadata(input_dir)` | Discovers the core file, dispatches v1/v2, and populates `ConverterMetadata` via the shared getters |
| `build_converter_code(job_parameters, input_data=)` | Builds the v2 `Code` block; `stage_guard` assembles the surrounding `DataProcess` and `Processing` |

`instrument_id` is the only required field — it drives Bergamo versus mesoscope
routing. `dataset_name` and `subject_id` are logging labels, and `fov_ids` may
be empty for malformed input, which consumers fail fast on.

### `utils/tiff_metadata.py`

`ScanImageMetadata` — lazy-parsed wrapper around ScanImage TIFF headers.
Exposes properties: `numVolumes`, `numSlices`, `channelSave`, `wavelengths`,
and pixel scale values. Used by all splitter classes to determine frame layout
without loading pixel data.

### `utils/full_field_utils.py`

Utilities for stitching full-field surface images and overlaying ROI boundaries.
Used by `TiffSplitterCLI._process_full_field_image()` when a `*fullfield*.tiff`
is present; produces `stitched_full_field_img.h5` in the output directory.

### `utils/image_utils.py`

Generic PIL/image helpers. Currently provides `add_border_and_label`, which
annotates depth-slice PNG thumbnails with a coloured border and text label
before they are embedded in QC outputs.

### Logging

Structured JSON logging comes from the pinned `log-schema` dependency, not from
this package. `job.py` and `bergamo_stitcher.py` call
`log_schema.main.setup_logging(use_cloudwatch=True, model={...})` with
`acquisition_name` and `process_name`, so every record carries that context
without it being passed at each call site. The CloudWatch handler degrades to
console with a warning when it cannot initialise, so a blocked network does not
fail a run.

### `utils/platform_utils.py`

Temporary helpers for reading `platform.json`. Currently exposes
`is_child_session_via_platform_json(path)`, which returns `True` when a
`parent_session` key is present. Used in `job.py` to gate the depth-TIFF
pairing QC step, and in `qc.pair_depth_tifs_with_avg_depth_pngs` to read
`imaging_plane_groups` for parent-depth matching. Will be removed once the
depth-pairing step is migrated to `acquisition.json`.

## Development

```bash
# Run tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=aind_pophys_converter --cov-fail-under=85

# Lint / format
black src/ tests/ --line-length 79
isort src/ tests/
flake8 src/ tests/
interrogate src/
```

## License

MIT — see [LICENSE](LICENSE).
