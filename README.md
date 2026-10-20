# aind-pophys-converter

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-86%25-yellow?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Python library for converting raw population physiology imaging data to AIND
standard formats. Handles multi-plane mesoscope (ScanImage TIFF → split HDF5)
and single-plane Bergamo (per-epoch TIFFs → stitched HDF5) instruments, emitting
a v2 `processing.json` on both paths and a `quality_control.json` on the
mesoscope path. Structured logs go to CloudWatch.

## Installation

Editable, from source:

```bash
git clone https://github.com/AllenNeuralDynamics/aind-pophys-converter.git
cd aind-pophys-converter
pip install -e .
```

This package is not published to PyPI. Depend on it by **exact git commit**,
never a branch or tag:

```
aind-pophys-converter @ git+https://github.com/AllenNeuralDynamics/aind-pophys-converter.git@<sha>
```

A given commit pins `aind-pophys-metadata`, which in turn pins the schema
exactly.

## Usage

`job.run()` reads its configuration from CLI arguments or environment variables
via `JobSettings`:

| Parameter | Type | Required | Description |
|---|---|---|---|
| `--input_dir` | path | yes | Directory containing the core acquisition file (v1 `session.json` or v2 `acquisition.json`), a `pophys/` subdirectory, and an optional `data_description.json` |
| `--output_dir` | path | yes | Directory where HDF5 files, QC metrics, and images are written |
| `--temp_dir` | path | no | Scratch directory for intermediate files during mesoscope splitting |
| `--debug` | bool | no (default `False`) | Clip data volume — 100 frames/plane for mesoscope, 1 file/epoch for Bergamo |

All parameters can also be set as environment variables (e.g. `INPUT_DIR=/data/...`).

## Processing paths

The top-level `job.run()` reads metadata from the input directory, then dispatches to one of two
paths based on the instrument type.

### Bergamo (single-plane)

**Trigger:** `"Bergamo" in metadata.instrument_id`

The Bergamo microscope produces a directory of per-epoch TIFF files under
`pophys/`. `BergamoTiffStitcher.run_converter()`:

1. Derives the output filename from the first canonical FOV id
   (`metadata.fov_ids[0]`, e.g. `plane_0` for a single-plane session).
2. Resolves stimulus epochs to their TIFF files. Epoch membership and order
   come from the acquisition metadata (`metadata.epoch_stems`, ordered by
   `stimulus_start_time`); a `*.tif` glob under `pophys/`, grouped by stem
   and sorted numerically, supplies each epoch's files. Files whose stem matches no
   stimulus epoch names — (stacks) — are ignored, and a
   missing epoch or a missing epoch file raises.
3. Reads each TIFF with `ScanImageTiffReader`, caching frames incrementally
   into a single HDF5 file, in acquisition order.
4. Writes epoch location, trial location, epoch filename, and ScanImage header
   metadata as datasets in the same HDF5.

Output: `<output_dir>/<fov_id>/<fov_id>.h5`, `processing.json`

### Mesoscope (multi-plane)

**Trigger:** `instrument_id.startswith("MESO")` (mesoscope rigs are named
`MESO.*`).

The mesoscope produces multiplexed ScanImage TIFFs that interleave frames from
multiple ROIs and depths. `TiffSplitterCLI.run_job()`:

1. Reads metadata in the input_dir to determine FOVs, and locates each plane's
   TIFF pages by its `platform.json` ScanImage ROI and z
   (`scanimage_roi_index`, `scanimage_scanfield_z`)
2. For each plane, runs `AvgImageTiffSplitter` twice (depth and surface TIFFs)
   and `ZStackSplitter` (local z-stack, if `*local_z*.tiff` files are present)
   to write per-FOV depth, surface, and z-stack outputs.
3. Processes the timeseries splitter separately across all planes, writing one
   HDF5 per plane from the multiplexed `*timeseries.tiff`.
4. After splitting, `job.py` collects QC metrics:
   - `write_avg_depth_slices` + `pair_depth_tifs_with_avg_depth_pngs` — for
     child sessions only (gated by `is_child_session_via_platform_json`):
     writes per-z averaged-depth TIFs then pairs them with parent depth TIFs
     read from `platform.json`, producing merged PNGs and a list of metrics.

   `qc.finalize_quality_control` then appends the vasculature metric (from
   `*_vasculature.tif`, when present) and writes the collected metrics as one
   `quality_control.json`.

Output: per-plane HDF5s, `processing.json` and `quality_control.json`.

If pre-split directories are already present in the input, `run_job()` skips
splitting and writes only stub `.txt` pointer files.

## Modules

- **`job.py`** — the entry point. `run()` loads settings and metadata, then
  dispatches to the Bergamo or mesoscope path inside a `stage_guard` that always
  emits a `processing.json`. Structured JSON logs come from the pinned
  `log-schema` dependency (CloudWatch, degrading to console when the network is
  blocked).

- **`settings.py`** — `JobSettings`, the pydantic-settings model parsing the CLI
  args and environment variables. Also what the Code Ocean app panel is
  generated from.

- **`bergamo_stitcher.py`** — the single-plane path. `BergamoTiffStitcher`
  resolves the metadata's stimulus epochs to their TIFFs and streams them into
  one HDF5 in acquisition order; `BaseStitcher` holds the shared HDF5 writers.

- **`mesoscope_splitter.py`** — the multi-plane path. `TiffSplitterCLI` iterates
  the imaging planes and dispatches to the splitters in `tiff_splitters`, plus
  optional full-field stitching. `find_split_directories` lets an already-split
  input skip re-splitting.

- **`tiff_splitters.py`** — the ScanImage TIFF splitting classes
  (`TiffSplitterBase` and its `AvgImageTiffSplitter` / `TimeSeriesSplitter` /
  `ZStackSplitter` subclasses) that parse ROI/z layout and write the per-plane
  outputs.

- **`qc.py`** — the `quality_control.json` path (mesoscope only). Metric
  producers return metrics; `finalize_quality_control` writes one document for
  the run, never per-metric fragments.

- **`utils/metadata_utils.py`** — the converter's only point of contact with
  schema-version differences; everything else reads the schema-agnostic
  `ConverterMetadata` (`instrument_id`, `fov_ids`, `dataset_name`, `subject_id`,
  `epoch_stems`). The reads themselves live in the shared
  [`aind-pophys-metadata`](https://github.com/AllenNeuralDynamics/aind-pophys-metadata)
  package. `instrument_id` drives Bergamo-versus-mesoscope routing; the rest are
  labels or may be empty for malformed input.

- **`utils/tiff_metadata.py`** — `ScanImageMetadata`, a lazy wrapper over
  ScanImage headers used by the splitters to read frame layout without loading
  pixels.

- **`utils/full_field_utils.py`** — stitches the full-field surface image into
  `stitched_full_field_img.h5` when a `*fullfield*.tiff` is present.

- **`utils/image_utils.py`** — small PIL helpers (`add_border_and_label`) for QC
  thumbnails.

- **`utils/platform_utils.py`** — `platform.json` readers: each plane's
  ScanImage position for the splitter, and the child-session gate for the
  depth QC.

## Development

The CI gate is `flake8 . && interrogate --verbose . && coverage run -m unittest
discover && coverage report` — **unittest, not pytest**. Reproduce it locally
before pushing:

```bash
flake8 . && interrogate --verbose . && coverage run -m unittest discover && coverage report
```

Coverage and interrogate are both gated at 100%; black and flake8 are set to 79
columns. Coverage counts `tests` as well as the package, so a non-test helper
dropped into `tests/` is measured and will fail the gate — keep tooling out of
that directory.

## Level of Support

- [x] Supported: We are releasing this code to the public as a tool we expect
  others to use. Issues are welcomed, and we expect to address them promptly;
  pull requests will be vetted by our staff before inclusion.

## License

MIT — see [LICENSE](LICENSE).
