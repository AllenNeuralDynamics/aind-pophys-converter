# Converter fixtures

The small hand-authored fixtures live under `synthetic/`. They exercise schema
dispatch and the minimal single-plane and multi-plane contracts without
vendoring production data.

The production metadata is organized by asset and schema version:

```text
tests/fixtures/
├── bergamo/v1
├── bergamo/v2
├── meso_parent/v1
├── meso_parent/v2
├── meso_child/v1
└── meso_child/v2
```

The `v1` directories preserve the supplied raw metadata. The `v2` directories
contain deterministic `acquisition.json` and `data_description.json` snapshots
generated through `aind-metadata-manager`'s legacy-metadata upgrade path. They
are converter-input snapshots, not a complete copy of the upgrader output.

| fixture | Code Ocean asset | source prefix |
|---|---|---|
| `bergamo` | `85e168ee-b960-4721-8c1d-7845b0a6dc19` | `single-plane-ophys_772414_2025-04-11_17-41-16` |
| `meso_parent` | `af4527a8-0a52-4d15-bfa7-9f4334da4e49` | `multiplane-ophys_849375_2026-06-17_15-28-08` |
| `meso_child` | `ff2fa171-9905-40d8-b058-cbf177e8cbb0` | `multiplane-ophys_849375_2026-06-20_13-13-04` |

The Bergamo v1 metadata was recovered from DocDB record
`e8e0098a-b1cb-400b-8835-a04d9e5d9802`. Its raw TIFFs remain in the project-level
`scratch/` directory, not this repository. The Meso TIFF samples are likewise
kept in the project's `scratch/real_asset_tifs/` directory.

`tests/test_metadata_utils.py` loads both schema versions for each production
asset and verifies that they produce equal `ConverterMetadata`. It also checks
the Meso platform/session relationship and the Bergamo epoch stems. Full
upgrader output is not snapshotted because the generated Bergamo instrument
metadata contains duplicate component names whose numeric suffixes are not
stable between runs.
