# Converter fixtures

The small fixtures under `synthetic/` exercise schema dispatch and the minimal
single-plane and multi-plane contracts. They are hand-authored or reduced from
real session metadata, with every identifying value replaced.

The remaining fixtures are anonymised session metadata, organized by asset and
schema version:

```text
tests/fixtures/
├── bergamo/v1
├── bergamo/v2
├── meso_parent/v1
├── meso_parent/v2
├── meso_child/v1
└── meso_child/v2
```

| fixture | origin |
|---|---|
| `bergamo` | anonymised from a Bergamo single-plane session |
| `meso_parent` | anonymised from a mesoscope root session |
| `meso_child` | anonymised from a mesoscope child session |

The `v1` directories hold the session-layout metadata. The `v2` directories
contain deterministic `acquisition.json` and `data_description.json` snapshots
generated through `aind-metadata-manager`'s legacy-metadata upgrade path. They
are converter-input snapshots, not a complete copy of the upgrader output.

Subject ids, personnel names, ethics protocol numbers, animal and housing
records, device serial numbers, repository links, project names and session
names are synthetic. Every session timestamp is moved by one constant per
session, so epoch order and durations are unchanged. Acquisition parameters and
plane geometry are as acquired. The fixtures hold metadata only; no image data
is stored here.

`meso_parent` is a **root** session: its platform.json has no
`parent_session`, so it exercises the non-child converter branch (depth pairing
skipped). `meso_child` is a child whose `parent_session` points at a third
session (`...2025-03-01...`), so it exercises the child depth-pairing branch.

`tests/test_metadata_utils.py` loads both schema versions for each anonymised
asset and verifies that they produce equal `ConverterMetadata`. It also checks
the Meso platform/session relationship (root vs child) and the Bergamo epoch
stems. Full upgrader output is not snapshotted because the generated Bergamo
instrument metadata contains duplicate component names whose numeric suffixes
are not stable between runs.
