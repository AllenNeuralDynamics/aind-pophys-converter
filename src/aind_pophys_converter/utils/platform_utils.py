"""Platform JSON utilities (temporary — will be removed when platform.json
dependency is retired)."""
import json
from pathlib import Path


def is_child_session_via_platform_json(platform_fp: Path) -> bool:
    """Return True if platform.json contains a parent_session field.

    Parameters
    ----------
    platform_fp : Path
        Path to the platform.json file.

    Returns
    -------
    bool
        True if a parent_session key is present and non-null.
    """
    print("checking is_child_session_via_platform_json", platform_fp)
    with open(platform_fp) as f:
        platform_json = json.load(f)
    parent_session = platform_json.get("parent_session", None)
    print("returning", parent_session is not None)
    return parent_session is not None
