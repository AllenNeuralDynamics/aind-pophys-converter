"""Tests for JobSettings."""

import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from unittest.mock import patch

if TYPE_CHECKING:
    from aind_pophys_converter.settings import JobSettings


class TestJobSettings(unittest.TestCase):
    """Tests for JobSettings."""

    def _make_settings(
        self, extra_argv: Optional[List[str]] = None
    ) -> "JobSettings":
        """Instantiate JobSettings with a controlled sys.argv."""
        from aind_pophys_converter.settings import JobSettings

        argv = ["prog", "--input_dir", "/in", "--output_dir", "/out"]
        if extra_argv:
            argv.extend(extra_argv)
        with patch.object(sys, "argv", argv):
            return JobSettings()

    def test_required_fields(self) -> None:
        """input_dir and output_dir are set from argv."""
        s = self._make_settings()
        self.assertEqual(str(s.input_dir), "/in")
        self.assertEqual(str(s.output_dir), "/out")

    def test_default_temp_dir_is_none(self) -> None:
        """temp_dir defaults to None."""
        s = self._make_settings()
        self.assertIsNone(s.temp_dir)

    def test_default_debug_is_false(self) -> None:
        """debug defaults to False."""
        s = self._make_settings()
        self.assertFalse(s.debug)

    def test_debug_flag(self) -> None:
        """debug is set from argv."""
        s = self._make_settings(["--debug", "True"])
        self.assertTrue(s.debug)

    def test_temp_dir_set(self) -> None:
        """temp_dir is converted to Path when provided."""
        s = self._make_settings(["--temp_dir", "/tmp/mytemp"])
        self.assertEqual(s.temp_dir, Path("/tmp/mytemp"))

    def test_input_dir_mutation(self) -> None:
        """input_dir can be reassigned (used in run())."""
        s = self._make_settings()
        new_path = Path("/scratch/data/pophys")
        s.input_dir = new_path
        self.assertEqual(s.input_dir, new_path)


if __name__ == "__main__":
    unittest.main()
