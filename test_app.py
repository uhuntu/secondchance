import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from app import _delete_previous_previews, _sweep_abandoned_previews


def _write_aged_file(path: Path, age_hours: float) -> None:
    path.write_bytes(b"preview")
    timestamp = time.time() - age_hours * 3600
    os.utime(path, (timestamp, timestamp))


def test_stale_previews_swept_but_recent_ones_kept():
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        fresh = directory / "abc_recorrect_preview_1111111111_page.jpg"
        stale = directory / "abc_recorrect_preview_2222222222_page.jpg"
        _write_aged_file(fresh, 1)
        _write_aged_file(stale, 30)
        _sweep_abandoned_previews(directory)
        assert fresh.is_file()
        assert not stale.exists()


def test_sweep_ignores_non_preview_uploads():
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        original = directory / "abc_original.jpg"
        reviewed = directory / "abc_reviewed_9999999999_q1.jpg"
        _write_aged_file(original, 100)
        _write_aged_file(reviewed, 100)
        _sweep_abandoned_previews(directory)
        assert original.is_file()
        assert reviewed.is_file()


def test_new_preview_removes_previous_one_for_same_page():
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        previous = directory / "abc_recorrect_preview_1111111111_page.jpg"
        previous.write_bytes(b"preview")
        _delete_previous_previews(directory, "abc")
        assert not previous.exists()
        other_page = directory / "def_recorrect_preview_3333333333_page.jpg"
        other_page.write_bytes(b"preview")
        _delete_previous_previews(directory, "abc")
        assert other_page.is_file()
