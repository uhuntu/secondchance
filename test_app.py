import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine

from app import (
    _delete_previous_previews,
    _sweep_abandoned_previews,
    ensure_schema,
    question_to_dict,
)
from models import Question


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


def test_question_to_dict_reports_crop_reviewed_column():
    fields = dict(
        id=1,
        subject="数学",
        image_path="abc_q1.jpg",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert question_to_dict(Question(**fields, crop_reviewed=True))["crop_reviewed"] is True
    assert question_to_dict(Question(**fields, crop_reviewed=False))["crop_reviewed"] is False


def test_ensure_schema_backfills_crop_reviewed_from_filenames():
    with TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{Path(directory) / 'old.db'}")
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    "CREATE TABLE questions ("
                    "id INTEGER PRIMARY KEY, subject VARCHAR(50) NOT NULL,"
                    "knowledge_point VARCHAR(200), note TEXT,"
                    "image_path VARCHAR(300) NOT NULL, source_image_path VARCHAR(300),"
                    "answer_masks TEXT, created_at DATETIME)"
                )
                conn.exec_driver_sql(
                    "INSERT INTO questions (subject, image_path) VALUES"
                    " ('数学', 'abc_original_reviewed_1234567890_q1.jpg'),"
                    " ('数学', 'abc_original_q1.jpg')"
                )
            ensure_schema(engine)
            with engine.begin() as conn:
                rows = conn.exec_driver_sql(
                    "SELECT image_path, crop_reviewed FROM questions ORDER BY id"
                ).fetchall()
            assert rows[0][1] == 1  # 旧库用文件名里的 _reviewed_ 标记确认状态，需回填
            assert rows[1][1] == 0
        finally:
            engine.dispose()  # Windows 下连接池不释放会锁住临时库文件
