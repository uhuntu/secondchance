import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import (
    _delete_previous_previews,
    _replace_question_crops,
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


def _fresh_engine_with_schema(directory: Path):
    """建一个独立的临时 SQLite 引擎并跑一次 ensure_schema，返回 (engine, SessionLocal)。"""
    engine = create_engine(f"sqlite:///{directory / 'test.db'}")
    ensure_schema(engine)
    return engine, sessionmaker(bind=engine)


def test_replace_question_crops_grows_group_and_resets_metadata():
    """重新切题后题组变多时：旧 crop 文件被清，metadata 全部重置，新题按 confirmed 标记。"""
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        engine, SessionLocal = _fresh_engine_with_schema(directory)
        try:
            db = SessionLocal()
            source = directory / "abc_original.jpg"
            source.write_bytes(b"source")
            old1 = directory / "abc_old_q1.jpg"
            old2 = directory / "abc_old_q2.jpg"
            old1.write_bytes(b"old1")
            old2.write_bytes(b"old2")

            q1 = Question(
                subject="数学",
                image_path=str(old1),
                source_image_path=str(source),
                crop_reviewed=True,
                knowledge_point="约分",
                answer_masks="[[100, 100, 200, 200]]",
            )
            q2 = Question(
                subject="数学",
                image_path=str(old2),
                source_image_path=str(source),
                crop_reviewed=True,
                knowledge_point="通分",
            )
            db.add_all([q1, q2])
            db.commit()
            db.refresh(q1)

            # 这次重切变成 3 题（多识别出一个红叉）
            new_crops = [directory / f"abc_new_q{i}.jpg" for i in range(1, 4)]
            for path in new_crops:
                path.write_bytes(b"new")

            response = _replace_question_crops(
                q1, db, str(source), new_crops, confirmed=False
            )

            # 响应里报告的计数：2 条更新 + 1 条新增
            assert response["updated_count"] == 2
            assert response["created_count"] == 1
            assert response["removed_count"] == 0
            assert len(response["questions"]) == 3

            group = (
                db.query(Question)
                .filter(Question.source_image_path == str(source))
                .order_by(Question.id.asc())
                .all()
            )
            assert len(group) == 3
            # 全部重置：未确认 + 知识点/遮挡清空（图像变了，旧标注不再有意义）
            for q in group:
                assert q.crop_reviewed is False
                assert q.knowledge_point == ""
                assert q.answer_masks == ""
                assert q.subject == "数学"

            assert not old1.exists()
            assert not old2.exists()
            for path in new_crops:
                assert path.exists()
            # 原卷仍在：组里还有题引用它
            assert source.exists()
            db.close()
        finally:
            engine.dispose()


def test_replace_question_crops_shrinks_group_and_deletes_orphan_files():
    """题组变少时：多余的题被删、它们的图片也被清。"""
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        engine, SessionLocal = _fresh_engine_with_schema(directory)
        try:
            db = SessionLocal()
            source = directory / "abc_original.jpg"
            source.write_bytes(b"source")
            old = [directory / f"abc_old_q{i}.jpg" for i in range(1, 4)]
            for path in old:
                path.write_bytes(b"old")

            qs = [
                Question(
                    subject="数学",
                    image_path=str(old[i]),
                    source_image_path=str(source),
                    crop_reviewed=False,
                )
                for i in range(3)
            ]
            db.add_all(qs)
            db.commit()
            db.refresh(qs[0])

            # 这次重切只保留 1 题
            new_crop = directory / "abc_new_q1.jpg"
            new_crop.write_bytes(b"new")

            response = _replace_question_crops(
                qs[0], db, str(source), [new_crop], confirmed=True
            )

            assert response["updated_count"] == 1
            assert response["created_count"] == 0
            assert response["removed_count"] == 2

            remaining = (
                db.query(Question)
                .filter(Question.source_image_path == str(source))
                .all()
            )
            assert len(remaining) == 1
            assert remaining[0].crop_reviewed is True  # confirmed=True 透传

            # 3 个旧文件全清（即使被删的题），新文件保留
            for path in old:
                assert not path.exists()
            assert new_crop.exists()
            db.close()
        finally:
            engine.dispose()
