import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import _apply_split, ensure_schema, question_to_dict, _resolve_upload, _stored_name
from models import Question


def test_question_to_dict_reports_ready_to_use_state():
    fields = dict(
        id=1,
        subject="数学",
        image_path="abc_q1.jpg",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert question_to_dict(Question(**fields, crop_reviewed=True))["crop_reviewed"] is True
    assert question_to_dict(Question(**fields, crop_reviewed=False))["crop_reviewed"] is False


def test_ensure_schema_marks_legacy_rows_as_ready():
    """旧库没有 crop_reviewed 语义了：启动后所有历史错题都应该能直接进练习卷。"""
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
            # 两条都是历史遗留的错题，现在都不该再被挡在练习卷外面
            assert [row[1] for row in rows] == [1, 1]
        finally:
            engine.dispose()  # Windows 下连接池不释放会锁住临时库文件


def test_ensure_schema_migrates_absolute_paths_to_filenames():
    """旧库存绝对路径，项目一搬家全断；启动时统一收敛成文件名（幂等）。"""
    with TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{Path(directory) / 'abs.db'}")
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
                    "INSERT INTO questions (subject, image_path, source_image_path)"
                    " VALUES ('数学',"
                    " 'C:\\Users\\huntl\\retake\\uploads\\abc_reviewed_1_q1.jpg',"
                    " 'C:\\Users\\huntl\\retake\\uploads\\abc_original.jpg')"
                )
            ensure_schema(engine)
            with engine.begin() as conn:
                rows = conn.exec_driver_sql(
                    "SELECT image_path, source_image_path FROM questions"
                ).fetchall()
            assert rows[0] == ("abc_reviewed_1_q1.jpg", "abc_original.jpg")

            # 再跑一次不变（幂等）
            ensure_schema(engine)
            with engine.begin() as conn:
                rows = conn.exec_driver_sql(
                    "SELECT image_path, source_image_path FROM questions"
                ).fetchall()
            assert rows[0] == ("abc_reviewed_1_q1.jpg", "abc_original.jpg")
        finally:
            engine.dispose()


def test_stored_name_and_resolve_upload(tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path)
    assert _stored_name("/any/where/uploads/abc_q1.jpg") == "abc_q1.jpg"
    assert _stored_name(Path("abc_q1.jpg")) == "abc_q1.jpg"
    # 新数据：文件名 → 当前上传目录（项目搬家后依然找得到）
    assert _resolve_upload("abc_q1.jpg") == tmp_path / "abc_q1.jpg"
    # 旧数据：绝对路径且文件还在时照常使用
    legacy = tmp_path / "legacy.jpg"
    legacy.write_bytes(b"x")
    assert _resolve_upload(str(legacy)) == legacy


def _fresh_session(directory: Path):
    """建一个独立的临时 SQLite 引擎并跑一次 ensure_schema，返回 Session。"""
    engine = create_engine(f"sqlite:///{directory / 'test.db'}")
    ensure_schema(engine)
    return engine, sessionmaker(bind=engine)


def test_apply_split_grows_group_and_keeps_metadata_by_position():
    """切出更多题时旧图清掉，并按序号保住知识点和遮挡（用户不用重录标注）。"""
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        engine, SessionLocal = _fresh_session(directory)
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
                knowledge_point="约分",
                answer_masks="[[100, 100, 200, 200]]",
            )
            q2 = Question(
                subject="数学",
                image_path=str(old2),
                source_image_path=str(source),
                knowledge_point="通分",
            )
            db.add_all([q1, q2])
            db.commit()
            db.refresh(q1)

            new_crops = [directory / f"abc_new_q{i}.jpg" for i in range(1, 4)]
            for path in new_crops:
                path.write_bytes(b"new")

            questions = _apply_split(db, q1, str(source), new_crops)

            assert len(questions) == 3
            # 前两题按序号保留标注，第三题是新增的
            assert questions[0].knowledge_point == "约分"
            assert questions[0].answer_masks == "[[100, 100, 200, 200]]"
            assert questions[1].knowledge_point == "通分"
            assert questions[2].knowledge_point in ("", None)
            assert all(q.crop_reviewed for q in questions)

            # 旧切片文件被清掉，新切片留下；原卷仍在
            assert not old1.exists() and not old2.exists()
            for path in new_crops:
                assert path.exists()
            assert source.exists()
            db.close()
        finally:
            engine.dispose()


def test_apply_split_shrinks_group_and_deletes_orphan_files():
    """切得更少时：多余的题被删、它们的图片也被清。"""
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        engine, SessionLocal = _fresh_session(directory)
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
                )
                for i in range(3)
            ]
            db.add_all(qs)
            db.commit()
            db.refresh(qs[0])

            new_crop = directory / "abc_new_q1.jpg"
            new_crop.write_bytes(b"new")
            questions = _apply_split(db, qs[0], str(source), [new_crop])

            assert len(questions) == 1
            assert questions[0].crop_reviewed is True

            # 3 个旧文件全清（即使是被删的题），新文件保留
            for path in old:
                assert not path.exists()
            assert new_crop.exists()
            db.close()
        finally:
            engine.dispose()


def test_backup_bundles_database_and_uploads(tmp_path):
    """备份 zip 里应有可还原的 questions.db 和 uploads/，且不夹带项目里其他文件。"""
    import backup

    project = tmp_path / "project"
    uploads = project / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "abc_q1.jpg").write_bytes(b"image")
    (uploads / "abc_original.jpg").write_bytes(b"source")
    (project / "notes.txt").write_bytes(b"scratch")

    db_path = project / "questions.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        ensure_schema(engine)
        db = sessionmaker(bind=engine)()
        db.add(Question(subject="数学", image_path="abc_q1.jpg"))
        db.commit()
        db.close()
    finally:
        engine.dispose()

    original_db, original_uploads = backup.DB_PATH, backup.UPLOAD_DIR
    backup.DB_PATH, backup.UPLOAD_DIR = db_path, uploads
    try:
        target = backup.create_backup(tmp_path / "backups")
    finally:
        backup.DB_PATH, backup.UPLOAD_DIR = original_db, original_uploads

    assert target.is_file()
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {
            "questions.db",
            "uploads/abc_q1.jpg",
            "uploads/abc_original.jpg",
        }
        # 快照是一致可读的库，不是复制一半的文件
        assert archive.read("uploads/abc_q1.jpg") == b"image"