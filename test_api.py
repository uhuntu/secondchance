"""端到端 API 测试：基于 TestClient + 临时数据库/上传目录，不依赖真实服务进程。

运行：python -m pytest test_api.py
"""
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app as app_module
from app import app, ensure_schema
from database import get_db

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _synthetic_page_jpeg() -> bytes:
    """一张带边框和三个红叉的作业页，用来验证自动切题。"""
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (45, 35), (755, 1165), (20, 20, 20), 4)
    for y in (250, 650, 980):
        cv2.line(page, (600, y - 25), (650, y + 25), (0, 0, 220), 10)
        cv2.line(page, (650, y - 25), (600, y + 25), (0, 0, 220), 10)
    ok, buffer = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def api():
    """隔离环境：临时 SQLite 库 + 临时上传目录，覆盖 get_db 依赖和 UPLOAD_DIR。"""
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        engine = create_engine(f"sqlite:///{directory / 'test.db'}")
        ensure_schema(engine)
        TestSession = sessionmaker(bind=engine)
        uploads = directory / "uploads"
        uploads.mkdir()

        def override_get_db():
            db = TestSession()
            try:
                yield db
            finally:
                db.close()

        original_uploads = app_module.UPLOAD_DIR
        app.dependency_overrides[get_db] = override_get_db
        app_module.UPLOAD_DIR = uploads
        try:
            with TestClient(app) as client:
                yield SimpleNamespace(
                    client=client, uploads=uploads, Session=TestSession
                )
        finally:
            app.dependency_overrides.clear()
            app_module.UPLOAD_DIR = original_uploads
            engine.dispose()  # Windows 下连接池不释放会锁住临时库文件


def _create_question(api, name="wrong.png", content=PNG, subject="数学"):
    return api.client.post(
        "/api/questions",
        data={"subject": subject, "knowledge_point": "分数计算", "note": "正确答案 3/4"},
        files={"photo": (name, content, "image/png")},
    )


def test_full_lifecycle(api):
    """增 → 查/过滤/搜索 → 详情+原图 → 改 → 删（记录和照片都清掉）。"""
    created = _create_question(api)
    assert created.status_code == 201, created.text
    question = created.json()
    assert question["subject"] == "数学" and question["knowledge_point"] == "分数计算"
    # 单图直传视为已确认，可直接进练习卷
    assert question["crop_reviewed"] is True

    # 列表 + 科目过滤 + 关键词搜索
    listed = api.client.get("/api/questions").json()
    assert any(q["id"] == question["id"] for q in listed)
    assert all(q["subject"] == "数学" for q in api.client.get("/api/questions", params={"subject": "数学"}).json())
    assert any(q["id"] == question["id"] for q in api.client.get("/api/questions", params={"keyword": "分数"}).json())
    assert api.client.get("/api/questions", params={"keyword": "不存在的词"}).json() == []

    # 详情 + 照片原图访问
    detail = api.client.get(f"/api/questions/{question['id']}").json()
    assert detail["note"] == "正确答案 3/4"
    image = api.client.get(detail["image_url"])
    assert image.status_code == 200 and image.content == PNG

    # 编辑
    updated = api.client.put(
        f"/api/questions/{question['id']}",
        json={"knowledge_point": "约分", "note": "已订正"},
    ).json()
    assert updated["knowledge_point"] == "约分" and updated["note"] == "已订正"

    # 删除：记录 404、照片文件也清掉
    assert api.client.delete(f"/api/questions/{question['id']}").status_code == 204
    assert api.client.get(f"/api/questions/{question['id']}").status_code == 404
    assert list(api.uploads.glob("*.png")) == []


def test_create_rejects_bad_extension(api):
    response = api.client.post(
        "/api/questions",
        data={"subject": "数学"},
        files={"photo": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "不支持的文件类型" in response.json()["detail"]


def test_create_rejects_empty_and_oversized_files(api):
    empty = _create_question(api, content=b"")
    assert empty.status_code == 400 and "内容为空" in empty.json()["detail"]

    oversized = _create_question(api, content=b"\0" * (10 * 1024 * 1024 + 1))
    assert oversized.status_code == 400 and "10MB" in oversized.json()["detail"]


def test_answer_masks_roundtrip_and_validation(api):
    question = _create_question(api).json()

    masks = [[100, 100, 200, 200], [400.5, 300.25, 150, 80]]
    saved = api.client.put(
        f"/api/questions/{question['id']}/answer-masks", json={"masks": masks}
    )
    assert saved.status_code == 200
    assert json.loads(saved.json()["answer_masks"]) == masks

    # 超出图片范围（千分比坐标，x+width > 1000）
    assert api.client.put(
        f"/api/questions/{question['id']}/answer-masks",
        json={"masks": [[900, 100, 200, 100]]},
    ).status_code == 400
    # 太小
    assert api.client.put(
        f"/api/questions/{question['id']}/answer-masks",
        json={"masks": [[100, 100, 2, 2]]},
    ).status_code == 400
    # 格式错误（不是 4 个元素）
    assert api.client.put(
        f"/api/questions/{question['id']}/answer-masks",
        json={"masks": [[100, 100, 200]]},
    ).status_code == 400


def test_import_page_splits_red_marks_into_unconfirmed_candidates(api):
    """整页上传：自动放平、按红叉切成候选错题，且都未确认（不进练习卷）。"""
    response = api.client.post(
        "/api/pages",
        data={"subject": "数学"},
        files={"photo": ("page.jpg", _synthetic_page_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["page_detected"] is True
    assert payload["red_mark_count"] == 3
    assert len(payload["questions"]) == 3
    for question in payload["questions"]:
        assert question["crop_reviewed"] is False
        assert question["source_image_url"]  # 归组到同一张原卷
        # 候选切片可通过 /uploads 访问
        assert api.client.get(question["image_url"]).status_code == 200
