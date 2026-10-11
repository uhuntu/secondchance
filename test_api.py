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
from models import Question

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _synthetic_page_jpeg() -> bytes:
    """一张带边框和三个红叉的作业页，用来验证整页上传能切成多道错题。"""
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (45, 35), (755, 1165), (20, 20, 20), 4)
    for y in (250, 650, 980):
        cv2.line(page, (600, y - 25), (650, y + 25), (0, 0, 220), 10)
        cv2.line(page, (650, y - 25), (600, y + 25), (0, 0, 220), 10)
    ok, buffer = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buffer.tobytes()


def _blocky_page_jpeg(height: int = 1400, width: int = 900) -> bytes:
    """三段文字 + 两段明显空白带，空白带应该被当成题间分割线。"""
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    for top, bottom in [(100, 380), (560, 840), (1020, 1300)]:
        for y in range(top, bottom, 20):
            cv2.line(page, (90, y), (width - 90, y), (40, 40, 40), 3)
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
    # 没有"待确认"环节：录进来就能直接进练习卷
    assert question["crop_reviewed"] is True

    listed = api.client.get("/api/questions").json()
    assert any(q["id"] == question["id"] for q in listed)
    assert all(q["subject"] == "数学" for q in api.client.get("/api/questions", params={"subject": "数学"}).json())
    assert any(q["id"] == question["id"] for q in api.client.get("/api/questions", params={"keyword": "分数"}).json())
    assert api.client.get("/api/questions", params={"keyword": "不存在的词"}).json() == []

    detail = api.client.get(f"/api/questions/{question['id']}").json()
    assert detail["note"] == "正确答案 3/4"
    image = api.client.get(detail["image_url"])
    assert image.status_code == 200 and image.content == PNG

    updated = api.client.put(
        f"/api/questions/{question['id']}",
        json={"knowledge_point": "约分", "note": "已订正"},
    ).json()
    assert updated["knowledge_point"] == "约分" and updated["note"] == "已订正"

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


def test_records_store_filenames_only(api):
    """入库只存文件名：questions.db 和 uploads/ 可随项目整体搬家。"""
    created = _create_question(api).json()
    db = api.Session()
    try:
        question = db.get(Question, created["id"])
        assert question.image_path == Path(created["image_url"]).name
    finally:
        db.close()


def test_import_page_splits_into_ready_to_use_questions(api):
    """整页上传：自动切成多道错题，且都是可直接进练习卷的已确认状态。"""
    response = api.client.post(
        "/api/pages",
        data={"subject": "数学"},
        files={"photo": ("page.jpg", _blocky_page_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["lines"], "两段空白带应该被识别成题间分割线"
    assert len(payload["questions"]) == len(payload["lines"]) + 1
    for question in payload["questions"]:
        # 关键：录进来就是已确认，不会卡在"还没确认切割"的状态
        assert question["crop_reviewed"] is True
        assert question["source_image_url"]
        assert api.client.get(question["image_url"]).status_code == 200
    # 原卷和切片在库里都只存文件名
    db = api.Session()
    try:
        stored = [db.get(Question, q["id"]) for q in payload["questions"]]
        assert all("\\" not in q.image_path and "/" not in q.image_path for q in stored)
        assert all("\\" not in q.source_image_path for q in stored)
    finally:
        db.close()


def test_split_endpoint_recuts_the_same_page(api):
    """拖动分割线重新切题：题目数量跟着变，整页原图保留。"""
    payload = api.client.post(
        "/api/pages",
        data={"subject": "数学"},
        files={"photo": ("page.jpg", _blocky_page_jpeg(), "image/jpeg")},
    ).json()
    first = payload["questions"][0]

    height = payload["height"]
    # 故意切成 4 道，看数量是否跟着分割线走
    recut = api.client.post(
        f"/api/questions/{first['id']}/split",
        json={"lines": [height // 4, height // 2, (height * 3) // 4]},
    )
    assert recut.status_code == 200, recut.text
    assert len(recut.json()["questions"]) == 4

    # 原来的整页原图还能取到（"看整页"/"调整切题"依赖它）
    page_meta = api.client.get(f"/api/questions/{first['id']}/page")
    assert page_meta.status_code == 200
    assert api.client.get(page_meta.json()["source_image_url"]).status_code == 200


def test_split_preserves_knowledge_point_by_position(api):
    """按序号保留知识点：拖动分割线时不必把标注重录一遍。"""
    payload = api.client.post(
        "/api/pages",
        data={"subject": "数学"},
        files={"photo": ("page.jpg", _blocky_page_jpeg(), "image/jpeg")},
    ).json()
    first = payload["questions"][0]
    api.client.put(
        f"/api/questions/{first['id']}",
        json={"knowledge_point": "小数除法", "note": "第一道"},
    )

    api.client.post(
        f"/api/questions/{first['id']}/split", json={"lines": []}
    )
    kept = api.client.get(f"/api/questions/{first['id']}").json()
    assert kept["knowledge_point"] == "小数除法"
    assert kept["note"] == "第一道"


def test_split_rejects_out_of_range_and_tiny_bands(api):
    payload = api.client.post(
        "/api/pages",
        data={"subject": "数学"},
        files={"photo": ("page.jpg", _blocky_page_jpeg(), "image/jpeg")},
    ).json()
    first = payload["questions"][0]
    url = f"/api/questions/{first['id']}/split"

    # 超出页面范围
    assert api.client.post(url, json={"lines": [999999]}).status_code == 400
    # 贴边会切出过窄的题
    assert api.client.post(url, json={"lines": [2]}).status_code == 400
    # 切得过密
    assert api.client.post(url, json={"lines": [100, 110, 120, 130]}).status_code == 200


def test_suggest_endpoint_returns_lines_without_changing_data(api):
    payload = api.client.post(
        "/api/pages",
        data={"subject": "数学"},
        files={"photo": ("page.jpg", _blocky_page_jpeg(), "image/jpeg")},
    ).json()
    first = payload["questions"][0]
    before = len(api.client.get("/api/questions").json())

    suggested = api.client.post(f"/api/questions/{first['id']}/suggest")
    assert suggested.status_code == 200
    assert len(suggested.json()["lines"]) == len(payload["lines"])
    # 只是建议，不落库
    assert len(api.client.get("/api/questions").json()) == before


def test_straighten_repoints_group_at_the_new_page(api):
    """手动摆正会换掉整页文件：整组必须一起改指向新文件，否则原图就"丢了"。

    这是一条真实踩过的坑：曾经摆正时先删旧页、又没更新 source_image_path，
    结果这一组之后的"调整切题"全部 404。
    """
    payload = api.client.post(
        "/api/pages",
        data={"subject": "数学"},
        files={"photo": ("page.jpg", _blocky_page_jpeg(), "image/jpeg")},
    ).json()
    first = payload["questions"][0]
    before_src = payload["source_image_url"]
    assert before_src.startswith("/uploads/")

    width, height = payload["width"], payload["height"]
    corners = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    straightened = api.client.post(
        f"/api/questions/{first['id']}/straighten",
        json={"points": corners, "lines": []},
    )
    assert straightened.status_code == 200, straightened.text
    new_src = straightened.json()["source_image_url"]
    assert new_src != before_src

    # 旧整页应该已被删除，新整页可访问
    assert api.client.get(before_src).status_code == 404
    assert api.client.get(new_src).status_code == 200

    # 关键：库里每一行的 source_image_path 都指向仍然存在的新整页
    db = api.Session()
    try:
        rows = db.query(Question).filter(Question.source_image_path != "").all()
        assert rows
        for row in rows:
            assert row.source_image_path == Path(new_src).name
            assert (api.uploads / row.source_image_path).is_file()
    finally:
        db.close()

    # 重新打开编辑器必须成功（曾经这里直接 404）
    reopened = api.client.get(f"/api/questions/{first['id']}/page")
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["source_image_url"] == new_src


def test_deleting_a_question_keeps_the_page_while_siblings_exist(api):
    payload = api.client.post(
        "/api/pages",
        data={"subject": "数学"},
        files={"photo": ("page.jpg", _blocky_page_jpeg(), "image/jpeg")},
    ).json()
    questions = payload["questions"]
    assert len(questions) >= 2
    source_url = questions[0]["source_image_url"]

    assert api.client.delete(f"/api/questions/{questions[0]['id']}").status_code == 204
    # 还有同页的题在用原图，原图必须留着
    assert api.client.get(source_url).status_code == 200

    for question in questions[1:]:
        api.client.delete(f"/api/questions/{question['id']}")
    # 最后一道删完，原图才被清掉
    assert api.client.get(source_url).status_code == 404