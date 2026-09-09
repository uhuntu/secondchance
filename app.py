import json
import shutil
import uuid
from datetime import timezone
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ai import analyze_image, analyze_image_events
from database import Base, add_column_if_missing, engine, get_db
from figure import render_stack
from image_processing import crop_image_ranges, process_page
from models import Question

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

Base.metadata.create_all(bind=engine)
# 旧库补齐 AI 分析字段（幂等）
for _col, _ddl in [
    ("question_text", "TEXT DEFAULT ''"),
    ("child_answer", "TEXT DEFAULT ''"),
    ("error_analysis", "TEXT DEFAULT ''"),
    ("correct_answer", "TEXT DEFAULT ''"),
    ("explanation", "TEXT DEFAULT ''"),
    ("similar_questions", "TEXT DEFAULT ''"),
    ("original_redo", "TEXT DEFAULT ''"),
    ("source_image_path", "TEXT DEFAULT ''"),
]:
    add_column_if_missing("questions", _col, _ddl)

app = FastAPI(title="错题采集系统")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def question_to_dict(q: Question) -> dict:
    created_at = q.created_at
    if created_at.tzinfo is None:
        # SQLite drops timezone metadata; stored timestamps are UTC.
        created_at = created_at.replace(tzinfo=timezone.utc)
    return {
        "id": q.id,
        "subject": q.subject,
        "knowledge_point": q.knowledge_point,
        "note": q.note,
        "image_url": f"/uploads/{Path(q.image_path).name}",
        "source_image_url": (
            f"/uploads/{Path(q.source_image_path).name}" if q.source_image_path else ""
        ),
        "created_at": created_at.isoformat(),
        "analyzed": bool(q.question_text or q.error_analysis),
        "question_text": q.question_text,
        "child_answer": q.child_answer,
        "error_analysis": q.error_analysis,
        "correct_answer": q.correct_answer,
        "explanation": q.explanation,
        "similar_questions": q.similar_questions,
        "original_redo": q.original_redo,
    }


def _delete_file(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/questions")
def list_questions(
    subject: str | None = None,
    keyword: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Question)
    if subject:
        query = query.filter(Question.subject == subject)
    if keyword:
        like = f"%{keyword}%"
        query = query.filter(
            or_(Question.knowledge_point.like(like), Question.note.like(like))
        )
    questions = query.order_by(Question.created_at.desc()).all()
    return [question_to_dict(q) for q in questions]


@app.post("/api/questions", status_code=201)
async def create_question(
    subject: str = Form(...),
    knowledge_point: str = Form(""),
    note: str = Form(""),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ext = Path(photo.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型：{ext or '未知'}，仅支持 jpg/png/webp")

    content = await photo.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "文件超过 10MB 限制")
    if not content:
        raise HTTPException(400, "文件内容为空")

    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(content)

    question = Question(
        subject=subject.strip(),
        knowledge_point=knowledge_point.strip(),
        note=note.strip(),
        image_path=str(file_path),
    )
    db.add(question)
    db.commit()
    db.refresh(question)
    return question_to_dict(question)


@app.post("/api/pages", status_code=201)
async def import_page(
    subject: str = Form("数学"),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """保存整页原图，自动校正并按红色批改标记切成候选错题。"""
    ext = Path(photo.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型：{ext or '未知'}，仅支持 jpg/png/webp")
    content = await photo.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "文件超过 10MB 限制")
    if not content:
        raise HTTPException(400, "文件内容为空")

    stem = uuid.uuid4().hex
    source_path = UPLOAD_DIR / f"{stem}_original{ext}"
    source_path.write_bytes(content)
    try:
        result = process_page(source_path, UPLOAD_DIR, stem)
    except Exception as exc:
        source_path.unlink(missing_ok=True)
        raise HTTPException(400, f"照片处理失败：{str(exc)[:200]}")
    result.corrected_path.unlink(missing_ok=True)

    questions = []
    for crop_path in result.crop_paths:
        question = Question(
            subject=subject.strip() or "数学",
            image_path=str(crop_path),
            source_image_path=str(source_path),
        )
        db.add(question)
        questions.append(question)
    db.commit()
    for question in questions:
        db.refresh(question)
    return {
        "questions": [question_to_dict(question) for question in questions],
        "page_detected": result.page_detected,
        "red_mark_count": result.mark_count,
    }


def _recorrect_saved_page(
    question: Question,
    db: Session,
    page_points: list[list[float]] | None = None,
):
    source_image_path = question.source_image_path
    if not source_image_path:
        current_path = Path(question.image_path)
        if not current_path.is_file():
            raise HTTPException(404, "已保存的照片文件不存在，请重新上传照片")
        source_path = UPLOAD_DIR / (
            f"{uuid.uuid4().hex}_original{current_path.suffix.lower() or '.jpg'}"
        )
        try:
            shutil.copy2(current_path, source_path)
        except OSError as exc:
            raise HTTPException(500, f"原图备份失败：{str(exc)[:200]}")
        source_image_path = str(source_path)
        question.source_image_path = source_image_path
        db.flush()
    else:
        source_path = Path(source_image_path)
    if not source_path.is_file():
        raise HTTPException(404, "整张原卷文件已丢失，请重新上传照片")

    stem = f"{source_path.stem}_recorrect_{uuid.uuid4().hex[:10]}"
    try:
        result = process_page(source_path, UPLOAD_DIR, stem, page_points)
    except Exception as exc:
        raise HTTPException(400, f"重新校正失败：{str(exc)[:200]}")
    result.corrected_path.unlink(missing_ok=True)
    return _replace_question_crops(
        question, db, source_image_path, result.crop_paths, result
    )


def _replace_question_crops(
    question: Question,
    db: Session,
    source_image_path: str,
    crop_paths: list[Path],
    result=None,
):
    group = (
        db.query(Question)
        .filter(
            (Question.source_image_path == source_image_path)
            | (Question.id == question.id)
        )
        .order_by(Question.id.asc())
        .all()
    )
    new_paths = list(crop_paths)
    old_paths = [Path(item.image_path) for item in group]
    try:
        for item, crop_path in zip(group, new_paths):
            item.image_path = str(crop_path)
        for crop_path in new_paths[len(group):]:
            db.add(
                Question(
                    subject=question.subject,
                    image_path=str(crop_path),
                    source_image_path=source_image_path,
                )
            )
        for item in group[len(new_paths):]:
            db.delete(item)
        db.commit()
    except Exception:
        db.rollback()
        for path in new_paths:
            _delete_file(path)
        raise HTTPException(500, "重新校正结果保存失败，原记录未更改")

    for path in old_paths:
        if path not in new_paths:
            _delete_file(path)
    refreshed = (
        db.query(Question)
        .filter(Question.source_image_path == source_image_path)
        .order_by(Question.id.asc())
        .all()
    )
    return {
        "questions": [question_to_dict(item) for item in refreshed],
        "page_detected": result.page_detected if result else True,
        "red_mark_count": result.mark_count if result else len(new_paths),
        "updated_count": min(len(group), len(new_paths)),
        "created_count": max(0, len(new_paths) - len(group)),
        "removed_count": max(0, len(group) - len(new_paths)),
    }


class ManualCorrection(BaseModel):
    points: list[list[float]] | None = None


class CropConfirmation(BaseModel):
    preview_filename: str
    ranges: list[list[int]]


@app.post("/api/questions/{question_id}/recorrect")
def recorrect_saved_page(question_id: int, db: Session = Depends(get_db)):
    """Automatically re-run correction/cropping for one saved source page."""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    return _recorrect_saved_page(question, db)


@app.post("/api/questions/{question_id}/recorrect/manual")
def manually_recorrect_saved_page(
    question_id: int,
    payload: ManualCorrection,
    db: Session = Depends(get_db),
):
    """Correct/crop one saved page using four user-selected paper corners."""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    return _recorrect_saved_page(question, db, payload.points)


@app.post("/api/questions/{question_id}/recorrect/preview")
def preview_manual_recorrection(
    question_id: int,
    payload: ManualCorrection,
    db: Session = Depends(get_db),
):
    """Create a corrected-page preview without changing saved question crops."""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    if not question.source_image_path:
        current_path = Path(question.image_path)
        if not current_path.is_file():
            raise HTTPException(404, "已保存的照片文件不存在")
        source_path = UPLOAD_DIR / (
            f"{uuid.uuid4().hex}_original{current_path.suffix.lower() or '.jpg'}"
        )
        try:
            shutil.copy2(current_path, source_path)
        except OSError as exc:
            raise HTTPException(500, f"原图备份失败：{str(exc)[:200]}")
        question.source_image_path = str(source_path)
        db.commit()
    else:
        source_path = Path(question.source_image_path)
    if not source_path.is_file():
        raise HTTPException(404, "原卷图片不存在")
    stem = f"{source_path.stem}_recorrect_preview_{uuid.uuid4().hex[:10]}"
    try:
        result = process_page(source_path, UPLOAD_DIR, stem, payload.points)
    except Exception as exc:
        raise HTTPException(400, f"放平预览失败：{str(exc)[:200]}")
    for crop_path in result.crop_paths:
        _delete_file(crop_path)
    image = cv2.imread(str(result.corrected_path))
    if image is None:
        _delete_file(result.corrected_path)
        raise HTTPException(500, "无法读取放平预览")
    height, width = image.shape[:2]
    return {
        "preview_url": f"/uploads/{result.corrected_path.name}",
        "preview_filename": result.corrected_path.name,
        "width": width,
        "height": height,
        "ranges": [list(item) for item in result.crop_ranges],
    }


@app.post("/api/questions/{question_id}/recorrect/confirm")
def confirm_manual_recrops(
    question_id: int,
    payload: CropConfirmation,
    db: Session = Depends(get_db),
):
    """Save user-reviewed vertical crop ranges from a corrected-page preview."""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    if not question.source_image_path:
        raise HTTPException(400, "该记录没有整张原卷")
    source_path = Path(question.source_image_path)
    filename = Path(payload.preview_filename).name
    if (
        filename != payload.preview_filename
        or not filename.startswith(f"{source_path.stem}_recorrect_preview_")
        or not filename.endswith("_page.jpg")
    ):
        raise HTTPException(400, "预览文件无效")
    preview_path = UPLOAD_DIR / filename
    image = cv2.imread(str(preview_path))
    if image is None:
        raise HTTPException(404, "预览已失效，请重新选取四角")
    height, width = image.shape[:2]
    ranges = []
    for item in payload.ranges:
        if len(item) != 2:
            raise HTTPException(400, "切割范围格式错误")
        top, bottom = (int(item[0]), int(item[1]))
        if top < 0 or bottom > height or bottom - top < 80:
            raise HTTPException(400, "切割范围无效或过小")
        ranges.append((top, bottom))
    if not ranges or len(ranges) > 30:
        raise HTTPException(400, "请保留 1 到 30 个错题范围")
    ranges.sort()
    crops = crop_image_ranges(image, ranges)
    stem = f"{source_path.stem}_reviewed_{uuid.uuid4().hex[:10]}"
    crop_paths = []
    for index, crop in enumerate(crops, 1):
        path = UPLOAD_DIR / f"{stem}_q{index}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        crop_paths.append(path)
    try:
        response = _replace_question_crops(
            question, db, question.source_image_path, crop_paths
        )
    finally:
        _delete_file(preview_path)
    return response


@app.delete("/api/recorrect/preview/{filename}", status_code=204)
def discard_recorrection_preview(filename: str):
    """Delete an unconfirmed temporary correction preview."""
    safe_name = Path(filename).name
    if (
        safe_name != filename
        or "_recorrect_preview_" not in safe_name
        or not safe_name.endswith("_page.jpg")
    ):
        raise HTTPException(400, "预览文件无效")
    _delete_file(UPLOAD_DIR / safe_name)


class QuestionUpdate(BaseModel):
    subject: str | None = None
    knowledge_point: str | None = None
    note: str | None = None


@app.get("/api/questions/{question_id}")
def get_question(question_id: int, db: Session = Depends(get_db)):
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    return question_to_dict(question)


@app.put("/api/questions/{question_id}")
def update_question(
    question_id: int,
    payload: QuestionUpdate,
    db: Session = Depends(get_db),
):
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    if payload.subject is not None:
        question.subject = payload.subject.strip()
    if payload.knowledge_point is not None:
        question.knowledge_point = payload.knowledge_point.strip()
    if payload.note is not None:
        question.note = payload.note.strip()
    db.commit()
    db.refresh(question)
    return question_to_dict(question)


@app.post("/api/questions/{question_id}/analyze/stream")
def analyze_question_stream(question_id: int, db: Session = Depends(get_db)):
    """流式分析：SSE 实时推送模型思考过程，最后推送分析结果"""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")

    def gen():
        try:
            for kind, payload in analyze_image_events(question.image_path):
                if kind == "reasoning":
                    yield f"data: {json.dumps({'type': 'reasoning', 'text': payload}, ensure_ascii=False)}\n\n"
                elif kind == "content":
                    yield f"data: {json.dumps({'type': 'content', 'text': payload}, ensure_ascii=False)}\n\n"
                elif kind == "result":
                    question.knowledge_point = payload.get("知识点", question.knowledge_point)[:200]
                    question.question_text = payload.get("题目", "")
                    question.child_answer = payload.get("孩子的作答", "")
                    question.error_analysis = payload.get("错因分析", "")
                    question.correct_answer = payload.get("正确答案", "")
                    question.explanation = payload.get("讲解", "")
                    try:
                        question.similar_questions = json.dumps(
                            payload.get("举一反三", []), ensure_ascii=False
                        )
                    except (TypeError, ValueError):
                        question.similar_questions = "[]"
                    try:
                        question.original_redo = json.dumps(
                            payload.get("原题重练", {}), ensure_ascii=False
                        )
                    except (TypeError, ValueError):
                        question.original_redo = ""
                    db.commit()
                    db.refresh(question)
                    yield f"data: {json.dumps({'type': 'done', 'question': question_to_dict(question)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:500]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/questions/{question_id}/analyze")
def analyze_question(question_id: int, db: Session = Depends(get_db)):
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    try:
        result = analyze_image(question.image_path)
    except (RuntimeError, json.JSONDecodeError, KeyError, OSError) as e:
        raise HTTPException(502, f"AI 分析失败：{e}")

    question.knowledge_point = result.get("知识点", question.knowledge_point)[:200]
    question.question_text = result.get("题目", "")
    question.child_answer = result.get("孩子的作答", "")
    question.error_analysis = result.get("错因分析", "")
    question.correct_answer = result.get("正确答案", "")
    question.explanation = result.get("讲解", "")
    try:
        question.similar_questions = json.dumps(
            result.get("举一反三", []), ensure_ascii=False
        )
    except (TypeError, ValueError):
        question.similar_questions = "[]"
    try:
        question.original_redo = json.dumps(
            result.get("原题重练", {}), ensure_ascii=False
        )
    except (TypeError, ValueError):
        question.original_redo = ""
    db.commit()
    db.refresh(question)
    return question_to_dict(question)


@app.delete("/api/questions/{question_id}", status_code=204)
def delete_question(question_id: int, db: Session = Depends(get_db)):
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    source_path = question.source_image_path
    _delete_file(question.image_path)
    db.delete(question)
    db.commit()
    if source_path:
        still_used = (
            db.query(Question)
            .filter(Question.source_image_path == source_path)
            .first()
        )
        if not still_used:
            _delete_file(source_path)


@app.get("/uploads/{filename}")
def get_upload(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(file_path)


@app.get("/api/stack-figure")
def stack_figure(h: str):
    """把立体高度图（二维数组）渲染成方块示意图 PNG"""
    try:
        heights = json.loads(h)
        assert isinstance(heights, list) and heights
        assert all(isinstance(r, list) and r for r in heights)
        assert len(heights) <= 8 and max(len(r) for r in heights) <= 8
        flat = [v for r in heights for v in r]
        assert all(isinstance(v, int) and 0 <= v <= 10 for v in flat)
        assert 0 < sum(1 for v in flat if v) <= 64
    except (ValueError, AssertionError):
        raise HTTPException(400, "非法的高度图参数")
    return FileResponse(render_stack(heights))
