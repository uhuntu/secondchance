import json
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ai import analyze_image, analyze_image_events
from database import Base, add_column_if_missing, engine, get_db
from figure import render_stack
from image_processing import process_page
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
    return {
        "id": q.id,
        "subject": q.subject,
        "knowledge_point": q.knowledge_point,
        "note": q.note,
        "image_url": f"/uploads/{Path(q.image_path).name}",
        "source_image_url": (
            f"/uploads/{Path(q.source_image_path).name}" if q.source_image_path else ""
        ),
        "created_at": q.created_at.isoformat(),
        "analyzed": bool(q.question_text or q.error_analysis),
        "question_text": q.question_text,
        "child_answer": q.child_answer,
        "error_analysis": q.error_analysis,
        "correct_answer": q.correct_answer,
        "explanation": q.explanation,
        "similar_questions": q.similar_questions,
    }


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
    db.commit()
    db.refresh(question)
    return question_to_dict(question)


@app.delete("/api/questions/{question_id}", status_code=204)
def delete_question(question_id: int, db: Session = Depends(get_db)):
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    source_path = question.source_image_path
    try:
        os.remove(question.image_path)
    except OSError:
        pass
    db.delete(question)
    db.commit()
    if source_path:
        still_used = (
            db.query(Question)
            .filter(Question.source_image_path == source_path)
            .first()
        )
        if not still_used:
            try:
                os.remove(source_path)
            except OSError:
                pass


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
