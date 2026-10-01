import json
import shutil
import time
import uuid
from datetime import timezone
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from database import Base, add_column_if_missing, engine, get_db
from image_processing import crop_image_ranges, heic_to_jpeg_bytes, process_page
from models import Question

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _stored_name(path: str | Path) -> str:
    """入库只存文件名：questions.db 和 uploads/ 可以随项目整体搬家。"""
    return Path(path).name


def _resolve_upload(name: str | Path) -> Path:
    """库里的文件名 → 实际路径；兼容旧库残留的绝对路径（文件还在就照用）。"""
    path = Path(name)
    if path.is_absolute() and path.is_file():
        return path
    return UPLOAD_DIR / path.name


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic"}
HEIC_EXTENSIONS = {".heic", ".heif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
_CHUNK_SIZE = 256 * 1024


async def _read_upload(photo: UploadFile, limit: int = MAX_FILE_SIZE) -> bytes:
    """分块读取，超出上限立刻中止：大文件不会被先整份读进内存再判大小。"""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await photo.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(400, f"文件超过 {limit // (1024 * 1024)}MB 限制")
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(400, "文件内容为空")
    return b"".join(chunks)


def ensure_schema(target_engine) -> None:
    """建表 + 旧库补齐字段（幂等）。旧数据把确认状态记在文件名里，这里一并回填。"""
    Base.metadata.create_all(bind=target_engine)
    for _col, _ddl in [
        ("source_image_path", "TEXT DEFAULT ''"),
        ("answer_masks", "TEXT DEFAULT ''"),
        ("crop_reviewed", "BOOLEAN NOT NULL DEFAULT 0"),
    ]:
        add_column_if_missing(target_engine, "questions", _col, _ddl)
    with target_engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE questions SET crop_reviewed = 1 "
            "WHERE image_path LIKE '%\\_reviewed\\_%' ESCAPE '\\'"
        )
        # 旧库存的是绝对路径，项目一挪目录全断；统一收敛成文件名（幂等）。
        for qid, image_path, source_image_path in conn.exec_driver_sql(
            "SELECT id, image_path, source_image_path FROM questions"
        ).fetchall():
            new_image = _stored_name(image_path) if image_path else image_path
            new_source = (
                _stored_name(source_image_path)
                if source_image_path
                else source_image_path
            )
            if new_image != image_path or new_source != source_image_path:
                conn.execute(
                    text(
                        "UPDATE questions SET image_path = :image,"
                        " source_image_path = :source WHERE id = :id"
                    ),
                    {"image": new_image, "source": new_source, "id": qid},
                )


ensure_schema(engine)

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
        "crop_reviewed": bool(q.crop_reviewed),
        "answer_masks": q.answer_masks or "[]",
        "created_at": created_at.isoformat(),
    }


def _delete_file(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _delete_previous_previews(directory: Path, stem: str) -> None:
    for path in directory.glob(f"{stem}_recorrect_preview_*_page.jpg"):
        _delete_file(path)


def _sweep_abandoned_previews(
    directory: Path = UPLOAD_DIR, max_age_hours: int = 24
) -> None:
    cutoff = time.time() - max_age_hours * 3600
    for path in directory.glob("*_recorrect_preview_*_page.jpg"):
        try:
            if path.stat().st_mtime < cutoff:
                _delete_file(path)
        except OSError:
            pass


_sweep_abandoned_previews()


def _normalize_upload(content: bytes, ext: str) -> tuple[bytes, str]:
    """HEIC 浏览器无法直接显示，统一转码为 JPG 再入库。"""
    if ext in HEIC_EXTENSIONS:
        try:
            return heic_to_jpeg_bytes(content), ".jpg"
        except Exception as exc:
            raise HTTPException(400, f"HEIC 照片解码失败：{str(exc)[:200]}")
    return content, ext


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
        raise HTTPException(400, f"不支持的文件类型：{ext or '未知'}，仅支持 jpg/png/webp/gif/heic")

    content = await _read_upload(photo)
    content, ext = _normalize_upload(content, ext)

    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(content)

    # 单图直传没有走"校正+切割"流水线，对应原卷只有一张图，
    # 视为用户已确认，可直接进练习卷。
    question = Question(
        subject=subject.strip(),
        knowledge_point=knowledge_point.strip(),
        note=note.strip(),
        image_path=_stored_name(file_path),
        crop_reviewed=True,
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
        raise HTTPException(400, f"不支持的文件类型：{ext or '未知'}，仅支持 jpg/png/webp/gif/heic")
    content = await _read_upload(photo)
    content, ext = _normalize_upload(content, ext)

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
            image_path=_stored_name(crop_path),
            source_image_path=_stored_name(source_path),
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
        current_path = _resolve_upload(question.image_path)
        if not current_path.is_file():
            raise HTTPException(404, "已保存的照片文件不存在，请重新上传照片")
        source_path = UPLOAD_DIR / (
            f"{uuid.uuid4().hex}_original{current_path.suffix.lower() or '.jpg'}"
        )
        try:
            shutil.copy2(current_path, source_path)
        except OSError as exc:
            raise HTTPException(500, f"原图备份失败：{str(exc)[:200]}")
        source_image_path = _stored_name(source_path)
        question.source_image_path = source_image_path
        db.flush()
    else:
        source_path = _resolve_upload(source_image_path)
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
    confirmed: bool = False,
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
    old_paths = [_resolve_upload(item.image_path) for item in group]
    try:
        for item, crop_path in zip(group, new_paths):
            item.image_path = _stored_name(crop_path)
            item.knowledge_point = ""
            item.answer_masks = ""
            item.crop_reviewed = confirmed
        for crop_path in new_paths[len(group):]:
            db.add(
                Question(
                    subject=question.subject,
                    image_path=_stored_name(crop_path),
                    source_image_path=source_image_path,
                    crop_reviewed=confirmed,
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
        current_path = _resolve_upload(question.image_path)
        if not current_path.is_file():
            raise HTTPException(404, "已保存的照片文件不存在")
        source_path = UPLOAD_DIR / (
            f"{uuid.uuid4().hex}_original{current_path.suffix.lower() or '.jpg'}"
        )
        try:
            shutil.copy2(current_path, source_path)
        except OSError as exc:
            raise HTTPException(500, f"原图备份失败：{str(exc)[:200]}")
        question.source_image_path = _stored_name(source_path)
        db.commit()
    else:
        source_path = _resolve_upload(question.source_image_path)
    if not source_path.is_file():
        raise HTTPException(404, "原卷图片不存在")
    _delete_previous_previews(UPLOAD_DIR, source_path.stem)
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
        "page_detected": result.page_detected,
        "red_mark_count": result.mark_count,
        "ranges": [],
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
    source_path = _resolve_upload(question.source_image_path)
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
            question, db, question.source_image_path, crop_paths, confirmed=True
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


class AnswerMasksUpdate(BaseModel):
    masks: list[list[float]]


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


@app.put("/api/questions/{question_id}/answer-masks")
def update_answer_masks(
    question_id: int,
    payload: AnswerMasksUpdate,
    db: Session = Depends(get_db),
):
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    masks = []
    for item in payload.masks:
        if len(item) != 4:
            raise HTTPException(400, "遮挡区域格式错误")
        x, y, width, height = (float(value) for value in item)
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > 1000
            or y + height > 1000
        ):
            raise HTTPException(400, "遮挡区域超出图片范围")
        if width < 5 or height < 5:
            raise HTTPException(400, "遮挡区域太小")
        masks.append([round(x, 2), round(y, 2), round(width, 2), round(height, 2)])
    if len(masks) > 50:
        raise HTTPException(400, "单题最多设置 50 个遮挡区域")
    question.answer_masks = json.dumps(masks, ensure_ascii=False)
    db.commit()
    db.refresh(question)
    return question_to_dict(question)


@app.delete("/api/questions/{question_id}", status_code=204)
def delete_question(question_id: int, db: Session = Depends(get_db)):
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    source_path = question.source_image_path
    _delete_file(_resolve_upload(question.image_path))
    db.delete(question)
    db.commit()
    if source_path:
        still_used = (
            db.query(Question)
            .filter(Question.source_image_path == source_path)
            .first()
        )
        if not still_used:
            _delete_file(_resolve_upload(source_path))


@app.get("/uploads/{filename}")
def get_upload(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name:
        raise HTTPException(400, "非法的文件名")
    file_path = UPLOAD_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(file_path)
