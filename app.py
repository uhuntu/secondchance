import json
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
import ai_split
from image_processing import (
    analyze,
    crop_by_lines,
    heic_to_jpeg_bytes,
    suggest_split_lines,
    warp_with_corners,
)
from models import Question

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_SPLITS = 20  # 一页最多切20 道，避免误操作把练习卷打成几十页
MIN_BAND_HEIGHT = 60  # 每道题至少这么高，否则切出来的片段没法看


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
        # 切题不再有"待确认"环节：录进来就能直接进练习卷。
        conn.exec_driver_sql("UPDATE questions SET crop_reviewed = 1")


ensure_schema(engine)

app = FastAPI(title="Second Chance")
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


def _normalize_upload(content: bytes, ext: str) -> tuple[bytes, str]:
    """HEIC 浏览器无法直接显示，统一转码为 JPG 再入库。"""
    if ext in HEIC_EXTENSIONS:
        try:
            return heic_to_jpeg_bytes(content), ".jpg"
        except Exception as exc:
            raise HTTPException(400, f"HEIC 照片解码失败：{str(exc)[:200]}")
    return content, ext


def _save_page(image, stem: str) -> Path:
    """存下这一页（已放平或原图），所有切割坐标都相对它。"""
    path = UPLOAD_DIR / f"{stem}_page.jpg"
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def _validate_lines(lines: list, height: int) -> list[int]:
    """切割线去重、排序、限量，并保证每道题不至于被切得太矮。"""
    cleaned: list[int] = []
    for line in lines:
        value = int(line)
        if not 0 < value < height:
            raise HTTPException(400, "切割线超出了这一页的范围")
        if not cleaned or value - cleaned[-1] >= MIN_BAND_HEIGHT:
            cleaned.append(value)
    if len(cleaned) > MAX_SPLITS:
        raise HTTPException(400, f"一页最多切 {MAX_SPLITS + 1} 道题")
    if cleaned and (cleaned[0] < MIN_BAND_HEIGHT or height - cleaned[-1] < MIN_BAND_HEIGHT):
        raise HTTPException(400, "切割线离页面边缘太近，会切出过窄的题目")
    return cleaned


def _split_into_crops(image, lines: list[int], stem: str) -> list[Path]:
    crops = crop_by_lines(image, lines)
    paths = []
    for index, crop in enumerate(crops, 1):
        path = UPLOAD_DIR / f"{stem}_q{index}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        paths.append(path)
    return paths


def _group_of(db: Session, source_image_path: str) -> list[Question]:
    return (
        db.query(Question)
        .filter(Question.source_image_path == source_image_path)
        .order_by(Question.id.asc())
        .all()
    )


def _apply_split(
    db: Session,
    question: Question,
    source_name: str,
    crop_paths: list[Path],
    rename_group_from: str | None = None,
) -> list[Question]:
    """把一页的题目整体换成新的切片集合。

    按序号保留旧的知识点/遮挡：用户拖动分割线时，第 2 道题通常还是第 2 道题，
    全清空等于逼他重录一遍。

    rename_group_from：整页换了新文件（手动摆正）时，把这一组一起改指向新整页，
    否则 source_image_path 会留在已删除的旧文件上。
    """
    group = (
        _group_of(db, rename_group_from) if rename_group_from else _group_of(db, source_name)
    )
    old_paths = [_resolve_upload(item.image_path) for item in group]
    try:
        for index, item in enumerate(group):
            if rename_group_from:
                item.source_image_path = source_name
            if index < len(crop_paths):
                item.image_path = _stored_name(crop_paths[index])
                item.crop_reviewed = True
            else:
                db.delete(item)
        for crop_path in crop_paths[len(group):]:
            db.add(
                Question(
                    subject=question.subject,
                    image_path=_stored_name(crop_path),
                    source_image_path=source_name,
                    crop_reviewed=True,
                )
            )
        db.commit()
    except Exception:
        db.rollback()
        for path in crop_paths:
            _delete_file(path)
        raise HTTPException(500, "保存切割结果失败，原记录未更改")

    for path in old_paths:
        if path not in crop_paths:
            _delete_file(path)
    return _group_of(db, source_name)


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
    """单图直传：一张照片就是一道错题，不做切割。"""
    ext = Path(photo.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型：{ext or '未知'}，仅支持 jpg/png/webp/gif/heic")

    content = await _read_upload(photo)
    content, ext = _normalize_upload(content, ext)

    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(content)

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


def _create_page(
    db: Session,
    subject: str,
    content: bytes,
    ext: str,
    page_points: list[list[float]] | None = None,
) -> dict:
    """整页入库：尽力放平 → 建议分割线 → 直接切成错题。

    分割只是建议：用户随后可以拖动分割线改，切得不对就删掉重来。
    """
    stem = uuid.uuid4().hex
    raw_path = UPLOAD_DIR / f"{stem}_raw{ext}"
    raw_path.write_bytes(content)
    try:
        analysis = analyze(raw_path)
        if page_points is not None:
            image = warp_with_corners(analysis.image, page_points)
            lines = ai_split.merged_lines(image, suggest_split_lines(image))
        else:
            image = analysis.image
            lines = ai_split.merged_lines(image, analysis.lines)
        page_detected = analysis.page_detected
        lines = _validate_lines(lines, image.shape[0])
        page_path = _save_page(image, stem)
        crop_paths = _split_into_crops(image, lines, stem)
    except HTTPException:
        _delete_file(raw_path)
        raise
    except ValueError as exc:
        _delete_file(raw_path)
        raise HTTPException(400, str(exc))
    except Exception as exc:
        _delete_file(raw_path)
        raise HTTPException(400, f"照片处理失败：{str(exc)[:200]}")
    _delete_file(raw_path)

    source_name = _stored_name(page_path)
    representative = Question(
        subject=subject.strip() or "数学",
        image_path=_stored_name(crop_paths[0]),
        source_image_path=source_name,
        crop_reviewed=True,
    )
    db.add(representative)
    try:
        db.flush()
        questions = _apply_split(db, representative, source_name, crop_paths)
    except HTTPException:
        _delete_file(page_path)
        raise
    return {
        "questions": [question_to_dict(q) for q in questions],
        "page_detected": page_detected,
        "source_image_url": f"/uploads/{source_name}",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "lines": lines,
    }


@app.post("/api/pages", status_code=201)
async def import_page(
    subject: str = Form("数学"),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """拍整页作业：自动切成错题，切得不理想再拖分割线调整。"""
    ext = Path(photo.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型：{ext or '未知'}，仅支持 jpg/png/webp/gif/heic")
    content = await _read_upload(photo)
    content, ext = _normalize_upload(content, ext)
    return _create_page(db, subject, content, ext)


class SplitRequest(BaseModel):
    lines: list[int]


class ManualStraighten(SplitRequest):
    points: list[list[float]] | None = None


@app.post("/api/questions/{question_id}/split")
def split_question(
    question_id: int,
    payload: SplitRequest,
    db: Session = Depends(get_db),
):
    """按用户确认的分割线重新切这一页。"""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    source_name = question.source_image_path
    if not source_name:
        raise HTTPException(400, "这道题没有整页原图，无法重新切割")
    page_path = _resolve_upload(source_name)
    image = cv2.imread(str(page_path))
    if image is None:
        raise HTTPException(404, "整页原图已丢失，请重新拍照上传")

    lines = _validate_lines(payload.lines, image.shape[0])
    crop_paths = _split_into_crops(image, lines, f"{page_path.stem}_s{uuid.uuid4().hex[:8]}")
    questions = _apply_split(db, question, source_name, crop_paths)
    return {"questions": [question_to_dict(q) for q in questions]}


@app.post("/api/questions/{question_id}/straighten")
def straighten_question(
    question_id: int,
    payload: ManualStraighten,
    db: Session = Depends(get_db),
):
    """拍照歪了、桌面上有干扰时：手动选四个角重新摆正，并给出新的建议分割线。"""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    if not payload.points:
        raise HTTPException(400, "请先选择纸张的四个角")
    source_name = question.source_image_path
    if not source_name:
        raise HTTPException(400, "这道题没有整页原图，无法手动摆正")
    page_path = _resolve_upload(source_name)
    raw = cv2.imread(str(page_path))
    if raw is None:
        raise HTTPException(404, "整页原图已丢失，请重新拍照上传")
    try:
        straightened = warp_with_corners(raw, payload.points)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    lines = ai_split.merged_lines(straightened, suggest_split_lines(straightened))
    _validate_lines(lines, straightened.shape[0])
    # 先写新图再删旧图：中途失败也不会把这一组的原图弄丢。
    new_path = _save_page(straightened, f"{page_path.stem}_m{uuid.uuid4().hex[:8]}")
    new_name = _stored_name(new_path)
    crop_paths = _split_into_crops(
        straightened, lines, f"{new_path.stem}_s{uuid.uuid4().hex[:8]}"
    )
    try:
        # 整组一起改指向新整页，否则 source_image_path 会指向已删除的旧文件。
        questions = _apply_split(db, question, new_name, crop_paths, rename_group_from=source_name)
    except HTTPException:
        _delete_file(new_path)
        raise
    _delete_file(page_path)
    return {
        "questions": [question_to_dict(q) for q in questions],
        "source_image_url": f"/uploads/{new_path.name}",
        "width": int(straightened.shape[1]),
        "height": int(straightened.shape[0]),
        "lines": lines,
    }


@app.post("/api/questions/{question_id}/suggest")
def suggest_question(question_id: int, db: Session = Depends(get_db)):
    """对已存的整页重新跑一遍空白带检测，只返回建议分割线，不改数据。"""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    if not question.source_image_path:
        raise HTTPException(400, "这道题没有整页原图")
    page_path = _resolve_upload(question.source_image_path)
    image = cv2.imread(str(page_path))
    if image is None:
        raise HTTPException(404, "整页原图已丢失，请重新拍照上传")
    try:
        return {
            "lines": _validate_lines(
                ai_split.merged_lines(image, suggest_split_lines(image)), image.shape[0]
            )
        }
    except HTTPException:
        # 建议线不合法（贴边/太密）时宁可不给，也不要给一堆废线。
        return {"lines": []}


@app.get("/api/questions/{question_id}/page")
def get_page(question_id: int, db: Session = Depends(get_db)):
    """重新打开切割编辑器时，取这一页的原图和当前分割线。"""
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    if not question.source_image_path:
        raise HTTPException(400, "这道题没有整页原图")
    page_path = _resolve_upload(question.source_image_path)
    image = cv2.imread(str(page_path))
    if image is None:
        raise HTTPException(404, "整页原图已丢失，请重新拍照上传")
    group = _group_of(db, question.source_image_path)
    height = int(image.shape[0])
    # 现存切片的高度换算成这一页上的分割线，编辑器打开就是上次的样子。
    lines = []
    offset = 0
    for item in group[:-1]:
        crop = cv2.imread(str(_resolve_upload(item.image_path)))
        offset += int(crop.shape[0]) if crop is not None else 0
        if offset < height:
            lines.append(offset)
    return {
        "source_image_url": f"/uploads/{page_path.name}",
        "width": int(image.shape[1]),
        "height": height,
        "lines": lines,
        "question_count": len(group),
    }


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