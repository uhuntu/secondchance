import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from database import Base, add_column_if_missing, engine, get_db
import ai_split
from image_processing import (
    analyze,
    crop_by_lines,
    remove_handwriting,
    heic_to_jpeg_bytes,
    suggest_split_lines,
    warp_with_corners,
)
from models import PageSplit, PracticeRecord, Question

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
        ("suspected_wrong", "BOOLEAN NOT NULL DEFAULT 0"),
        ("ink_boxes", "TEXT DEFAULT '[]'"),
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


def question_to_dict(q: Question, practice: dict[int, tuple[int, object]] | None = None) -> dict:
    created_at = q.created_at
    if created_at.tzinfo is None:
        # SQLite drops timezone metadata; stored timestamps are UTC.
        created_at = created_at.replace(tzinfo=timezone.utc)
    practice_count, last_practiced = 0, None
    if practice and q.id in practice:
        practice_count, last_practiced = practice[q.id]
        if last_practiced is not None:
            if last_practiced.tzinfo is None:
                # SQLite drops timezone metadata; stored timestamps are UTC.
                last_practiced = last_practiced.replace(tzinfo=timezone.utc)
            last_practiced = last_practiced.isoformat()
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
        "suspected_wrong": bool(q.suspected_wrong),
        "ink_boxes": q.ink_boxes or "[]",
        "answer_masks": q.answer_masks or "[]",
        "practice_count": practice_count,
        "last_practiced_at": last_practiced,
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


def _split_into_crops(image, lines: list[int], stem: str, skip_first: bool = False) -> list[Path]:
    crops = crop_by_lines(image, lines)
    if skip_first and len(crops) > 1:
        # AI 已确认第一片是页眉（校名、单元号……），不是题，直接丢弃
        crops = crops[1:]
    paths = []
    for index, crop in enumerate(crops, 1):
        path = UPLOAD_DIR / f"{stem}_q{index}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        paths.append(path)
    return paths


def _save_page_split(db: Session, source_name: str, lines: list[int], skip_first: bool) -> None:
    """把编辑器/AI 的切法存下来，编辑器下次打开原样还原。"""
    row = db.get(PageSplit, source_name)
    if row is None:
        row = PageSplit(source_image_path=source_name)
        db.add(row)
    row.lines = json.dumps([int(line) for line in lines])
    row.skip_first = bool(skip_first)
    db.commit()


def _load_page_split(db: Session, source_name: str) -> tuple[list[int], bool] | None:
    row = db.get(PageSplit, source_name)
    if row is None:
        return None
    try:
        return json.loads(row.lines), bool(row.skip_first)
    except (json.JSONDecodeError, TypeError):
        return None


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
    wrong_flags: list[bool | None] | None = None,
    ink_boxes: list[list[list[int]]] | None = None,
    knowledge_points: list[str | None] | None = None,
) -> list[Question]:
    """把一页的题目整体换成新的切片集合。

    按序号保留旧的知识点/遮挡：用户拖动分割线时，第 2 道题通常还是第 2 道题，
    全清空等于逼他重录一遍。

    rename_group_from：整页换了新文件（手动摆正）时，把这一组一起改指向新整页，
    否则 source_image_path 会留在已删除的旧文件上。

    wrong_flags：AI 对每道题的对错判断（与 crop_paths 对齐）。编辑器保存
    不带它——重切不动 AI 的判断。

    knowledge_points：AI 对每道题的知识点建议（与 crop_paths 对齐）。只往
    空着的知识点里填，用户手填的标注永不覆盖；编辑器保存不带它。
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
                if wrong_flags is not None and index < len(wrong_flags):
                    item.suspected_wrong = bool(wrong_flags[index])
                if ink_boxes is not None and index < len(ink_boxes):
                    item.ink_boxes = json.dumps(ink_boxes[index])
                if (
                    knowledge_points is not None
                    and index < len(knowledge_points)
                    and knowledge_points[index]
                    and not item.knowledge_point
                ):
                    item.knowledge_point = knowledge_points[index]
            else:
                db.delete(item)
        for index, crop_path in enumerate(crop_paths[len(group):]):
            new_index = len(group) + index
            suggestion = (
                knowledge_points[new_index]
                if knowledge_points and len(knowledge_points) > new_index
                else None
            )
            db.add(
                Question(
                    subject=question.subject,
                    image_path=_stored_name(crop_path),
                    source_image_path=source_name,
                    crop_reviewed=True,
                    suspected_wrong=bool(wrong_flags[new_index]) if wrong_flags and len(wrong_flags) > new_index else False,
                    ink_boxes=json.dumps(ink_boxes[new_index]) if ink_boxes and len(ink_boxes) > new_index else "[]",
                    knowledge_point=suggestion or "",
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
    rows = query.order_by(Question.id.asc()).all()
    # 同一页的题是同一瞬间批量入库的（Windows 时钟粒度下 created_at 甚至
    # 完全相同），按 created_at 倒序会把页内排成从下往上。改成"整页为单位"：
    # 组内 id 升序（= 切片从上到下），组间按组内最大 id 倒序（= 越新创建的
    # 页越靠上）。id 单调递增，不依赖时间戳精度。
    groups: dict[str, list[Question]] = {}
    for q in rows:
        groups.setdefault(q.source_image_path or f"single:{q.id}", []).append(q)
    questions: list[Question] = []
    for key in sorted(groups, key=lambda k: max(q.id for q in groups[k]), reverse=True):
        questions.extend(groups[key])
    # 练习记录聚合：一次 group-by 拿全量（次数 + 最近一次时间），避免逐题 N+1
    practice_rows = (
        db.query(
            PracticeRecord.question_id,
            func.count(PracticeRecord.id),
            func.max(PracticeRecord.created_at),
        )
        .group_by(PracticeRecord.question_id)
        .all()
    )
    practice = {qid: (count, last) for qid, count, last in practice_rows}
    return [question_to_dict(q, practice=practice) for q in questions]


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
            lines, ai_active, wrong_flags, hw_boxes, knowledge_flags = ai_split.split_plan(
                image, suggest_split_lines(image)
            )
        else:
            image = analysis.image
            lines, ai_active, wrong_flags, hw_boxes, knowledge_flags = ai_split.split_plan(
                image, analysis.lines
            )
        page_detected = analysis.page_detected
        lines = _validate_lines(lines, image.shape[0])
        page_path = _save_page(image, stem)
        crop_paths = _split_into_crops(image, lines, stem, skip_first=ai_active)
        ink_boxes_per_crop = ai_split.boxes_per_crop(hw_boxes, lines, image.shape[0], ai_active)
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
        questions = _apply_split(
            db, representative, source_name, crop_paths,
            wrong_flags=wrong_flags, ink_boxes=ink_boxes_per_crop,
            knowledge_points=knowledge_flags,
        )
    except HTTPException:
        _delete_file(page_path)
        raise
    _save_page_split(db, source_name, lines, ai_active)
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
    skip_first: bool = False  # 第一段是页眉（AI 重新自动切时会标记），保存时丢弃
    boxes: list[list[int]] = []  # AI 标注的手写痕迹框（整页坐标，随重新自动切透传）


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
    ink_boxes_per_crop = ai_split.boxes_per_crop(
        payload.boxes, lines, image.shape[0], payload.skip_first
    )
    crop_paths = _split_into_crops(
        image, lines, f"{page_path.stem}_s{uuid.uuid4().hex[:8]}",
        skip_first=payload.skip_first,
    )
    questions = _apply_split(
        db, question, source_name, crop_paths, ink_boxes=ink_boxes_per_crop,
    )
    _save_page_split(db, source_name, lines, payload.skip_first)
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
    lines, ai_active, wrong_flags, hw_boxes, knowledge_flags = ai_split.split_plan(
        straightened, suggest_split_lines(straightened)
    )
    _validate_lines(lines, straightened.shape[0])
    # 先写新图再删旧图：中途失败也不会把这一组的原图弄丢。
    new_path = _save_page(straightened, f"{page_path.stem}_m{uuid.uuid4().hex[:8]}")
    new_name = _stored_name(new_path)
    crop_paths = _split_into_crops(
        straightened, lines, f"{new_path.stem}_s{uuid.uuid4().hex[:8]}",
        skip_first=ai_active,
    )
    ink_boxes_per_crop = ai_split.boxes_per_crop(hw_boxes, lines, straightened.shape[0], ai_active)
    try:
        # 整组一起改指向新整页，否则 source_image_path 会指向已删除的旧文件。
        questions = _apply_split(
            db, question, new_name, crop_paths,
            rename_group_from=source_name, wrong_flags=wrong_flags,
            ink_boxes=ink_boxes_per_crop, knowledge_points=knowledge_flags,
        )
    except HTTPException:
        _delete_file(new_path)
        raise
    _save_page_split(db, new_name, lines, ai_active)
    old_split = db.get(PageSplit, source_name)
    if old_split is not None:
        db.delete(old_split)
        db.commit()
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
        lines, ai_active, _wrong_flags, hw_boxes, _knowledge_flags = ai_split.split_plan(
            image, suggest_split_lines(image)
        )
        return {
            "lines": _validate_lines(lines, image.shape[0]),
            "ai_active": ai_active,
            "boxes": hw_boxes,
        }
    except HTTPException:
        # 建议线不合法（贴边/太密）时宁可不给，也不要给一堆废线。
        return {"lines": [], "ai_active": False, "boxes": []}


@app.post("/api/questions/{question_id}/judge")
def judge_question_page(question_id: int, db: Session = Depends(get_db)):
    """AI 重新判这一页每道题的对错，更新卡片上的疑似错题徽标。

    判断按页内位置落到切片上：一张切片里套着几道 AI 识别的题，
    就继承它们的判断（老页面的粗切片里有一道错，整片标疑似）。
    顺带把 AI 的知识点建议填进空着的知识点；已有标注不动。
    AI 没参与（没配 key、网络失败）时不动任何标记。
    """
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
        lines, ai_active, wrong_flags, hw_boxes, knowledge_flags = ai_split.split_plan(
            image, suggest_split_lines(image)
        )
    except HTTPException:
        return {"updated": False, "suspected": 0, "tagged": 0, "ai_active": False}
    if not ai_active:
        return {"updated": False, "suspected": 0, "tagged": 0, "ai_active": False}

    group = _group_of(db, question.source_image_path)
    # 每张切片按页内位置圈定范围，套进来的 AI 判断取"或"
    tops = [0]
    for item in group[:-1]:
        crop = cv2.imread(str(_resolve_upload(item.image_path)))
        tops.append(tops[-1] + (int(crop.shape[0]) if crop is not None else 0))
    bounds = [*tops, image.shape[0] + 1]

    suspected = 0
    tagged = 0
    for index, item in enumerate(group):
        top, bottom = bounds[index], bounds[index + 1]
        inside = [
            (bool(w), k)
            for line, w, k in zip(lines, wrong_flags, knowledge_flags)
            if top <= line < bottom
        ]
        if inside:
            item.suspected_wrong = any(w for w, _k in inside)
            if item.suspected_wrong:
                suspected += 1
            if not item.knowledge_point:
                suggestion = next((k for _w, k in inside if k), None)
                if suggestion:
                    item.knowledge_point = suggestion
                    tagged += 1
    # 顺带把 AI 的手写痕迹框按题存下来（练习卷去手写用）。
    # 只在 AI 的题数和现存切片数一致时做，对不上的老页面请先重新自动切。
    if len(lines) == len(group):
        per_crop = ai_split.boxes_per_crop(
            hw_boxes, lines, image.shape[0], skip_first=True
        )
        for index, item in enumerate(group):
            if index < len(per_crop):
                item.ink_boxes = json.dumps(per_crop[index])
    db.commit()
    return {"updated": True, "suspected": suspected, "tagged": tagged, "ai_active": True}


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
    # 优先用存下来的切法（含页眉边界）；老页面没存过，才用切片高度反推。
    stored = _load_page_split(db, question.source_image_path)
    if stored is not None:
        lines, skip_first = stored
    else:
        skip_first = False
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
        "skip_first": skip_first,
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


@app.delete("/api/questions/{question_id}/page", status_code=204)
def delete_question_page(question_id: int, db: Session = Depends(get_db)):
    """整页删除：同一页的全部题、所有切片和共享的整页原图一起删。

    页以 source_image_path 圈定；单图直传的题没有整页，退化为普通单题删除。
    """
    question = db.get(Question, question_id)
    if not question:
        raise HTTPException(404, "错题不存在")
    source_name = question.source_image_path
    group = _group_of(db, source_name) if source_name else [question]
    for item in group:
        _delete_file(_resolve_upload(item.image_path))
        db.delete(item)
    db.commit()
    if source_name:
        _delete_file(_resolve_upload(source_name))


class PrintRequest(BaseModel):
    question_ids: list[int]


@app.post("/api/prints", status_code=201)
def record_print(payload: PrintRequest, db: Session = Depends(get_db)):
    """打印练习卷时自动记一笔：哪些题、哪天练的。零操作步骤，失败不影响打印。"""
    ids = sorted(set(payload.question_ids))
    existing = {
        row.id for row in db.query(Question.id).filter(Question.id.in_(ids))
    }
    if not existing:
        raise HTTPException(400, "没有可记录的题目")
    paper_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    db.add_all(
        PracticeRecord(paper_id=paper_id, question_id=qid, created_at=now)
        for qid in existing
    )
    db.commit()
    return {
        "paper_id": paper_id,
        "recorded": len(existing),
        "created_at": now.isoformat(),
    }


@app.get("/api/prints")
def list_prints(db: Session = Depends(get_db)):
    """出卷历史：最近 10 张卷，新→旧；已删的题只影响明细，不影响次数。"""
    rows = (
        db.query(PracticeRecord)
        .order_by(PracticeRecord.created_at.desc(), PracticeRecord.id.desc())
        .all()
    )
    papers: dict[str, list[PracticeRecord]] = {}
    for row in rows:
        papers.setdefault(row.paper_id, []).append(row)

    result = []
    for paper_id, items in list(papers.items())[:10]:
        # 按题 id 升序 = 卷面从上到下（同批入库 id 单调）
        question_ids = sorted(item.question_id for item in items)
        found = {
            q.id: q
            for q in db.query(Question).filter(Question.id.in_(question_ids))
        }
        result.append({
            "paper_id": paper_id,
            "created_at": items[0].created_at.replace(tzinfo=timezone.utc).isoformat(),
            "count": len(items),
            "questions": [
                {
                    "id": found[qid].id,
                    "subject": found[qid].subject,
                    "knowledge_point": found[qid].knowledge_point,
                    "image_url": f"/uploads/{Path(found[qid].image_path).name}",
                }
                for qid in question_ids if qid in found
            ],
        })
    return result


@app.get("/uploads/{filename}")
def get_upload(filename: str, clean: int = 0, db: Session = Depends(get_db)):
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name:
        raise HTTPException(400, "非法的文件名")
    file_path = UPLOAD_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(404, "文件不存在")
    if clean:
        # 练习卷的"去手写"模式：现场生成铅笔/红笔/深笔迹移除版，原图不动
        image = cv2.imread(str(file_path))
        if image is not None:
            regions: list[list[int]] = []
            row = db.query(Question).filter(Question.image_path == filename).first()
            if row is not None:
                try:
                    regions = json.loads(row.ink_boxes or "[]")
                except json.JSONDecodeError:
                    regions = []
            cleaned = remove_handwriting(image, extra_regions=regions)
            ok, buffer = cv2.imencode(".jpg", cleaned, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                return Response(content=buffer.tobytes(), media_type="image/jpeg")
    return FileResponse(file_path)