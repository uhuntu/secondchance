"""Page straightening and suggested split lines.

Design rule: every automatic step is a *suggestion*. A failure must never block
the user, so each helper returns a usable fallback instead of raising. The
frontend turns suggestions into draggable split lines the user can adjust,
delete, or ignore entirely.
"""

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()


def heic_to_jpeg_bytes(content: bytes) -> bytes:
    """HEIC 照片浏览器无法直接显示，入库前统一转码为 JPG 字节流。"""
    with Image.open(io.BytesIO(content)) as img:
        buffer = io.BytesIO()
        ImageOps.exif_transpose(img).convert("RGB").save(buffer, "JPEG", quality=92)
        return buffer.getvalue()


def load_image(source: Path | str) -> np.ndarray:
    """Read any supported photo into a BGR array, honouring EXIF rotation."""
    with Image.open(source) as pil:
        rgb = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@dataclass
class PageAnalysis:
    """A straightened (or original) page plus suggested horizontal split lines."""

    image: np.ndarray
    lines: list[int]  # suggested split positions in image pixels, ascending
    page_detected: bool

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _warp_to(image: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = ordered
    target_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    target_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (target_width, target_height))


def _detect_page(image: np.ndarray) -> np.ndarray | None:
    """Best-effort paper quadrilateral. Returns None when unsure."""
    height, width = image.shape[:2]
    scale = min(1.0, 1800 / max(height, width))
    preview = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = preview.shape[0] * preview.shape[1] * 0.35
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        if cv2.contourArea(contour) < minimum_area:
            continue
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(polygon) == 4:
            return polygon.reshape(4, 2) / scale

    # Curled pages do not approximate to four points; fall back to the largest
    # bright, low-saturation region. 亮度取照片自己的高分区位，暗光照片的
    # 纸面亮度远低于 105 这种固定值，固定阈值会直接找不到纸。
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    v_floor = max(70.0, float(np.percentile(value, 90)) * 0.55)
    paper_mask = cv2.inRange(hsv, (0, 0, v_floor), (180, 110, 255))
    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)),
        iterations=2,
    )
    bright_contours, _ = cv2.findContours(
        paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not bright_contours:
        return None
    contour = max(bright_contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < minimum_area:
        return None
    return cv2.boxPoints(cv2.minAreaRect(contour)) / scale


def straighten(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Straighten a photographed page. Falls back to the original on any doubt."""
    height, width = image.shape[:2]
    page = _detect_page(image)
    if page is None:
        return image, False
    ordered = _order_points(page)
    page_area = abs(cv2.contourArea(np.asarray(ordered, dtype="float32")))
    # A contour covering virtually the whole frame is the photo edge or a bright
    # background, not paper. Treating it as a page would crop real content away.
    if page_area >= width * height * 0.94:
        return image, False
    target_width = int(
        max(np.linalg.norm(ordered[2] - ordered[3]), np.linalg.norm(ordered[1] - ordered[0]))
    )
    target_height = int(
        max(np.linalg.norm(ordered[1] - ordered[2]), np.linalg.norm(ordered[0] - ordered[3]))
    )
    if target_width < 300 or target_height < 300:
        return image, False
    return _warp_to(image, ordered), True


def straightening_corners(image: np.ndarray) -> list[list[float]]:
    """Detected paper corners for the manual four-point helper, or [] if unsure."""
    page = _detect_page(image)
    return [] if page is None else _order_points(page).tolist()


def warp_with_corners(image: np.ndarray, points: list[list[float]]) -> np.ndarray:
    """Warp using four user-picked corners. Raises only on genuinely bad input."""
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("必须选择四个纸张角点")
    height, width = image.shape[:2]
    page = np.asarray(points, dtype="float32")
    if not np.isfinite(page).all():
        raise ValueError("纸张角点无效")
    if (
        (page[:, 0] < 0).any()
        or (page[:, 0] >= width).any()
        or (page[:, 1] < 0).any()
        or (page[:, 1] >= height).any()
    ):
        raise ValueError("纸张角点超出照片范围")
    ordered = _order_points(page)
    if abs(cv2.contourArea(ordered)) < width * height * 0.08:
        raise ValueError("选择的纸张区域太小，请重新选择四个角")
    target_width = int(
        max(np.linalg.norm(ordered[2] - ordered[3]), np.linalg.norm(ordered[1] - ordered[0]))
    )
    target_height = int(
        max(np.linalg.norm(ordered[1] - ordered[2]), np.linalg.norm(ordered[0] - ordered[3]))
    )
    if target_width < 300 or target_height < 300:
        raise ValueError("选择的纸张区域太小，请重新选择四个角")
    return _warp_to(image, ordered)


def _page_mask(image: np.ndarray) -> np.ndarray:
    """Approximate the paper area so the surrounding desk does not pollute profiles.

    亮度阈值跟着照片自身的亮度走：暗光下的照片纸面可能只有 100 出头，
    固定阈值会把整帧判成"不是纸"，退化成整帧参与统计，黑色背景直接
    灌满墨水轮廓，一条切线都出不来。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright_ref = float(np.percentile(gray, 90))
    floor = max(90.0, bright_ref - 70.0)
    mask = ((gray > floor) & (hsv[:, :, 1] < 80)).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    )
    if mask.mean() < 0.12:  # 还是不够亮：按 Otsu 把"较亮的那一类"当纸再试一次
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = ((otsu > 0) & (hsv[:, :, 1] < 110)).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        )
    if mask.mean() < 0.12:  # almost nothing bright left; treat the whole frame as page
        return np.ones(image.shape[:2], dtype=bool)
    return mask > 0


def _ink_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dark pixels inside the paper area, with the teacher's red pen excluded.

    红叉、红圈、红波浪线经常横在题间空隙上；红笔在灰度图里也偏暗，
    不排除的话真实空白带会被填满。孩子的铅笔字是灰色的，照常算墨水。

    "暗"也是相对纸面说的：暗光照片里纸面本身只有 100 出头，固定阈值
    会把整页纸都判成墨水。按行取纸面亮度的中位数做参照，顺带把页面
    一侧的阴影也归一化掉。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (hsv[:, :, 1] > 60) & ((hsv[:, :, 0] < 20) | (hsv[:, :, 0] > 160))
    paper_row = np.where(mask, gray, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 整行无纸的行是全 NaN
        with np.errstate(all="ignore"):
            reference = np.nanmedian(paper_row, axis=1)
    reference = np.where(np.isnan(reference), 255.0, reference)
    ink_floor = np.clip(0.72 * reference, 45.0, 165.0)
    return (gray < ink_floor[:, None]) & mask & ~red


def _row_ink_profile(image: np.ndarray, mask: np.ndarray, ink: np.ndarray | None = None) -> np.ndarray:
    """Per-row fraction of the paper width covered by ink.

    按覆盖率而不是像素个数统计：孩子把答案挤在题间空隙时通常只占行宽
    的一小截，数像素会超过阈值，看覆盖率就知道这一行大半还是空的。
    横向先膨胀一次，把同一个字的笔画连起来再算，免得一行字被拆散。
    没有纸的行（桌面）视为非空白，免得纸外区域冒出假的空白带。
    """
    if ink is None:
        ink = _ink_mask(image, mask)
    ink = ink.astype(np.uint8)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    covered = ink.sum(axis=1).astype(np.float32)
    paper = mask.sum(axis=1).astype(np.float32)
    coverage = np.where(paper > 0, covered / np.maximum(paper, 1), 1.0)
    return cv2.GaussianBlur(coverage.reshape(-1, 1), (1, 0), 3).ravel()


def _row_left_edge(ink: np.ndarray, x_min: int, x_max: int) -> np.ndarray:
    """每行最左墨水的 x 位置（只在 [x_min, x_max] 里找），没有墨水则 NaN。

    题号行（"5.""6."）顶格起步，选项、换行、图都缩进——所以"一条空隙
    下方的内容比上方更靠左"就是新题开始的几何特征，不需要认字。
    左边界卡掉装订侧的手写列，右边界保证只看版面左侧。
    """
    window = ink[:, x_min:x_max]
    has_ink = window.any(axis=1)
    first = window.argmax(axis=1).astype(np.float32) + x_min
    return np.where(has_ink, first, np.nan)


def _block_left_edge(lefts: np.ndarray, top: int, bottom: int) -> float:
    """窗口内若干文字行左缘的低分位数；个别污染行（铅笔道、红笔残留）带偏不了。"""
    segment = lefts[max(0, top):bottom]
    segment = segment[~np.isnan(segment)]
    if segment.size < 3:
        return np.nan
    return float(np.percentile(segment, 20))


def _gap_bands(profile: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Maximal runs of rows whose ink count sits below the threshold."""
    blank = profile < threshold
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, is_blank in enumerate(blank):
        if is_blank and start is None:
            start = y
        elif not is_blank and start is not None:
            bands.append((start, y - 1))
            start = None
    if start is not None:
        bands.append((start, len(blank) - 1))
    return bands


# 一行纸面被墨水覆盖的比例低于这个值就算空白行
BLANK_COVERAGE = 0.12


def _merge_across_thin_strokes(
    bands: list[tuple[int, int]], max_sliver: int
) -> list[tuple[int, int]]:
    """把只隔着一两笔细字迹的空白带并成一条。

    手写答案偶尔有一笔斜着划过空隙：两侧其实都是空白，中间那几行墨水
    只是薄薄一层。不合并的话一个完整空隙会被拆成两条不够宽的碎带。
    """
    if not bands:
        return bands
    merged = [bands[0]]
    for top, bottom in bands[1:]:
        sliver = top - merged[-1][1] - 1  # 两条空白带之间的墨水行数
        if 0 < sliver <= max_sliver:
            merged[-1] = (merged[-1][0], bottom)
        else:
            merged.append((top, bottom))
    return merged


def suggest_split_lines(image: np.ndarray | Path | str) -> list[int]:
    """Suggest horizontal cuts at the widest blank gaps between text bands.

    Gap width separates a paragraph break (a question boundary) from ordinary
    leading, so the widest gaps are the best first guess. A gap only counts when
    there is ink both above and below it, which keeps the page margins from
    turning into pointless cuts. Suggestions are only a starting point: the user
    adjusts them by dragging.
    """
    if not isinstance(image, np.ndarray):
        image = load_image(image)
    height, width = image.shape[:2]
    mask = _page_mask(image)
    ink = _ink_mask(image, mask)
    profile = _row_ink_profile(image, mask, ink)
    if float(profile.max()) <= 0:
        return []

    threshold = BLANK_COVERAGE
    gaps = _gap_bands(profile, threshold)
    if not gaps:
        return []

    max_sliver = max(8, int(height * 0.004))
    gaps = _merge_across_thin_strokes(gaps, max_sliver)

    min_gap = max(6, int(height * 0.006))
    meaningful = [band for band in gaps if (band[1] - band[0] + 1) >= min_gap]
    if not meaningful:
        return []

    # 有没有墨水看"是否存在"，不看覆盖率：纸外行在轮廓里是 1.0，
    # 拿它判断会把页边空白当成"下方还有内容"。
    row_has_ink = ink.sum(axis=1) > width * 0.004

    def has_ink_above(top: int) -> bool:
        return bool(row_has_ink[:top].any())

    def has_ink_below(bottom: int) -> bool:
        return bool(row_has_ink[bottom + 1:].any())

    def band_width(band: tuple[int, int]) -> int:
        return band[1] - band[0] + 1

    # A page margin is also a wide blank band; only real breaks qualify.
    # 先把页边和贴边太近的带滤掉再统计 leading，不然页边空白会抬高基准。
    candidates = [
        band
        for band in meaningful
        if has_ink_above(band[0])
        and has_ink_below(band[1])
        and height * 0.02 <= (band[0] + band[1]) // 2 <= height * 0.98
    ]
    if not candidates:
        return []

    # 下方内容是否比上方更靠左（顶格）：题号新起一行的几何特征。
    x_min, x_max = int(width * 0.10), int(width * 0.55)
    lefts = _row_left_edge(ink, x_min, x_max)
    span = max(40, int(height * 0.035))
    pad = min_gap
    shift = width * 0.01

    def starts_new_block(band: tuple[int, int]) -> bool:
        top, bottom = band
        above = _block_left_edge(lefts, top - span, top - pad)
        below = _block_left_edge(lefts, bottom + pad, bottom + span)
        if np.isnan(above) or np.isnan(below):
            return False
        return below <= above - shift

    aligned = [band for band in candidates if starts_new_block(band)]

    # Leading is the smallest common gap; anything notably taller is a break.
    leading = float(np.median([band_width(band) for band in candidates]))
    cutoff = max(leading * 1.9, min_gap * 2.0)
    confident = [band for band in candidates if band_width(band) >= cutoff]
    chosen = list(dict.fromkeys(aligned + confident))
    if len(chosen) < 3:
        # 页面排得密、题间空隙跟行距拉不开时，光靠宽度会漏掉大半边界。
        # 宁可多切几条让用户删（一次点击），也不漏掉让用户对着大块图补线；
        # 但只补"明显比普通行距宽一点"的候选，题目本就少的页面不硬凑数。
        topup_floor = max(leading * 1.1, min_gap * 2.0)
        ranked = sorted(
            (band for band in candidates if band_width(band) >= topup_floor),
            key=band_width,
            reverse=True,
        )
        for band in ranked:
            if len(chosen) >= 3:
                break
            if band not in chosen:
                chosen.append(band)
    chosen.sort()

    lines: list[int] = []
    for top, bottom in chosen:
        center = (top + bottom) // 2
        # Skip cuts that would land on top of each other.
        if lines and center - lines[-1] < min_gap:
            continue
        lines.append(center)
    return lines


def crop_by_lines(image: np.ndarray, lines: list[int]) -> list[np.ndarray]:
    """Split an image at the given y positions. Empty lines yield the whole image."""
    height = image.shape[0]
    edges = [0, *[line for line in sorted(lines) if 0 < line < height], height]
    return [image[top:bottom] for top, bottom in zip(edges, edges[1:]) if bottom > top]


def analyze(source: Path | str) -> PageAnalysis:
    """Straighten a photo and suggest split lines. Never raises on image content."""
    image = load_image(source)
    corrected, detected = straighten(image)
    return PageAnalysis(corrected, suggest_split_lines(corrected), detected)