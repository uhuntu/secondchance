"""Page straightening and suggested split lines.

Design rule: every automatic step is a *suggestion*. A failure must never block
the user, so each helper returns a usable fallback instead of raising. The
frontend turns suggestions into draggable split lines the user can adjust,
delete, or ignore entirely.
"""

import io
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
    # bright, low-saturation region.
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    paper_mask = cv2.inRange(hsv, (0, 0, 105), (180, 105, 255))
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
    """Approximate the paper area so the surrounding desk does not pollute profiles."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((gray > 130) & (hsv[:, :, 1] < 70)).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    )
    if mask.mean() < 0.12:  # almost nothing bright left; treat the whole frame as page
        return np.ones(image.shape[:2], dtype=bool)
    return mask > 0


def _row_ink_profile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row count of dark pixels inside the paper area."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = ((gray < 165) & mask).sum(axis=1).astype(np.float32)
    return cv2.GaussianBlur(ink.reshape(-1, 1), (1, 0), 3).ravel()


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
    profile = _row_ink_profile(image, mask)
    if float(profile.max()) <= 0:
        return []

    threshold = max(3.0, width * 0.02)
    gaps = _gap_bands(profile, threshold)
    if not gaps:
        return []

    min_gap = max(6, int(height * 0.006))
    meaningful = [band for band in gaps if (band[1] - band[0] + 1) >= min_gap]
    if not meaningful:
        return []

    # Leading is the smallest common gap; anything notably taller is a break.
    leading = float(np.median([b[1] - b[0] + 1 for b in meaningful]))
    cutoff = max(leading * 1.9, min_gap * 2.0)
    breaks = [band for band in meaningful if (band[1] - band[0] + 1) >= cutoff]
    if not breaks:
        breaks = meaningful

    def has_ink_above(top: int) -> bool:
        head = profile[:top]
        return bool(head.size and head.max() >= threshold)

    def has_ink_below(bottom: int) -> bool:
        tail = profile[bottom + 1:]
        return bool(tail.size and tail.max() >= threshold)

    lines: list[int] = []
    for top, bottom in breaks:
        # A page margin is also a wide blank band; only real breaks qualify.
        if not has_ink_above(top) or not has_ink_below(bottom):
            continue
        center = (top + bottom) // 2
        if center < height * 0.02 or center > height * 0.98:
            continue
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