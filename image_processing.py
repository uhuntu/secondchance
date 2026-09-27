"""Automatic worksheet straightening and red-mark based question cropping."""

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


@dataclass
class PageResult:
    corrected_path: Path
    crop_paths: list[Path]
    mark_count: int
    page_detected: bool
    crop_ranges: list[tuple[int, int]]


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


def _warp_page(image: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    scale = min(1.0, 1800 / max(height, width))
    preview = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = preview.shape[0] * preview.shape[1] * 0.35
    page = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        if cv2.contourArea(contour) < minimum_area:
            continue
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(polygon) == 4:
            page = polygon.reshape(4, 2) / scale
            break
    if page is None:
        # Curled pages often do not approximate cleanly to four points. Fall back
        # to the largest bright, low-saturation paper-shaped region.
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
        if bright_contours:
            contour = max(bright_contours, key=cv2.contourArea)
            if cv2.contourArea(contour) >= minimum_area:
                page = cv2.boxPoints(cv2.minAreaRect(contour)) / scale
    if page is None:
        return image, False
    tl, tr, br, bl = _order_points(page)
    page_area = abs(cv2.contourArea(np.array([tl, tr, br, bl], dtype="float32")))
    # A contour covering virtually the whole photo is usually the image frame,
    # not the paper. Treat it as a failed detection so the UI can request four
    # manual corners instead of pretending that the page was straightened.
    if page_area >= width * height * 0.94:
        return image, False
    target_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    target_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if target_width < 300 or target_height < 300:
        return image, False
    destination = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(_order_points(page), destination)
    return cv2.warpPerspective(image, matrix, (target_width, target_height)), True


def _warp_page_from_points(
    image: np.ndarray, points: list[list[float]]
) -> tuple[np.ndarray, bool]:
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
    tl, tr, br, bl = ordered
    target_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    target_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if target_width < 300 or target_height < 300:
        raise ValueError("选择的纸张区域太小，请重新选择四个角")
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
    return cv2.warpPerspective(image, matrix, (target_width, target_height)), True


def _red_mark_centers(image: np.ndarray) -> list[int]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 75, 60), (14, 255, 255)), cv2.inRange(hsv, (165, 75, 60), (180, 255, 255)))
    height, width = red.shape
    joined = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, width // 90), max(7, height // 180))), iterations=2)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(joined)
    centers = []
    minimum_mark_area = max(700, int(width * height * 0.00045))
    minimum_mark_span = max(30, int(width * 0.028))
    minimum_red_pixels = max(80, int(width * height * 0.00005))
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        raw_mark = red[y : y + h, x : x + w]
        raw_y, raw_x = np.nonzero(raw_mark)
        red_pixels = len(raw_x)
        raw_span = (
            max(int(np.ptp(raw_x)), int(np.ptp(raw_y)))
            if red_pixels
            else 0
        )
        # Teacher crosses/circles are prominent marks. Small red ticks, printed
        # decorations and isolated pen strokes should not create questions.
        if (
            area < minimum_mark_area
            or max(w, h) < minimum_mark_span
            or red_pixels < minimum_red_pixels
            or raw_span < minimum_mark_span
        ):
            continue
        if (
            x < width * 0.015
            or x + w > width * 0.985
            or y < height * 0.055
            or y + h > height * 0.96
        ):
            continue
        centers.append(int(centroids[index][1]))
    centers.sort()
    merged = []
    # Multiple strokes/circles on the same answer line belong to one subquestion.
    merge_distance = max(70, int(height * 0.05))
    for center in centers:
        if merged and center - merged[-1] < merge_distance:
            merged[-1] = (merged[-1] + center) // 2
        else:
            merged.append(center)
    return merged


def _crop_ranges(height: int, centers: list[int]) -> list[tuple[int, int]]:
    if not centers:
        return [(0, height)]
    padding = int(height * 0.025)
    boundaries = [0, *((a + b) // 2 for a, b in zip(centers, centers[1:])), height]
    ranges = []
    for index, center in enumerate(centers):
        top = min(max(0, boundaries[index] - padding), max(0, center - int(height * 0.13)))
        bottom = max(min(height, boundaries[index + 1] + padding), min(height, center + int(height * 0.06)))
        if bottom - top >= 100:
            ranges.append((top, bottom))
    return ranges or [(0, height)]


def crop_image_ranges(
    image: np.ndarray, ranges: list[tuple[int, int]]
) -> list[np.ndarray]:
    height, width = image.shape[:2]
    return [image[top:bottom, :width] for top, bottom in ranges]


def process_page(
    source: Path,
    output_dir: Path,
    stem: str,
    page_points: list[list[float]] | None = None,
) -> PageResult:
    """Correct a photographed page and save one crop for each detected red-mark band."""
    with Image.open(source) as pil:
        image = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(pil).convert("RGB")), cv2.COLOR_RGB2BGR)
    corrected, detected = (
        _warp_page_from_points(image, page_points)
        if page_points is not None
        else _warp_page(image)
    )
    centers = _red_mark_centers(corrected)
    ranges = _crop_ranges(corrected.shape[0], centers)
    crops = crop_image_ranges(corrected, ranges)
    corrected_path = output_dir / f"{stem}_page.jpg"
    cv2.imwrite(str(corrected_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, 92])
    crop_paths = []
    for index, crop in enumerate(crops, 1):
        path = output_dir / f"{stem}_q{index}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        crop_paths.append(path)
    return PageResult(corrected_path, crop_paths, len(centers), detected, ranges)
