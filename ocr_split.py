"""OCR 题号定位：把"顶格的题号行"变成切线建议，与几何空白带取并集。

行距宽松的卷子上，题间空隙和普通行距一样宽，纯几何方法分不开；
但题号永远顶格出现在左边距，这个信号不依赖空隙宽度。RapidOCR
（PP-OCRv4 的 onnxruntime CPU 版）只用两样东西：

- 文本检测框：每一行文字的位置（只跑左侧区域，题号不会在页中间）；
- 行首一小条的识别文本：判断是不是题号（"5." "2（北京" "三、"），
  整行内容不关心，识别只做开头几个字，省一大截时间。

细节：
- 红笔先抹白再进 OCR，红杠划过题号行会破坏检测和识别；
- OCR 会吞掉数字后的小数点（"2.（北京"→"2（北京"），模式里对
  "数字+中文/括号/大写字母"也放行；
- 题号行必须顶格：以识别候选的左缘中位数为基准，偏右的不是题号
  （虚线框里的算式、竖式、"1.5m" 这类由此排除）；
- 任何失败（没装依赖、模型挂了、图不对）都返回空，切题退回纯
  几何，绝不阻塞上传。
"""

import re

import cv2
import numpy as np

from image_processing import (
    BLANK_COVERAGE,
    _gap_bands,
    _merge_across_thin_strokes,
    _page_mask,
    _row_ink_profile,
)

# 题号的三种长相：数字加点、数字直接接正文（点被 OCR 吃掉）、中文数字加顿号
QUESTION_PREFIX = re.compile(
    r"^\s*(?:"
    r"\d{1,2}\s*[.、．]"                              # 5. 5、5．
    r"|\d{1,2}(?=[（(【\[A-Z一-龥])"                  # 2（北京 4AX
    r"|[一二三四五六七八九十]{1,3}\s*[、.．]"          # 一、 三、
    r"|[一二三四五六七八九十]{1,3}[一-龥]"            # 一精（顿号被 OCR 吞掉）
    r")"
)

OCR_MAX_SIDE = 1400      # OCR 在缩略图上跑，位置按比例还原
LEFT_COLUMN = 0.40       # 题号只出现在左侧，检测框限制在左 40%
TOP_MARGIN_RATIO = 0.008  # 切线放在题号行上方一点点
_engine = None


def _get_engine():
    """懒加载 RapidOCR 单例；没装依赖返回 None，调用方走纯几何。"""
    global _engine
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR(det_limit_side_len=960, det_limit_type="max", rec_batch_num=16)
        except Exception:
            _engine = False
    return None if _engine is False else _engine


def _without_red_pen(image: np.ndarray) -> np.ndarray:
    """红笔抹白：和切题的红笔排除同一套色域，防红杠划断题号行。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (hsv[:, :, 1] > 60) & ((hsv[:, :, 0] < 20) | (hsv[:, :, 0] > 160))
    clean = image.copy()
    clean[red] = (255, 255, 255)
    return clean


def line_strips(image: np.ndarray) -> list[tuple[int, int, str, float]]:
    """左侧区域每一行文字：[(y, x, 行首文本, 置信度)]；任何失败返回 []。

    只识别行首一小条（题号必然在行首，整行内容不关心），比整行
    识别快一大截。AI 层靠它把"题号+开头几个字"匹配到像素位置。
    """
    engine = _get_engine()
    if engine is None or image is None or image.size == 0:
        return []
    try:
        height, width = image.shape[:2]
        scale = min(1.0, OCR_MAX_SIDE / max(height, width))
        small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        small = _without_red_pen(small)
        sh, sw = small.shape[:2]

        boxes, _ = engine(small[:, : int(sw * LEFT_COLUMN)], use_det=True, use_cls=False, use_rec=False)
        if not boxes:
            return []

        strip_w = max(56, int(sw * 0.06))
        crops, positions = [], []
        for box in boxes:
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            x0, y0 = min(xs), min(ys)
            crop = small[
                max(0, int(y0) - 2): int(max(ys)) + 2,
                max(0, int(x0) - 2): min(sw, int(x0) + strip_w),
            ]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            crops.append(crop)
            positions.append((x0, y0))
        if not crops:
            return []
        texts, _ = engine.text_rec(crops)
        return [
            (int(y0 / scale), int(x0 / scale), text, float(score))
            for (text, score), (x0, y0) in zip(texts, positions)
        ]
    except Exception:
        return []


def filter_starts(
    strips: list[tuple[int, int, str, float]], width: int
) -> list[tuple[int, int]]:
    """从行首文本里挑题号行：题号模式 + 顶格校验（width 为整页像素宽）。"""
    candidates = [
        (y, x)
        for y, x, text, score in strips
        if score >= 0.5 and QUESTION_PREFIX.match(text or "")
    ]
    if not candidates:
        return []
    # 题号行的左缘高度一致：以候选的中位数为基准，明显偏右的不是题号
    # （缩进的选项/算式），偏左的也不是（装订边的手写、印章）。
    margin = float(np.median([x for _y, x in candidates]))
    return [(y, x) for y, x in candidates if margin - width * 0.04 <= x <= margin + width * 0.02]


def question_starts(image: np.ndarray) -> list[tuple[int, int]]:
    """疑似题号行的 (y, x) 位置，按 y 升序；失败返回空列表。"""
    try:
        width = int(image.shape[1])
        return sorted(filter_starts(line_strips(image), width))
    except Exception:
        return []


def _snap_to_blank_band(bands: list[tuple[int, int]], y: int, tolerance: int) -> int | None:
    """题号上方的空白带中心是最稳的落点；附近没有空白带才用估算位置。"""
    best, best_distance = None, tolerance
    for top, bottom in bands:
        center = (top + bottom) // 2
        distance = abs(center - y)
        if distance <= best_distance:
            best, best_distance = center, distance
    return best


def merged_lines(image: np.ndarray, geometric_lines: list[int]) -> list[int]:
    """OCR 题号线 + 几何建议线取并集，去重后返回。

    OCR 只增不删：它负责几何漏掉的真实题界（行距宽松的卷子、红笔
    干扰、手写挤占空隙都拦不住它），几何线继续兜底 OCR 认不出的
    页面。落点优先吸附到题号上方的空白带中心，避免切穿上一题。
    """
    lines = list(geometric_lines)
    starts = question_starts(image)
    if starts:
        height = image.shape[0]
        mask = _page_mask(image)
        profile = _row_ink_profile(image, mask)
        bands = _gap_bands(profile, BLANK_COVERAGE)
        bands = _merge_across_thin_strokes(bands, max(8, int(height * 0.004)))
        min_gap = max(6, int(height * 0.006))
        for y, _x in sorted(starts):
            target = max(0, y - int(height * TOP_MARGIN_RATIO))
            snapped = _snap_to_blank_band(bands, target, int(height * 0.02))
            lines.append(snapped if snapped is not None else target)
    lines.sort()
    if not lines:
        return []
    min_gap = max(6, int(image.shape[0] * 0.006))
    deduped = [lines[0]]
    for line in lines[1:]:
        if line - deduped[-1] >= min_gap:
            deduped.append(line)
    return deduped
