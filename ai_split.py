"""AI 理解层：让视觉大模型指出每道题从哪一行开始，OCR 负责像素定位。

OCR 只认识"长得像题号的行"，它不知道一行字是不是新题的开始——
行距宽松的卷子把题干和自己的图切开、手写填满的题缝里找边界，都是
理解问题。这里让视觉大模型（豆包 Seed，火山方舟 OpenAI 兼容接口）
列出"每道题第一行的题号 + 开头几个字"，再拿 OCR 的行首文本去匹配
定位。大模型直接给像素坐标不可靠，所以语义归 AI、坐标归 OCR。

合并规则：
- AI 失败（没配 key、网络、解析不出）→ 完全退回"几何 + OCR 题号"的
  并集行为，绝不阻塞上传；
- AI 成功 → 切线完全交给 AI/OCR 圈定的题号行，几何线全部丢弃；
  页首到第一道题之间是页眉（校名、单元号），调用方应丢弃这一段，
  不保存成题——所以 N 道题正好 N 张卡。用 split_plan() 拿 AI 参与标记。

凭证：优先环境变量 ARK_API_KEY，其次项目根的 api_key.txt（一行）。
模型/网关可用 ARK_MODEL / ARK_BASE_URL 覆盖。
"""

import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np

import ocr_split

# Agent Plan 的 key 走 /api/plan/v3，平台 key 走 /api/v3；按 key 类型用环境变量覆盖
ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
ARK_MODEL = os.environ.get("ARK_MODEL", "doubao-seed-2-1-pro-260915")
AI_TIMEOUT = float(os.environ.get("AI_TIMEOUT", "60"))

PROMPT = (
    "这是小学数学作业页的照片。请找出页面上每道题第一行的开头文字，"
    "包括：大题（1. 2. 3.）、大题下的小题（(1) (2)）、以中文数字编号且紧跟题目内容的独立题，"
    "以及没有编号但独立成题的行。页眉、学校名、页脚、分数、选项行（A. B. C.）不算。"
    "只输出 JSON 数组，按从上到下顺序，每项形如 "
    '{"marker": "题号原文", "text": "该行开头4到8个字，照抄照片原文，不要改写", '
    '"section": 该行是否为章节标题（如「一、精心选择」「二、认真填写」这类只有题型说明、'
    "下面还有 1. 2. 小题的行；是填 true，不是填 false)}。"
    "不要输出任何其他内容。"
)


def _api_key() -> str | None:
    key = os.environ.get("ARK_API_KEY", "").strip()
    if key:
        return key
    try:
        key = (Path(__file__).parent / "api_key.txt").read_text(encoding="utf-8").strip()
        return key or None
    except OSError:
        return None


def _normalize(text: str) -> str:
    """只留字母数字汉字：OCR/模型对标点和全半角各执一词，比内容比字。"""
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text or "")


def _parse_markers(content: str) -> list[str] | None:
    """从模型回复里抠出 JSON 数组，容忍 ```json 包裹和废话。"""
    match = re.search(r"\[.*\]", content or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    markers = []
    for item in data:
        if not isinstance(item, dict):
            if isinstance(item, str):
                markers.append(item)
            continue
        # 章节标题（"一、精心选择"这类）不是题界：留着只会切出碎片卡。
        # 靠模型显式标注，比提示词"别输出它"可靠——模型偶尔不听话。
        if item.get("section") is True:
            continue
        marker = str(item.get("marker", "")).strip()
        text = str(item.get("text", "")).strip()
        # 有的模型把题号在两个字段里各写一遍（"一、" + "一、精心选择。"），
        # 互相包含时取长的那个，不然拼出 "一、一、…" 会破坏前缀匹配
        if text and marker and (
            _normalize(text).startswith(_normalize(marker))
            or _normalize(marker).startswith(_normalize(text))
        ):
            markers.append(text if len(text) >= len(marker) else marker)
        else:
            markers.append(marker + text)
    return markers or None


def ai_markers(image: np.ndarray) -> list[str] | None:
    """问 AI 每道题第一行的"题号+开头几个字"；失败返回 None（走纯几何降级）。"""
    key = _api_key()
    if key is None or image is None or image.size == 0:
        return None
    try:
        height, width = image.shape[:2]
        scale = min(1.0, 1400 / max(height, width))
        small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        ok, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        image_uri = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
        response = httpx.post(
            f"{ARK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": ARK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 2000,
                # 关掉深度思考：这个任务只是"读出题号行"，想得越久等得越久
                # （实测 64s → 8s），准确度没有差别。不支持该参数的模型会
                # 报错，走降级路径。
                "thinking": {"type": "disabled"},
            },
            timeout=AI_TIMEOUT,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_markers(content)
    except Exception:
        return None


def _match_markers(
    strips: list[tuple[int, int, str, float]], markers: list[str]
) -> list[tuple[int, int]]:
    """把 AI 说的每行匹配到 OCR 检测框上，返回按 y 升序的题界位置。

    从上往下逐个匹配且只往后找：AI 的顺序本身就是页面顺序，能防止
    "1.3.66" 错配到别的行。识别有噪（"2.（北京"→"2（北"），比归一化
    后的公共前缀，≥2 个字算命中；首位数字对上而后面被误读时放宽到 1。
    """
    starts: list[tuple[int, int]] = []
    floor = -1
    for marker in markers:
        key = _normalize(marker)
        if len(key) < 2:
            continue
        best, best_prefix = None, 0
        for y, x, text, _score in strips:
            if y <= floor:
                continue
            candidate = _normalize(text)
            if not candidate:
                continue
            prefix = 0
            for a, b in zip(key, candidate):
                if a != b:
                    break
                prefix += 1
            if prefix > best_prefix:
                best, best_prefix = (y, x), prefix
        matched = None
        if best is not None and best_prefix >= 2:
            matched = best
        elif best is not None and best_prefix == 1 and key[0].isdigit():
            matched = best  # 题号数字对上了，后半被 OCR 误读
        if matched is not None:
            starts.append(matched)
            floor = matched[0]
    return starts


def _dedup_starts(starts: list[tuple[int, int]], min_gap: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for y, x in sorted(starts):
        if result and y - result[-1][0] < min_gap:
            continue
        result.append((y, x))
    return result


def _dedup_lines(lines: list[int], min_gap: int) -> list[int]:
    result: list[int] = []
    for line in sorted(lines):
        if result and line - result[-1] < min_gap:
            continue
        result.append(line)
    return result


def split_plan(image: np.ndarray, geometric_lines: list[int]) -> tuple[list[int], bool]:
    """AI 语义 + OCR 定位 + 几何空白带的最终合并：返回 (切线, AI 是否参与)。

    OCR 和 AI 相互独立，并行跑省一半等待。AI 参与时第一条线之上是
    页眉（校名、单元号……），不是题：调用方保存切片时应丢弃第一片，
    N 道题正好 N 张卡。AI 没给出任何可用信息时退回
    ocr_split.merged_lines 的行为，页眉照旧成为第一张卡。
    """
    height, width = image.shape[:2]
    min_gap = max(6, int(height * 0.006))
    with ThreadPoolExecutor(max_workers=2) as pool:
        strips_task = pool.submit(ocr_split.line_strips, image)
        markers_task = pool.submit(ai_markers, image)
        strips = strips_task.result()
        markers = markers_task.result()

    if markers:
        # AI 在场时只信 AI 的题界：章节标题已被提示词排除，若再并上 OCR
        # 的中文数字模式，"一、精心选择"这类章节线会混回来变成碎片卡。
        starts = _dedup_starts(_match_markers(strips, markers), min_gap)
    else:
        starts = sorted(ocr_split.filter_starts(strips, width))
    if not starts:
        return ocr_split.merged_lines(image, geometric_lines), False

    mask = ocr_split._page_mask(image)
    profile = ocr_split._row_ink_profile(image, mask)
    bands = ocr_split._merge_across_thin_strokes(
        ocr_split._gap_bands(profile, ocr_split.BLANK_COVERAGE),
        max(8, int(height * 0.004)),
    )
    lines = []
    for y, _x in starts:
        target = max(0, y - int(height * ocr_split.TOP_MARGIN_RATIO))
        snapped = ocr_split._snap_to_blank_band(bands, target, int(height * 0.02))
        lines.append(snapped if snapped is not None else target)

    if not markers:
        # AI 没参与：几何线照旧全保留（并集），只多出 OCR 题号线
        return _dedup_lines(lines + list(geometric_lines), min_gap), False

    # AI 成功：切线完全交给 AI/OCR 圈定的题号行。AI 已经把每道题的
    # 起始说全了（含无编号独立题），此时几何线能提供的只剩两类——
    # 题干和自己的图之间错切的线、页眉页脚的碎片线，恰好都是用户
    # 要删的边角料，一并丢弃。
    return _dedup_lines(lines, min_gap), True


def merged_lines(image: np.ndarray, geometric_lines: list[int]) -> list[int]:
    """只要切线（兼容入口）。"""
    return split_plan(image, geometric_lines)[0]

