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

# 知识点词表：与 static/app.js 的 MATH_KNOWLEDGE_POINTS 保持一致（那边管输入联想）。
# 标签口径统一，AI 填的值才能命中前端联想和关键词搜索。
KP_VOCAB = [
    "观察简单组合体", "三视图辨认",
    "小数乘法", "小数乘整数", "小数乘小数", "积的近似数", "运算定律推广到小数",
    "小数除法", "除数是整数的小数除法", "一个数除以小数",
    "商的近似数", "循环小数", "用估算解决问题",
    "图形的运动", "平移", "旋转", "轴对称",
    "用字母表示数", "数量关系", "等式的性质",
    "多边形的面积", "平行四边形的面积", "三角形的面积",
    "梯形的面积", "组合图形的面积", "公顷和平方千米",
    "密铺", "可能性",
]

PROMPT = (
    "这是小学数学作业页的照片。请找出页面上每道题第一行的开头文字，"
    "包括：大题（1. 2. 3.）、大题下的小题（(1) (2)）、以中文数字编号且紧跟题目内容的独立题，"
    "以及没有编号但独立成题的行。页眉、学校名、页脚、分数、选项行（A. B. C.）不算。"
    "同时判断每道题孩子是否做错：看老师的批改痕迹（红叉、圈出错处、扣分），"
    "有明显错误痕迹填 true，批改是勾或没有痕迹、看不出来填 false。"
    "再给每道题写一个知识点标签：这道题在考什么，不超过10个字，尽量从这些里选："
    + "、".join(KP_VOCAB)
    + "；都不合适就自己概括，看不出考什么填空字符串。"
    "再找出图里所有手写痕迹的位置：孩子用铅笔/钢笔/圆珠笔写的答案、算式、"
    "文字、图形，以及老师用非红色笔写的批注；印刷体（题目、选项、图形、表格线）"
    "不算。每个痕迹给一个包围盒，坐标相对整张图归一化到 0-999，格式 [左, 上, 右, 下]，"
    "框要贴住痕迹本身。"
    '只输出一个 JSON 对象：{"questions": [按从上到下顺序，每项 '
    '{"marker": "题号原文", "text": "该行开头4到8个字，照抄照片原文", '
    '"section": 是否章节标题(true/false), "wrong": 该题是否做错(true/false), '
    '"knowledge": "该题知识点，10字以内，按上面的要求填"}], '
    '"handwriting": [{"box": [左, 上, 右, 下]}]}，不要输出任何其他内容。'
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


def _repair_number_arrays(text: str) -> str:
    """补回模型吞掉的逗号：手写框常被写成 [835 76 954 172] 甚至
    [826 76, 954 175]（成对丢、混着丢），json 解析不了。

    只动"纯数字 + 空白/逗号混合分隔"的数组，别的文本一概不碰。
    """
    return re.sub(
        r"\[\s*-?\d+(?:[\s,]+-?\d+)+\s*\]",
        lambda m: "[" + re.sub(r"[\s,]+", ",", m.group(0)[1:-1].strip()) + "]",
        text,
    )


def _parse_markers(content: str) -> tuple[list[dict], list[list[int]]] | None:
    """从模型回复里抠出题目列表和手写框，容忍 ```json 包裹和废话。

    兼容两种响应：新格式是 {"questions": [...], "handwriting": [{"box": ...}]}，
    旧格式是裸数组（没有手写框，框列表为空）。
    """
    match = re.search(r"\{.*\}|\[.*\]", content or "", re.S)
    if not match:
        return None
    isinstance_markers_object = match.group(0).lstrip().startswith("{")
    raw = _repair_number_arrays(match.group(0))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 回复可能被 max_tokens 截断：丢掉末尾不完整的条目再试一次
        cut = raw.rfind("}")
        if cut <= 1:
            return None
        closer = "]}" if isinstance_markers_object else "]"
        try:
            data = json.loads(raw[:cut + 1] + closer)
        except json.JSONDecodeError:
            return None

    items = data
    hw_boxes: list[list[int]] = []
    if isinstance(data, dict):
        items = data.get("questions")
        for entry in data.get("handwriting") or []:
            if isinstance(entry, dict) and isinstance(entry.get("box"), list):
                box = [int(round(v)) for v in entry["box"] if isinstance(v, (int, float))]
                if len(box) == 4:
                    hw_boxes.append(box)
    if not isinstance(items, list):
        return None
    markers = []
    for item in items:
        if not isinstance(item, dict):
            if isinstance(item, str):
                markers.append({"text": item, "wrong": None, "knowledge": None})
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
            combined = text if len(text) >= len(marker) else marker
        else:
            combined = marker + text
        wrong = item.get("wrong")
        knowledge = item.get("knowledge")
        if not isinstance(knowledge, str) or not knowledge.strip():
            knowledge = None
        markers.append({
            "text": combined,
            "wrong": bool(wrong) if isinstance(wrong, bool) else None,
            "knowledge": knowledge.strip()[:20] if knowledge else None,
        })
    if not markers:
        return None
    return markers, hw_boxes


def ai_markers(image: np.ndarray) -> tuple[list[dict], list[list[int]]] | None:
    """问 AI 每道题第一行的"题号+开头几个字"和手写痕迹包围盒。

    返回 (题号列表, 归一化手写框)；失败返回 None（走纯几何降级）。"""
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
                "max_tokens": 4000,
                # 关掉深度思考：这个任务只是"读出题号行"，想得越久等得越久
                # （实测 64s → 8s），准确度没有差别。不支持该参数的模型会
                # 报错，走降级路径。
                "thinking": {"type": "disabled"},
            },
            timeout=AI_TIMEOUT,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _parse_markers(content)
        if parsed is None:
            return None
        return parsed
    except Exception:
        return None


def _match_markers(
    strips: list[tuple[int, int, str, float]], markers: list[dict]
) -> list[tuple[int, int, bool | None, str | None]]:
    """把 AI 说的每行匹配到 OCR 检测框上，返回按 y 升序的 (题界, 对错, 知识点)。

    从上往下逐个匹配且只往后找：AI 的顺序本身就是页面顺序，能防止
    "1.3.66" 错配到别的行。识别有噪（"2.（北京"→"2（北"），比归一化
    后的公共前缀，≥2 个字算命中；首位数字对上而后面被误读时放宽到 1。
    """
    starts: list[tuple[int, int, bool | None, str | None]] = []
    floor = -1
    for marker in markers:
        key = _normalize(marker.get("text", ""))
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
            starts.append(
                (matched[0], matched[1], marker.get("wrong"), marker.get("knowledge"))
            )
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


def split_plan(
    image: np.ndarray, geometric_lines: list[int]
) -> tuple[list[int], bool, list[bool | None], list[list[int]], list[str | None]]:
    """AI 语义 + OCR 定位 + 几何空白带的最终合并。

    返回 (切线, AI 是否参与, 每条线对应题的对错判断, 手写痕迹框, 每题知识点)。
    手写框是整图像素坐标 [左, 上, 右, 下]，与切线无关（AI 对整页的
    标注）；调用方把它按题切分后交给去手写用。

    OCR 和 AI 相互独立，并行跑省一半等待。AI 参与时第一条线之上是
    页眉（校名、单元号……），不是题：调用方保存切片时应丢弃第一片，
    N 道题正好 N 张卡；对错判断、知识点与切线一一对齐（第 i 条线 = 第 i 题）。
    AI 没给出任何可用信息时退回 ocr_split.merged_lines 的行为，
    对错/知识点列表和手写框为空，页眉照旧成为第一张卡。
    """
    height, width = image.shape[:2]
    min_gap = max(6, int(height * 0.006))
    with ThreadPoolExecutor(max_workers=2) as pool:
        strips_task = pool.submit(ocr_split.line_strips, image)
        markers_task = pool.submit(ai_markers, image)
        strips = strips_task.result()
        markers, hw_boxes_norm = markers_task.result() or ([], [])

    wrong_flags: list[bool | None] = []
    knowledge_flags: list[str | None] = []
    hw_boxes_px: list[list[int]] = []
    if markers:
        # AI 在场时只信 AI 的题界：章节标题已被提示词排除，若再并上 OCR
        # 的中文数字模式，"一、精心选择"这类章节线会混回来变成碎片卡。
        matched = _match_markers(strips, markers)
        starts = _dedup_starts([(y, x) for y, x, _w, _k in matched], min_gap)
        # 对错判断、知识点跟着各自的题界走；去重丢掉的行不带走
        kept = set(starts)
        wrong_flags = [w for y, x, w, _k in matched if (y, x) in kept]
        knowledge_flags = [k for y, x, _w, k in matched if (y, x) in kept]
        # 归一化手写框换算成整图像素坐标
        hw_boxes_px = [
            [
                int(round(v / 999 * (width if i % 2 == 0 else height)))
                for i, v in enumerate(box)
            ]
            for box in hw_boxes_norm
        ]
    else:
        starts = sorted(ocr_split.filter_starts(strips, width))
    if not starts:
        return ocr_split.merged_lines(image, geometric_lines), False, [], [], []

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
        return _dedup_lines(lines + list(geometric_lines), min_gap), False, [], [], []

    # AI 成功：切线完全交给 AI/OCR 圈定的题号行。AI 已经把每道题的
    # 起始说全了（含无编号独立题），此时几何线能提供的只剩两类——
    # 题干和自己的图之间错切的线、页眉页脚的碎片线，恰好都是用户
    # 要删的边角料，一并丢弃。
    return _dedup_lines(lines, min_gap), True, wrong_flags, hw_boxes_px, knowledge_flags


def merged_lines(image: np.ndarray, geometric_lines: list[int]) -> list[int]:
    """只要切线（兼容入口）。"""
    return split_plan(image, geometric_lines)[0]


def boxes_per_bands(
    boxes: list[list[int]], bands: list[tuple[int, int]]
) -> list[list[list[int]]]:
    """把整页坐标的手写框按每张切片在整页里的范围分下去，转成切片内坐标。

    bands[i] = (顶, 底) 是第 i 张切片覆盖的整页 y 范围；跨界的框会被
    裁剪进相邻两片（两片都会清框内的笔迹）。
    """
    per_crop: list[list[list[int]]] = [[] for _ in bands]
    for x1, y1, x2, y2 in boxes:
        for index, (top, bottom) in enumerate(bands):
            if y2 <= top or y1 >= bottom:
                continue
            per_crop[index].append(
                [x1, max(y1, top) - top, x2, min(y2, bottom) - top]
            )
    return per_crop


def boxes_per_crop(
    boxes: list[list[int]], lines: list[int], height: int, skip_first: bool
) -> list[list[list[int]]]:
    """把整页坐标的手写框按切线分到各张切片，转换成切片内坐标。

    第 i 张切片覆盖 [bounds[i], bounds[i+1]]；跨界的框会被裁剪进
    相邻两片（两片都会清框内的笔迹）。skip_first 时第一片（页眉）
    已被丢弃，框随切片编号整体前移。
    """
    bounds = [0, *lines, height]
    crops = list(zip(bounds, bounds[1:]))
    if skip_first and crops:
        crops = crops[1:]
    return boxes_per_bands(boxes, crops)

