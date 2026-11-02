"""AI 理解层：让视觉大模型指出每道题从哪一行开始，OCR 负责像素定位。

OCR 只认识"长得像题号的行"，它不知道一行字是不是新题的开始——
行距宽松的卷子把题干和自己的图切开、手写填满的题缝里找边界，都是
理解问题。这里让视觉大模型（豆包 Seed，火山方舟 OpenAI 兼容接口）
列出"每道题第一行的题号 + 开头几个字"，再拿 OCR 的行首文本去匹配
定位。大模型直接给像素坐标不可靠，所以语义归 AI、坐标归 OCR。

合并规则：
- AI 失败（没配 key、网络、解析不出）→ 完全退回"几何 + OCR 题号"的
  并集行为，绝不阻塞上传；
- AI 成功 → AI 题号线 + 几何线取并集（AI 常漏报，几何线补覆盖），
  AI 额外带来对错判断和知识点；页首到第一道题之间是页眉（校名、
  单元号……），调用方应丢弃这一段，所以 N 道题至少 N 张卡。
  用 split_plan() 拿 AI 参与标记。

凭证：优先环境变量 ARK_API_KEY，其次项目根的 api_key.txt（一行）。
模型/网关可用 ARK_MODEL / ARK_BASE_URL 覆盖。
"""

import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import httpx
import numpy as np

import ocr_split

# Agent Plan 的 key 走 /api/plan/v3，Coding Plan 走 /api/coding/v3，平台 key 走 /api/v3；
# 按 key 类型用环境变量覆盖
ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
ARK_MODEL = os.environ.get("ARK_MODEL", "doubao-seed-2-1-pro-260915")
AI_TIMEOUT = float(os.environ.get("AI_TIMEOUT", "120"))
# 模型输出方差很大（同一张图 5~39 个标记），单次调用常漏掉一半的题。
# 并行多调几次取匹配并集，不增加等待时间，只多花点配额。
AI_ATTEMPTS = max(1, int(os.environ.get("AI_ATTEMPTS", "2")))

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
    "再找出所有填空题的空位：成对的印刷小括号（ ）之间的区域、填空横线上方"
    "的书写区域。空位框要罩住整个空（包括孩子写进里面的答案），但不要包含"
    "括号或横线本身；坐标同样归一化到 0-999。没有填空题就给空数组。"
    '只输出一个 JSON 对象：{"questions": [按从上到下顺序，每项 '
    '{"marker": "题号原文", "text": "该行开头4到8个字，照抄照片原文", '
    '"section": 是否章节标题(true/false), "wrong": 该题是否做错(true/false), '
    '"knowledge": "该题知识点，10字以内，按上面的要求填"}], '
    '"handwriting": [{"box": [左, 上, 右, 下]}], '
    '"blanks": [{"box": [左, 上, 右, 下]}]}，不要输出任何其他内容。'
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


def _extract_questions_array(raw: str) -> dict | None:
    """从写坏的 JSON 里按括号配平抠出 questions 数组，尽力而为。"""
    key = raw.find('"questions"')
    if key < 0:
        return None
    start = raw.find("[", key)
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    items = json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return {"questions": items} if isinstance(items, list) else None
    return None


def _parse_markers(
    content: str,
) -> tuple[list[dict], list[list[int]], list[list[int]]] | None:
    """从模型回复里抠出题目列表、手写框和填空空位，容忍 ```json 包裹和废话。

    兼容两种响应：新格式是 {"questions": [...], "handwriting": [...],
    "blanks": [...]}，旧格式是裸数组（没有手写框/空位，两个列表为空）。
    """
    match = re.search(r"\{.*\}|\[.*\]", content or "", re.S)
    if not match:
        return None
    isinstance_markers_object = match.group(0).lstrip().startswith("{")
    raw = _repair_number_arrays(match.group(0))
    # 模型偶尔在数组后多打一个引号（"... [456 126 486 151]"}），
    # JSON 报废。] 和 ,/} 之间的游离引号直接删掉。
    raw = re.sub(r'\]"\s*([,\]}])', r"]\1", raw)
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 回复可能被 max_tokens 截断：丢掉末尾不完整的条目再试一次
        cut = raw.rfind("}")
        if cut > 1:
            closer = "]}" if isinstance_markers_object else "]"
            try:
                data = json.loads(raw[:cut + 1] + closer)
            except json.JSONDecodeError:
                pass
        if data is None and isinstance_markers_object:
            # 还不行就只救 questions 数组：题号/错题/知识点在前面，
            # 别让排后面的手写框写坏了把整份 AI 成果拖下水。
            data = _extract_questions_array(raw)

    items = data
    hw_boxes: list[list[int]] = []
    blanks: list[list[int]] = []
    if isinstance(data, dict):
        items = data.get("questions")
        for entry in data.get("handwriting") or []:
            if isinstance(entry, dict) and isinstance(entry.get("box"), list):
                box = [int(round(v)) for v in entry["box"] if isinstance(v, (int, float))]
                if len(box) == 4:
                    hw_boxes.append(box)
        for entry in data.get("blanks") or []:
            if isinstance(entry, dict) and isinstance(entry.get("box"), list):
                box = [int(round(v)) for v in entry["box"] if isinstance(v, (int, float))]
                if len(box) == 4:
                    blanks.append(box)
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
    return markers, hw_boxes, blanks


def ai_markers(
    image: np.ndarray,
) -> tuple[list[dict], list[list[int]], list[list[int]]] | None:
    """问 AI 每道题第一行的"题号+开头几个字"、手写痕迹和填空空位包围盒。

    返回 (题号列表, 归一化手写框, 归一化填空空位)；失败返回 None（走纯几何降级）。"""
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
            print(f"[ai_split] AI 返回内容解析不出，降级 OCR：{content[:120]}", flush=True)
            return None
        return parsed
    except Exception as exc:
        print(f"[ai_split] AI 题号识别失败，降级 OCR：{type(exc).__name__}: {str(exc)[:200]}", flush=True)
        return None


def _match_markers(
    strips: list[tuple[int, int, str, float]], markers: list[dict], width: int | None = None
) -> list[tuple[int, int, bool | None, str | None]]:
    """把 AI 说的每行匹配到 OCR 检测框上，返回按 y 升序的 (题界, 对错, 知识点)。

    从上往下逐个匹配且只往后找：AI 的顺序本身就是页面顺序，能防止
    "1.3.66" 错配到别的行。识别有噪（"2.（北京"→"2（北"），比归一化
    后的公共前缀，≥2 个字算命中；首位数字对上而后面被误读时放宽到 1
    （width 给定时只认左栏，防止全页框里同数字的算式混进来）；
    前缀对不上再退一步比整体相似度（多栏卷子的题干常被 OCR 拦腰截断）。
    """
    starts: list[tuple[int, int, bool | None, str | None]] = []
    floor = -1
    ordered = sorted(strips, key=lambda s: s[0])
    for marker in markers:
        key = _normalize(marker.get("text", ""))
        if len(key) < 2:
            continue
        best, best_prefix, best_ratio = None, 0, 0.0
        for y, x, text, _score in ordered:
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
            if best_prefix < 2 and len(key) >= 4 and len(candidate) >= 4:
                ratio = SequenceMatcher(None, key, candidate).ratio()
                if ratio > best_ratio:
                    best, best_ratio = (y, x), ratio
        matched = None
        if best is not None and best_prefix >= 2:
            matched = best
        elif (
            best is not None
            and best_prefix == 1
            and key[0].isdigit()
            and (width is None or best[1] <= width * 0.4)
        ):
            matched = best  # 题号数字对上了，后半被 OCR 误读
        elif best is not None and best_ratio >= 0.6:
            matched = best  # 前缀被 OCR 打断，整体 likeness 还在
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


def dedupe_blanks(boxes: list[list[int]]) -> list[list[int]]:
    """压制空位框的过度框选与重复。

    模型常把一道题的几个空连成一大片、又单独报出精确的小框；大框会连
    印刷文字一起盖掉。规则：两框重叠超过小框面积一半时视为同一个空，
    保留小框（定位更准）丢大框；完全相同的框只留一个。
    """
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for box in boxes:
        key = tuple(box)
        if key not in seen:
            seen.add(key)
            unique.append(list(box))

    def _area(b: list[int]) -> int:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def _overlap(a: list[int], b: list[int]) -> int:
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return max(0, w) * max(0, h)

    dropped: set[int] = set()
    for i, a in enumerate(unique):
        for j in range(i + 1, len(unique)):
            b = unique[j]
            smaller = min(_area(a), _area(b))
            if smaller > 0 and _overlap(a, b) / smaller > 0.5:
                dropped.add(i if _area(a) >= _area(b) else j)
    return [box for k, box in enumerate(unique) if k not in dropped]


def split_plan(
    image: np.ndarray, geometric_lines: list[int]
) -> tuple[list[int], bool, list[bool | None], list[list[int]], list[str | None]]:
    """AI 语义 + OCR 定位 + 几何空白带的最终合并。

    返回 (切线, AI 是否参与, 每条线对应题的对错判断, 手写痕迹框, 每题知识点,
    填空空位框)。手写框/空位是整图像素坐标 [左, 上, 右, 下]，与切线无关
    （AI 对整页的标注）；调用方把它们按题切分，手写框交给去手写用，
    空位存进 answer_masks 供练习卷遮答案。

    OCR 和 AI 相互独立，并行跑省一半等待。AI 参与时第一条线之上是
    页眉（校名、单元号……），不是题：调用方保存切片时应丢弃第一片。
    对错判断、知识点只跟 AI 认出的题界一一对齐，几何线补进的切线
    带占位的 None。AI 没给出任何可用信息时退回 ocr_split.merged_lines
    的行为，对错/知识点列表和手写框为空，页眉照旧成为第一张卡。
    """
    height, width = image.shape[:2]
    min_gap = max(6, int(height * 0.006))
    with ThreadPoolExecutor(max_workers=2 + AI_ATTEMPTS) as pool:
        strips_task = pool.submit(ocr_split.line_strips, image)
        marker_tasks = [pool.submit(ai_markers, image) for _ in range(AI_ATTEMPTS)]
        # 多栏卷子的题号常在行中间起头，全页文字框才能接住 AI 报的每道题
        boxes_task = pool.submit(ocr_split.text_boxes, image)
        strips = strips_task.result()
        results = [r for t in marker_tasks if (r := t.result()) is not None]

    wrong_flags: list[bool | None] = []
    knowledge_flags: list[str | None] = []
    hw_boxes_px: list[list[int]] = []
    blanks_px: list[list[int]] = []
    marker_sets = [markers for markers, _boxes, _blanks in results if markers]
    if marker_sets:
        page_boxes = boxes_task.result() or strips
        # 模型输出方差大，单次调用常漏一半的题：各次调用的匹配结果取并集
        matched_all: list[tuple[int, int, bool | None, str | None]] = []
        for markers, _boxes, _blanks in results:
            if markers:
                matched_all.extend(_match_markers(page_boxes, markers, width))
        matched_all.sort(key=lambda m: m[0])
        # 按最小间距去重（同一次调用内部的重复、两次调用定位到同一题）
        matched: list[tuple[int, int, bool | None, str | None]] = []
        last_y = -10**9
        for entry in matched_all:
            if entry[0] - last_y >= min_gap:
                matched.append(entry)
                last_y = entry[0]
        starts = [(y, x) for y, x, _w, _k in matched]
        wrong_flags = [w for _y, _x, w, _k in matched]
        knowledge_flags = [k for _y, _x, _w, k in matched]
        # 手写框取 marker 最多那次调用的（页面级标注，多份叠加会过度遮盖）
        best = max(results, key=lambda r: len(r[0]) if r[0] else -1)
        hw_boxes_norm = best[1]
        # 归一化手写框/空位换算成整图像素坐标
        def _to_px(boxes: list[list[int]]) -> list[list[int]]:
            return [
                [
                    int(round(v / 999 * (width if i % 2 == 0 else height)))
                    for i, v in enumerate(box)
                ]
                for box in boxes
            ]

        hw_boxes_px = _to_px(hw_boxes_norm)
        blanks_px = dedupe_blanks(_to_px(best[2]))
    else:
        starts = sorted(ocr_split.filter_starts(strips, width))

    mask = ocr_split._page_mask(image)
    profile = ocr_split._row_ink_profile(image, mask)
    bands = ocr_split._merge_across_thin_strokes(
        ocr_split._gap_bands(profile, ocr_split.BLANK_COVERAGE),
        max(8, int(height * 0.004)),
    )

    def _snap(starts_: list[tuple[int, int]]) -> list[int]:
        out = []
        for y, _x in starts_:
            target = max(0, y - int(height * ocr_split.TOP_MARGIN_RATIO))
            snapped = ocr_split._snap_to_blank_band(bands, target, int(height * 0.02))
            out.append(snapped if snapped is not None else target)
        return out

    # 降级并集 = OCR 题号线 + 几何线，也就是 AI 之前缺席时用户用惯的切法
    base_union = _dedup_lines(
        _snap(sorted(ocr_split.filter_starts(strips, width))) + list(geometric_lines),
        min_gap,
    )

    if not marker_sets:
        return base_union, False, [], [], [], []

    lines = _snap(starts)
    # 线和对错/知识点绑定去重（snap 可能让两条线靠得比 min_gap 近）
    rows = sorted(zip(lines, wrong_flags, knowledge_flags))
    aligned: list[tuple[int, bool | None, str | None]] = []
    for y, w, k in rows:
        if not aligned or y - aligned[-1][0] >= min_gap:
            aligned.append((y, w, k))

    # 降级并集永远并回来。以前的假设是"AI 已把每道题的起始说全"，实测
    # 不成立：多栏计算网格里的小题模型常不报（几何启发式在这类页面上
    # 也一条线都找不到），只信 AI 线会把十几道题挤成一张大杂烩卡。
    # OCR 题号线补足覆盖，AI 额外贡献行中间起头的题和对错/知识点。
    # 多出来的碎片卡在"修正"里一删就好，大杂烩难拆。
    for g in base_union:
        if all(abs(g - y) >= min_gap for y, _w, _k in aligned):
            aligned.append((g, None, None))
    aligned.sort()
    result: list[tuple[int, bool | None, str | None]] = []
    for y, w, k in aligned:
        if not result or y - result[-1][0] >= min_gap:
            result.append((y, w, k))
    return (
        [y for y, _w, _k in result],
        True,
        [w for _y, w, _k in result],
        hw_boxes_px,
        [k for _y, _w, k in result],
        blanks_px,
    )


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

