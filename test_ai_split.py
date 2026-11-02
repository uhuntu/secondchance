import numpy as np

import ai_split
import ocr_split


def test_parse_markers():
    fenced = (
        '```json\n{"questions": ['
        '{"marker": "2.", "text": "（北京市·东城区）", "wrong": true, "knowledge": "小数除法"}, '
        '{"marker": "3.", "text": "线面是福建省", "wrong": false, "knowledge": "循环小数"}], '
        '"handwriting": [{"box": [10, 20, 300, 120]}]}\n```'
    )
    assert ai_split._parse_markers(fenced) == (
        [
            {"text": "2.（北京市·东城区）", "wrong": True, "knowledge": "小数除法"},
            {"text": "3.线面是福建省", "wrong": False, "knowledge": "循环小数"},
        ],
        [[10, 20, 300, 120]],
        [],
    )
    # 缺 knowledge / 空串 / 非字符串都归 None
    missing = (
        '{"questions": ['
        '{"marker": "1.", "text": "第一题", "wrong": true, "knowledge": "   "}, '
        '{"marker": "2.", "text": "第二题", "wrong": false, "knowledge": 3}]}'
    )
    assert ai_split._parse_markers(missing) == (
        [
            {"text": "1.第一题", "wrong": True, "knowledge": None},
            {"text": "2.第二题", "wrong": False, "knowledge": None},
        ],
        [],
        [],
    )
    # 旧格式（裸数组，没有手写框）兼容
    assert ai_split._parse_markers('废话 [ {"marker": "5.", "text": "（龙岩市"} ] 废话') == (
        [{"text": "5.（龙岩市", "wrong": None, "knowledge": None}],
        [],
        [],
    )
    assert ai_split._parse_markers('["1.第一题"]') == (
        [{"text": "1.第一题", "wrong": None, "knowledge": None}],
        [],
        [],
    )
    assert ai_split._parse_markers("没有数组") is None
    assert ai_split._parse_markers("[{broken json]") is None
    assert ai_split._parse_markers("") is None


def test_parse_markers_drops_section_headers():
    """模型显式标注 section 的章节标题不是题界，直接丢弃。"""
    content = (
        '{"questions": ['
        '{"marker": "一、", "text": "精心选择。", "section": true, "wrong": null},'
        '{"marker": "1.", "text": "3.66÷0.32", "section": false, "wrong": true, "knowledge": "小数除法"},'
        '{"marker": "2.", "text": "(北京市·东", "section": false, "wrong": false}],'
        '"handwriting": []}'
    )
    assert ai_split._parse_markers(content) == (
        [
            {"text": "1.3.66÷0.32", "wrong": True, "knowledge": "小数除法"},
            {"text": "2.(北京市·东", "wrong": False, "knowledge": None},
        ],
        [],
        [],
    )
    # 旧格式（没有 section 字段）照常保留
    assert ai_split._parse_markers('[{"marker": "5.", "text": "（龙岩市"}]') == (
        [{"text": "5.（龙岩市", "wrong": None, "knowledge": None}],
        [],
        [],
    )


def _strips() -> list[tuple[int, int, str, float]]:
    return [
        (90, 400, "曾营小学2026", 0.9),   # 页眉，不是题号
        (130, 120, "学", 0.4),            # 左侧手写列
        (156, 410, "一、精心", 0.96),
        (198, 408, "1.3.66÷0", 0.99),
        (240, 460, "A个位", 0.96),        # 选项行
        (282, 409, "2（北", 0.95),        # 小数点被 OCR 吃掉
        (520, 410, "3.线面是", 0.97),
        (683, 412, "4AX", 0.97),          # × 被误读为 X
        (803, 409, "5.（龙", 0.95),
        (1152, 405, "6.肯", 0.95),        # 莆 被误读为 肯
    ]


def test_ocr_filter_starts_finds_margin_numbers():
    starts = ocr_split.filter_starts(_strips(), width=2481)
    ys = [y for y, _ in starts]
    # 页眉和选项行、手写列都不算；六个题号行 + 章节行都在
    assert ys == [156, 198, 282, 520, 683, 803, 1152]


def test_match_markers_handles_misreads_and_stays_in_order():
    markers = [
        {"text": "一、精心选择。", "wrong": None, "knowledge": None},
        {"text": "1.3.66÷0.32的商", "wrong": True, "knowledge": "小数除法"},
        {"text": "2.（北京市·东城区）在计算", "wrong": False, "knowledge": "循环小数"},
        {"text": "4.A×1.1=B÷1.1", "wrong": True, "knowledge": None},
        {"text": "6.（莆田市·涵江区）玲玲", "wrong": False, "knowledge": "小数乘法"},
    ]
    starts = ai_split._match_markers(_strips(), markers)
    # 2 和 4、6 的行首被 OCR 误读，靠"数字对上"兜底；顺序不乱，
    # 对错判断和知识点跟着各自的行走
    assert [(y, w, k) for y, _x, w, k in starts] == [
        (156, None, None), (198, True, "小数除法"), (282, False, "循环小数"),
        (683, True, None), (1152, False, "小数乘法"),
    ]


def test_match_markers_skips_unmatched():
    starts = ai_split._match_markers(_strips(), [{"text": "7.根本不存在的题目行", "wrong": True}])
    assert starts == []


def test_boxes_per_crop_splits_and_shifts():
    # 页高 1000，线 [300, 700] → 三片 [0,300] [300,700] [700,1000]
    boxes = [[50, 100, 200, 250], [100, 250, 300, 400], [400, 950, 500, 990]]
    per_crop = ai_split.boxes_per_crop(boxes, [300, 700], 1000, skip_first=False)
    # 框 0 整在第一片；框 1 跨 300 边界，两片各留裁剪后的部分
    assert per_crop[0] == [[50, 100, 200, 250], [100, 250, 300, 300]]
    assert per_crop[1] == [[100, 0, 300, 100]]
    assert per_crop[2] == [[400, 250, 500, 290]]
    # skip_first：第一片丢弃，框随编号前移
    per_crop2 = ai_split.boxes_per_crop(boxes, [300, 700], 1000, skip_first=True)
    assert per_crop2[0] == [[100, 0, 300, 100]]
    assert per_crop2[1] == [[400, 250, 500, 290]]


def _page_with_gaps(height: int = 1200, width: int = 800) -> np.ndarray:
    import cv2

    page = np.full((height, width, 3), 255, dtype=np.uint8)
    for top, bottom in [(80, 300), (420, 640), (760, 1000)]:
        for y in range(top, bottom, 22):
            cv2.line(page, (80, y), (width - 80, y), (40, 40, 40), 3)
    return page


def test_merged_lines_with_ai_keeps_geometric_union(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    strips = [(84, 80, "一、第一块", 0.9), (424, 80, "1.第二块", 0.9), (764, 80, "2.第三块", 0.9)]
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: strips)
    monkeypatch.setattr(ocr_split, "text_boxes", lambda img: strips)
    monkeypatch.setattr(
        ai_split, "ai_markers",
        lambda img: (
            [
                {"text": "一、第一块", "wrong": True, "knowledge": None},
                {"text": "1.第二块", "wrong": False, "knowledge": "小数乘法"},
                {"text": "2.第三块", "wrong": True, "knowledge": "多边形的面积"},
            ],
            [],  # 无手写框
            [],  # 无空位框
        ),
    )
    lines, ai_active, wrong_flags, hw_boxes, knowledge_flags, blanks = ai_split.split_plan(image, geometric)
    # AI 题界和几何线取并集：AI 常漏报，几何空白带补覆盖
    assert ai_active is True
    # AI 认出的三道题都在，flags 跟着各自的线
    assert [w for w in wrong_flags if w is not None] == [True, False, True]
    assert [k for k in knowledge_flags if k is not None] == ["小数乘法", "多边形的面积"]
    # 几何线独立于 AI 线时保留，flags 补 None
    assert len(wrong_flags) >= 3 and wrong_flags.count(None) == len(lines) - 3
    assert hw_boxes == []
    # 三块的起始线都在（AI 线），块间的空白带线也在（几何线）
    assert any(l < 120 for l in lines)
    assert any(380 < l < 480 for l in lines)
    assert any(720 < l < 820 for l in lines)
    assert all(g in lines for g in geometric)
    assert ai_split.merged_lines(image, geometric) == lines

    # AI 漏报时的兜底：几何线落在 AI 没认出的位置 → 保留，flags 补 None
    lines2, _ai, wrong2, _hw, knowledge2, _blanks2 = ai_split.split_plan(image, geometric + [550])
    assert 550 in lines2 and len(lines2) == len(lines) + 1
    idx = lines2.index(550)
    assert wrong2[idx] is None and knowledge2[idx] is None
    # 其余线的 flags 不变
    assert [w for i, w in enumerate(wrong2) if i != idx] == wrong_flags


def test_split_plan_without_ai_keeps_geometric_union(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    strips = [(84, 80, "1.第一块", 0.9)]
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: strips)
    monkeypatch.setattr(ai_split, "ai_markers", lambda img: None)
    lines, ai_active, wrong_flags, hw_boxes, knowledge_flags, blanks = ai_split.split_plan(image, geometric)
    assert ai_active is False and wrong_flags == [] and hw_boxes == []
    assert knowledge_flags == [] and blanks == []
    assert lines == ocr_split.merged_lines(image, geometric)


def test_merged_lines_degrades_without_api_key(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    monkeypatch.setattr(ai_split, "_api_key", lambda: None)
    assert ai_split.ai_markers(image) is None
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: [])
    lines, ai_active, wrong_flags, hw_boxes, knowledge_flags, blanks = ai_split.split_plan(image, geometric)
    assert ai_active is False and wrong_flags == [] and hw_boxes == []
    assert knowledge_flags == [] and blanks == []
    assert lines == ocr_split.merged_lines(image, geometric)


def test_snap_to_blank_band_picks_nearest_center():
    bands = [(300, 420), (640, 760)]
    assert ocr_split._snap_to_blank_band(bands, 400, 40) == 360
    assert ocr_split._snap_to_blank_band(bands, 700, 40) == 700  # 太远就不吸附
    assert ocr_split._snap_to_blank_band([], 100, 40) is None


def test_dedupe_blanks_prefers_smaller_box():
    # 大框罩住两个小框（连印刷文字一起盖）→ 大框丢、小框留
    big = [50, 100, 550, 300]
    small1 = [180, 150, 230, 290]
    small2 = [380, 150, 440, 290]
    assert ai_split.dedupe_blanks([big, small1, small2]) == [small1, small2]
    # 完全相同的框只留一个
    assert ai_split.dedupe_blanks([[10, 10, 60, 80], [10, 10, 60, 80]]) == [[10, 10, 60, 80]]
    # 近似重复（两次输出差几个像素）→ 留一个
    assert ai_split.dedupe_blanks([[100, 50, 160, 200], [98, 52, 162, 198]]) == [[100, 50, 160, 200]]
    # 互不重叠的原样保留
    assert ai_split.dedupe_blanks([[0, 0, 50, 50], [500, 500, 600, 600]]) == [
        [0, 0, 50, 50],
        [500, 500, 600, 600],
    ]


if __name__ == "__main__":
    test_parse_markers()
    print("ai_split tests passed (run with pytest for monkeypatch tests)")
