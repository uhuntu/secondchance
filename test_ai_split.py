import numpy as np

import ai_split
import ocr_split


def test_parse_markers():
    fenced = (
        '```json\n{"questions": ['
        '{"marker": "2.", "text": "（北京市·东城区）", "wrong": true}, '
        '{"marker": "3.", "text": "线面是福建省", "wrong": false}], '
        '"handwriting": [{"box": [10, 20, 300, 120]}]}\n```'
    )
    assert ai_split._parse_markers(fenced) == (
        [
            {"text": "2.（北京市·东城区）", "wrong": True},
            {"text": "3.线面是福建省", "wrong": False},
        ],
        [[10, 20, 300, 120]],
    )
    # 旧格式（裸数组，没有手写框）兼容
    assert ai_split._parse_markers('废话 [ {"marker": "5.", "text": "（龙岩市"} ] 废话') == (
        [{"text": "5.（龙岩市", "wrong": None}],
        [],
    )
    assert ai_split._parse_markers('["1.第一题"]') == (
        [{"text": "1.第一题", "wrong": None}],
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
        '{"marker": "1.", "text": "3.66÷0.32", "section": false, "wrong": true},'
        '{"marker": "2.", "text": "(北京市·东", "section": false, "wrong": false}],'
        '"handwriting": []}'
    )
    assert ai_split._parse_markers(content) == (
        [
            {"text": "1.3.66÷0.32", "wrong": True},
            {"text": "2.(北京市·东", "wrong": False},
        ],
        [],
    )
    # 旧格式（没有 section 字段）照常保留
    assert ai_split._parse_markers('[{"marker": "5.", "text": "（龙岩市"}]') == (
        [{"text": "5.（龙岩市", "wrong": None}],
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
        {"text": "一、精心选择。", "wrong": None},
        {"text": "1.3.66÷0.32的商", "wrong": True},
        {"text": "2.（北京市·东城区）在计算", "wrong": False},
        {"text": "4.A×1.1=B÷1.1", "wrong": True},
        {"text": "6.（莆田市·涵江区）玲玲", "wrong": False},
    ]
    starts = ai_split._match_markers(_strips(), markers)
    # 2 和 4、6 的行首被 OCR 误读，靠"数字对上"兜底；顺序不乱，
    # 对错判断跟着各自的行走
    assert [(y, w) for y, _x, w in starts] == [
        (156, None), (198, True), (282, False), (683, True), (1152, False)
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


def test_merged_lines_with_ai_drops_junk_inside_spans(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    strips = [(84, 80, "一、第一块", 0.9), (424, 80, "1.第二块", 0.9), (764, 80, "2.第三块", 0.9)]
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: strips)
    monkeypatch.setattr(
        ai_split, "ai_markers",
        lambda img: (
            [
                {"text": "一、第一块", "wrong": True},
                {"text": "1.第二块", "wrong": False},
                {"text": "2.第三块", "wrong": True},
            ],
            [],  # 无手写框
        ),
    )
    lines, ai_active, wrong_flags, hw_boxes = ai_split.split_plan(image, geometric)
    # 几何线全部让位给 AI 题界：3 道题 3 条线，调用方丢弃第一片（页眉）
    assert ai_active is True
    assert len(lines) == 3
    assert wrong_flags == [True, False, True]
    assert hw_boxes == []
    assert lines[0] < 120 and 300 < lines[1] < 440 and 640 < lines[2] < 780
    assert ai_split.merged_lines(image, geometric) == lines


def test_split_plan_without_ai_keeps_geometric_union(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    strips = [(84, 80, "1.第一块", 0.9)]
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: strips)
    monkeypatch.setattr(ai_split, "ai_markers", lambda img: None)
    lines, ai_active, wrong_flags, hw_boxes = ai_split.split_plan(image, geometric)
    assert ai_active is False and wrong_flags == [] and hw_boxes == []
    assert lines == ocr_split.merged_lines(image, geometric)


def test_merged_lines_degrades_without_api_key(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    monkeypatch.setattr(ai_split, "_api_key", lambda: None)
    assert ai_split.ai_markers(image) is None
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: [])
    lines, ai_active, wrong_flags, hw_boxes = ai_split.split_plan(image, geometric)
    assert ai_active is False and wrong_flags == [] and hw_boxes == []
    assert lines == ocr_split.merged_lines(image, geometric)


def test_snap_to_blank_band_picks_nearest_center():
    bands = [(300, 420), (640, 760)]
    assert ocr_split._snap_to_blank_band(bands, 400, 40) == 360
    assert ocr_split._snap_to_blank_band(bands, 700, 40) == 700  # 太远就不吸附
    assert ocr_split._snap_to_blank_band([], 100, 40) is None


if __name__ == "__main__":
    test_parse_markers()
    print("ai_split tests passed (run with pytest for monkeypatch tests)")
