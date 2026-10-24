import numpy as np

import ai_split
import ocr_split


def test_parse_markers():
    fenced = '```json\n[{"marker": "2.", "text": "（北京市·东城区）"}, {"marker": "3.", "text": "线面是福建省"}]\n```'
    assert ai_split._parse_markers(fenced) == ["2.（北京市·东城区）", "3.线面是福建省"]
    assert ai_split._parse_markers('废话 [ {"marker": "5.", "text": "（龙岩市"} ] 废话') == ["5.（龙岩市"]
    assert ai_split._parse_markers('["1.第一题", "2.第二题"]') == ["1.第一题", "2.第二题"]
    assert ai_split._parse_markers("没有数组") is None
    assert ai_split._parse_markers("[{broken json]") is None
    assert ai_split._parse_markers("") is None


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
    markers = ["一、精心选择。", "1.3.66÷0.32的商", "2.（北京市·东城区）在计算", "4.A×1.1=B÷1.1", "6.（莆田市·涵江区）玲玲"]
    starts = ai_split._match_markers(_strips(), markers)
    ys = [y for y, _ in starts]
    # 2 和 4、6 的行首被 OCR 误读，靠"数字对上"兜底；顺序不乱
    assert ys == [156, 198, 282, 683, 1152]


def test_match_markers_skips_unmatched():
    starts = ai_split._match_markers(_strips(), ["7.根本不存在的题目行"])
    assert starts == []


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
    monkeypatch.setattr(ai_split, "ai_markers", lambda img: ["一、第一块", "1.第二块", "2.第三块"])
    merged = ai_split.merged_lines(image, geometric)
    # 三道题各一条线，几何线全部让位给 AI 题界（题内错切 + 页眉页脚碎片都不要）
    assert len(merged) == 3
    assert merged[0] < 120 and 300 < merged[1] < 440 and 640 < merged[2] < 780


def test_merged_lines_with_ai_drops_margin_fragments(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    strips = [(424, 80, "1.第二块", 0.9), (764, 80, "2.第三块", 0.9)]
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: strips)
    monkeypatch.setattr(ai_split, "ai_markers", lambda img: ["1.第二块", "2.第三块"])
    merged = ai_split.merged_lines(image, geometric)
    # 首题之前的页眉碎片线（几何在第一题上方给的线）一并丢弃
    assert all(line > 300 for line in merged), merged


def test_merged_lines_without_ai_keeps_geometric_union(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    strips = [(84, 80, "1.第一块", 0.9)]
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: strips)
    monkeypatch.setattr(ai_split, "ai_markers", lambda img: None)
    assert ai_split.merged_lines(image, geometric) == ocr_split.merged_lines(image, geometric)


def test_merged_lines_degrades_without_api_key(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    monkeypatch.setattr(ai_split, "_api_key", lambda: None)
    assert ai_split.ai_markers(image) is None
    monkeypatch.setattr(ocr_split, "line_strips", lambda img: [])
    assert ai_split.merged_lines(image, geometric) == ocr_split.merged_lines(image, geometric)


if __name__ == "__main__":
    test_parse_markers()
    test_ocr_filter_starts_finds_margin_numbers()
    test_match_markers_handles_misreads_and_stays_in_order()
    test_match_markers_skips_unmatched()
    print("ai_split tests passed (run with pytest for monkeypatch tests)")
