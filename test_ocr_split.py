import numpy as np

import ocr_split


def test_question_prefix_patterns():
    # 命中：数字加点、点被 OCR 吃掉、中文数字顿号、顿号被吞
    for text in ["1.3.66÷0.32", "3、线面", "5．下列", "12.A和B", "2（北京", "4AX1.1", "一、精心选择", "一精", "三、解决问题"]:
        assert ocr_split.QUESTION_PREFIX.match(text), text
    # 不命中：选项、算式中间、竖式、字母开头、纯数字
    for text in ["A.个位", "AO", "AD②", "16÷4=4", "=16÷4", "7m", "①27÷", "教", "曾营", "236"]:
        assert not ocr_split.QUESTION_PREFIX.match(text), text


def _page_with_gaps(height: int = 1200, width: int = 800) -> np.ndarray:
    import cv2

    page = np.full((height, width, 3), 255, dtype=np.uint8)
    for top, bottom in [(80, 300), (420, 640), (760, 1000)]:
        for y in range(top, bottom, 22):
            cv2.line(page, (80, y), (width - 80, y), (40, 40, 40), 3)
    return page


def test_merged_lines_adds_ocr_starts_geometric_missed(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    # 假装 OCR 在每个内容块顶上找到了题号行（几何漏掉的位置）
    starts = [(424, 80), (764, 80)]
    monkeypatch.setattr(ocr_split, "question_starts", lambda img: starts)
    merged = ocr_split.merged_lines(image, geometric)
    assert len(merged) > len(geometric)
    for y, _ in starts:
        assert any(abs(line - y) <= 30 for line in merged), (y, merged)


def test_merged_lines_dedups_lines_close_to_ocr(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    # 题号就在几何线正下方：不应产生两条贴在一起的线
    starts = [(y + 20, 80) for y in geometric]
    monkeypatch.setattr(ocr_split, "question_starts", lambda img: starts)
    merged = ocr_split.merged_lines(image, geometric)
    assert merged == geometric


def test_merged_lines_without_ocr_returns_geometric(monkeypatch):
    from image_processing import suggest_split_lines

    image = _page_with_gaps()
    geometric = suggest_split_lines(image)
    monkeypatch.setattr(ocr_split, "question_starts", lambda img: [])
    assert ocr_split.merged_lines(image, geometric) == geometric


def test_snap_to_blank_band_picks_nearest_center():
    bands = [(300, 420), (640, 760)]
    assert ocr_split._snap_to_blank_band(bands, 400, 40) == 360
    assert ocr_split._snap_to_blank_band(bands, 700, 40) == 700  # 太远就不吸附
    assert ocr_split._snap_to_blank_band([], 100, 40) is None


if __name__ == "__main__":
    test_question_prefix_patterns()
    test_merged_lines_adds_ocr_starts_geometric_missed(monkeypatch=None) if False else None
    print("ocr_split tests passed (run with pytest for monkeypatch tests)")
