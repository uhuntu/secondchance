import io
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from image_processing import (
    analyze,
    crop_by_lines,
    enhance_print,
    remove_handwriting,
    heic_to_jpeg_bytes,
    straighten,
    suggest_split_lines,
    warp_with_corners,
)


def _page_with_gaps(height: int = 1200, width: int = 800) -> np.ndarray:
    """一张白纸，内容分成三段，中间留两段明显的空白带。"""
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    blocks = [(80, 300), (420, 640), (760, 1000)]
    for top, bottom in blocks:
        for y in range(top, bottom, 22):
            cv2.line(page, (80, y), (width - 80, y), (40, 40, 40), 3)
    return page


def _write(directory: Path, name: str, image: np.ndarray) -> Path:
    path = directory / name
    cv2.imwrite(str(path), image)
    return path


def test_split_lines_land_in_the_blank_gaps():
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        source = _write(directory, "page.jpg", _page_with_gaps())
        lines = suggest_split_lines(source)
        height = _page_with_gaps().shape[0]
        # 两段空白带分别在 300~420 和 640~760，线应落在其中
        assert len(lines) == 2, lines
        assert 300 <= lines[0] <= 420
        assert 640 <= lines[1] <= 760
        assert all(0 < line < height for line in lines)


def test_crop_by_lines_splits_into_bands():
    image = np.zeros((900, 400, 3), dtype=np.uint8)
    crops = crop_by_lines(image, [300, 600])
    assert [crop.shape[0] for crop in crops] == [300, 300, 300]
    # 没有分割线时整张图就是一道题
    assert len(crop_by_lines(image, [])) == 1


def test_analyze_returns_page_and_suggested_lines():
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        source = _write(directory, "page.jpg", _page_with_gaps())
        result = analyze(source)
        assert result.image.shape[0] > 0
        assert result.width > 0 and result.height == result.image.shape[0]
        assert len(result.lines) == 2


def test_straighten_falls_back_to_original_when_paper_is_ambiguous():
    """整幅都是亮色、找不到可靠纸张时，用原图而不是报错。"""
    flat = np.full((700, 900, 3), 245, dtype=np.uint8)
    cv2.rectangle(flat, (1, 1), (698, 898), (20, 20, 20), 3)
    result, detected = straighten(flat)
    assert detected is False
    assert result.shape == flat.shape


def test_remove_handwriting_drops_pencil_and_red_keeps_print():
    """铅笔和红笔被填回纸色，印刷体一个像素不少。

    线画到 200（约半行宽）为止：真实文字行的墨水不会占满整行，
    占满了会把"该行纸面亮度"的中位数拖成墨水色，参照系就失真了。
    """
    page = np.full((600, 400, 3), 245, dtype=np.uint8)
    for y in range(60, 120, 20):                     # 印刷体：深黑横线
        cv2.line(page, (40, y), (200, y), (30, 30, 30), 3)
    for y in range(200, 260, 20):                    # 铅笔：中灰横线
        cv2.line(page, (40, y), (200, y), (150, 150, 150), 3)
    cv2.circle(page, (200, 350), 40, (60, 60, 230), 5)  # 红笔圈

    cleaned = remove_handwriting(page)

    # 印刷体还在（采样一条印刷横线的中心）
    assert cleaned[60, 100, 0] < 80
    # 铅笔线中心被填回纸色
    assert cleaned[200, 100, 0] > 220
    # 红圈中心被填回纸色（三个通道都到纸色，不再是红笔的"蓝低红高"）
    assert cleaned[350, 200, 0] > 200 and abs(int(cleaned[350, 200, 2]) - int(cleaned[350, 200, 0])) < 30


def test_remove_handwriting_keeps_red_on_text_removes_red_in_blank():
    """红笔压着印刷文字时保留（否则题目被毁）；空白处的红笔涂鸦移除。"""
    page = np.full((600, 400, 3), 245, dtype=np.uint8)
    cv2.line(page, (40, 100), (200, 100), (30, 30, 30), 3)   # 印刷文字行
    cv2.line(page, (40, 104), (200, 104), (60, 60, 230), 3)  # 红杠紧贴文字
    cv2.line(page, (40, 400), (200, 400), (60, 60, 230), 3)  # 红杠在空白处

    cleaned = remove_handwriting(page)

    # 贴着文字的红杠保留（红通道明显高于蓝通道）
    assert cleaned[104, 100, 2] - cleaned[104, 100, 0] > 60
    # 空白处的红杠被填回纸色
    assert cleaned[400, 100, 0] > 200


def test_remove_handwriting_skips_pages_without_handwriting():
    """只有印刷的页面：找不到可去的铅笔，原样返回（不误删）。"""
    page = np.full((600, 400, 3), 245, dtype=np.uint8)
    for y in range(60, 500, 20):
        cv2.line(page, (40, y), (200, y), (30, 30, 30), 3)
    cleaned = remove_handwriting(page)
    assert np.array_equal(cleaned, page)


def test_enhance_print_whitens_paper_and_darkens_ink():
    """打印增强：灰底拉白、墨迹加深；纸张外的阴影刷白不留黑边。"""
    page = np.full((600, 400, 3), 215, dtype=np.uint8)   # 灰绿纸面（手机照）
    for y in range(60, 500, 40):                          # 印刷体：深灰横线
        cv2.line(page, (40, y), (300, y), (90, 90, 90), 3)
    page[:, 380:] = 120                                   # 纸外桌面阴影

    out = enhance_print(page)

    # 纸面变纯白
    assert out[300, 350, 0] >= 245
    # 墨迹加深（90 的灰线压到接近黑）
    assert out[60, 100, 0] < 60
    # 纸外阴影被刷白，不会印出黑边
    assert (out[:, 385:] == 255).all()


def test_warp_with_corners_rejects_points_outside_the_photo():
    image = np.full((900, 900, 3), 255, dtype=np.uint8)
    for bad in ([], [[0, 0], [1, 1]], [[0, 0], [10, 0], [10, 10], [2000, 10]]):
        try:
            warp_with_corners(image, bad)
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝这组角点: {bad}")


def test_heic_bytes_are_transcoded_to_jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buffer, format="HEIF")
    jpeg = heic_to_jpeg_bytes(buffer.getvalue())
    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.format == "JPEG"
        assert image.size == (64, 48)


if __name__ == "__main__":
    test_split_lines_land_in_the_blank_gaps()
    test_crop_by_lines_splits_into_bands()
    test_analyze_returns_page_and_suggested_lines()
    test_straighten_falls_back_to_original_when_paper_is_ambiguous()
    test_warp_with_corners_rejects_points_outside_the_photo()
    test_heic_bytes_are_transcoded_to_jpeg()
    print("image processing tests passed")