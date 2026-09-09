from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from image_processing import process_page


def test_red_marks_create_question_bands():
    page = np.full((1200, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (45, 35), (755, 1165), (20, 20, 20), 4)
    for y in (250, 650, 980):
        cv2.line(page, (600, y - 25), (650, y + 25), (0, 0, 220), 10)
        cv2.line(page, (650, y - 25), (600, y + 25), (0, 0, 220), 10)
    # Two wrong marks on the same answer line are one candidate question.
    cv2.line(page, (300, 625), (350, 675), (0, 0, 220), 10)
    cv2.line(page, (350, 625), (300, 675), (0, 0, 220), 10)
    # A small teacher tick is a correct-answer mark and must not create a crop.
    cv2.line(page, (180, 450), (188, 458), (0, 0, 220), 3)
    cv2.line(page, (188, 458), (205, 438), (0, 0, 220), 3)
    with TemporaryDirectory() as directory:
        source = Path(directory) / "page.jpg"
        cv2.imwrite(str(source), page)
        result = process_page(source, Path(directory), "test")
        assert result.page_detected
        assert result.mark_count == 3
        assert len(result.crop_paths) == 3
        assert all(path.is_file() for path in result.crop_paths)


def test_manual_corners_warp_page_before_cropping():
    photo = np.full((900, 900, 3), 70, dtype=np.uint8)
    page = np.full((700, 500, 3), 255, dtype=np.uint8)
    cv2.line(page, (390, 300), (440, 350), (0, 0, 220), 10)
    cv2.line(page, (440, 300), (390, 350), (0, 0, 220), 10)
    source_corners = np.float32([[160, 80], [720, 150], [780, 820], [90, 760]])
    target_corners = np.float32([[0, 0], [499, 0], [499, 699], [0, 699]])
    matrix = cv2.getPerspectiveTransform(target_corners, source_corners)
    warped = cv2.warpPerspective(page, matrix, (900, 900))
    mask = cv2.warpPerspective(
        np.full((700, 500), 255, dtype=np.uint8), matrix, (900, 900)
    )
    photo[mask > 0] = warped[mask > 0]
    with TemporaryDirectory() as directory:
        source = Path(directory) / "manual.jpg"
        cv2.imwrite(str(source), photo)
        result = process_page(
            source, Path(directory), "manual", source_corners.tolist()
        )
        corrected = cv2.imread(str(result.corrected_path))
        assert result.page_detected
        assert result.mark_count == 1
        assert 450 <= corrected.shape[1] <= 700
        assert 650 <= corrected.shape[0] <= 800


def test_photo_frame_is_not_mistaken_for_paper():
    photo = np.full((900, 700, 3), 245, dtype=np.uint8)
    cv2.rectangle(photo, (1, 1), (698, 898), (20, 20, 20), 3)
    with TemporaryDirectory() as directory:
        source = Path(directory) / "frame.jpg"
        cv2.imwrite(str(source), photo)
        result = process_page(source, Path(directory), "frame")
        assert not result.page_detected


if __name__ == "__main__":
    test_red_marks_create_question_bands()
    test_manual_corners_warp_page_before_cropping()
    test_photo_frame_is_not_mistaken_for_paper()
    print("image processing tests passed")
