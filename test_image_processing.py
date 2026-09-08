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
    with TemporaryDirectory() as directory:
        source = Path(directory) / "page.jpg"
        cv2.imwrite(str(source), page)
        result = process_page(source, Path(directory), "test")
        assert result.page_detected
        assert result.mark_count == 3
        assert len(result.crop_paths) == 3
        assert all(path.is_file() for path in result.crop_paths)


if __name__ == "__main__":
    test_red_marks_create_question_bands()
    print("image processing tests passed")
