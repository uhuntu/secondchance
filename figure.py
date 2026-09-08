"""把小正方体堆叠的高度图渲染成斜二测立体示意图。

高度图是二维数组：每个数字表示该位置叠了几个小正方体（0 表示没有），
第一行表示后排。渲染结果按内容哈希缓存为 PNG 文件。
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

CELL = 36          # 小正方体边长（像素）
DX = int(CELL * 0.42)   # 每往后一排向右的偏移
DY = int(CELL * 0.36)   # 每往后一排向上的偏移
FRONT = "#ffffff"
TOP = "#f2f2f2"
RIGHT = "#dcdcdc"
EDGE = "#333333"

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _draw_cube(draw: ImageDraw.ImageDraw, px: int, py: int) -> None:
    # 正面
    draw.rectangle([px, py - CELL, px + CELL, py], fill=FRONT, outline=EDGE)
    # 顶面
    draw.polygon(
        [
            (px, py - CELL),
            (px + DX, py - CELL - DY),
            (px + CELL + DX, py - CELL - DY),
            (px + CELL, py - CELL),
        ],
        fill=TOP,
        outline=EDGE,
    )
    # 右面
    draw.polygon(
        [
            (px + CELL, py - CELL),
            (px + CELL + DX, py - CELL - DY),
            (px + CELL + DX, py - DY),
            (px + CELL, py),
        ],
        fill=RIGHT,
        outline=EDGE,
    )


def render_stack(heights: list[list[int]]) -> Path:
    """渲染高度图，返回缓存的 PNG 路径（内容相同则复用）"""
    key = hashlib.md5(json.dumps(heights).encode()).hexdigest()
    path = FIG_DIR / f"{key}.png"
    if path.is_file():
        return path

    rows = len(heights)
    cols = max(len(r) for r in heights)
    max_h = max(max(r) for r in heights)

    width = cols * CELL + rows * DX + CELL
    height = max_h * CELL + rows * DY + CELL
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    ox = CELL // 2
    oy = height - DY - 4  # 前排最底层方块的底边

    # 从后排画到前排，每个 stack 从底往上画，保证遮挡关系正确
    # 后排（第一行）在视觉上更靠右上方，偏移量最大
    for y in range(rows):
        depth = rows - 1 - y  # 后排的 depth 大，向右上方偏移更多
        row = heights[y]
        for x in range(len(row)):
            for z in range(row[x]):
                px = ox + x * CELL + depth * DX
                py = oy - z * CELL - depth * DY
                _draw_cube(draw, px, py)

    img.save(path, "PNG")
    return path
