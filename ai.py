"""调用月之暗面（Moonshot）视觉模型分析错题照片（流式输出思考过程）。

API Key 读取优先级：环境变量 MOONSHOT_API_KEY > 项目目录 api_key.txt
"""
import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path

API_BASE = "https://api.moonshot.cn/v1/chat/completions"
MODEL = "kimi-k2.6"
KEY_FILE = Path(__file__).parent / "api_key.txt"

PROMPT = """你是一位耐心细致的小学数学老师，擅长辅导五年级学生。请分析这张错题照片，完成以下任务：

1. 识别题目：转写出这道题的完整文字（含图中给出的图形/条件的文字描述）。
2. 识别孩子的作答：写出孩子写下的答案或解题过程；如果照片里看不清或没有孩子的作答，就写"未能识别"。
3. 判断错误：分析孩子错在哪里、为什么错（是概念不清、计算失误、审题偏差还是方法错误）。
4. 给出正确答案。
5. 用孩子能听懂的话讲解这道题的思路。
6. 归类：给出本题的年级知识点名称（简短，5到12个字，如"小数乘法""组合图形的面积"）。
7. 举一反三：出 2 道考察同一知识点的类似题。要求：
- 题目文字本身保持简洁自然，像一个正常的练习题，不要出现"如右图""选项为四种图形"这类引用；
- "图"字段按题目需要取以下形式之一：
  a. 展示从某方向看到的平面图形：直接给字符画字符串，用 "■" 表示有方块、"□" 表示没有，行列上下对齐，并在上方标注观察方向（如"从上面看："），图不超过 5 列 3 行；
  b. 展示立体组合图形：给 JSON 对象 {"立体": [[...]]}，二维数组中每个数字表示对应地面位置叠了几个小正方体（0 表示该位置没有），第一行表示后排、每行从左到右，总数不超过 12 个；例如 {"立体": [[1,2],[2,0]]} 表示后排左起叠 1 个、2 个，前排左边叠 2 个；此时可另加 "字符画" 键给出关键视图辅助理解；
- 每道题附 "答案"，先不给详细解析。
仍按此 JSON 格式输出，举一反三部分的每项为 {"题目": "...", "图": "...", "答案": "..."}（不需要图时可省略"图"字段）：

仍按此 JSON 格式输出，举一反三部分的每项为 {"题目": "...", "图": "...", "答案": "..."}（不需要图时可省略"图"字段）：
{"知识点": "...", "题目": "...", "孩子的作答": "...", "错因分析": "...", "正确答案": "...", "讲解": "...", "举一反三": [{"题目": "...", "图": "...", "答案": "..."}, {"题目": "...", "图": "...", "答案": "..."}], "原题重练": {"题目": "...", "图": "..."}}

注意"原题重练"：把照片中的原题整理成一道干净的、可直接打印给孩子重做的题目——不得包含孩子的作答、批改痕迹等任何痕迹；数字、条件、图形必须与照片完全一致，不得改动；"图"字段规则与上面第 7 条相同（立体图形用 {"立体": [[...]]} 高度图，平面视图用字符画字符串）；如原题是纯文字计算题则不需要"图"字段。"""


def get_api_key() -> str | None:
    key = os.environ.get("MOONSHOT_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.is_file():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        return key or None
    return None


def _encode_image(image_path: str) -> str:
    """压缩后转 base64：限制最长边 1568px（模型推荐），加快传输和识别"""
    from PIL import Image

    img = Image.open(image_path)
    if max(img.size) > 1568:
        ratio = 1568 / max(img.size)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS
        )
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def analyze_image_events(image_path: str):
    """生成器：逐段产出 ("reasoning", 思考片段) / ("content", 结果片段)，
    最后产出 ("result", 解析后的字段 dict)。
    不做全局串行：账号并发超限会返回 429，由调用方重试，
    避免一个断开的请求卡住后面所有分析。"""
    key = get_api_key()
    if not key:
        raise RuntimeError("未配置 MOONSHOT_API_KEY（环境变量或 api_key.txt）")

    body = {
        "model": MODEL,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"},
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        API_BASE,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )

    content_parts: list[str] = []
    resp = None
    last_error = None
    for attempt in range(8):
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            last_error = RuntimeError(f"模型接口调用失败（HTTP {e.code}）：{detail}")
            # 429 限流（账号并发限制）时等待重试，其余错误直接失败
            if e.code == 429 and attempt < 7:
                yield ("reasoning", f"\n[账号繁忙，第 {attempt + 1} 次重试，请稍候…]\n")
                time.sleep(8 * (attempt + 1))
                continue
            raise last_error
    if resp is None:
        raise last_error

    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("reasoning_content"):
            yield ("reasoning", delta["reasoning_content"])
        if delta.get("content"):
            content_parts.append(delta["content"])
            yield ("content", delta["content"])

    text = "".join(content_parts).strip()
    # 兼容 ```json ... ``` 包裹的输出
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    yield ("result", json.loads(text.strip()))


def analyze_image(image_path: str) -> dict:
    """非流式封装：只取最终结果"""
    result = None
    for kind, payload in analyze_image_events(image_path):
        if kind == "result":
            result = payload
    if result is None:
        raise RuntimeError("模型未返回分析结果")
    return result
