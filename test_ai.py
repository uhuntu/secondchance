import json
import urllib.error

import ai


class FakeResponse:
    def __iter__(self):
        result = {
            "知识点": "测试",
            "题目": "1+1=?",
            "孩子的作答": "3",
            "错因分析": "计算错误",
            "正确答案": "2",
            "讲解": "一加一等于二",
            "举一反三": [],
        }
        chunk = {"choices": [{"delta": {"content": json.dumps(result, ensure_ascii=False)}}]}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n".encode()
        yield b"data: [DONE]\n"


def test_transient_connection_error_is_retried():
    original_key = ai.get_api_key
    original_encode = ai._encode_image
    original_open = ai.API_OPENER.open
    original_sleep = ai.time.sleep
    calls = 0

    def fake_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("SSL EOF")
        return FakeResponse()

    try:
        ai.get_api_key = lambda: "test-key"
        ai._encode_image = lambda _: "image"
        ai.API_OPENER.open = fake_open
        ai.time.sleep = lambda _: None
        events = list(ai.analyze_image_events("unused.jpg"))
    finally:
        ai.get_api_key = original_key
        ai._encode_image = original_encode
        ai.API_OPENER.open = original_open
        ai.time.sleep = original_sleep

    assert calls == 2
    assert any(kind == "reasoning" and "自动重连" in value for kind, value in events)
    assert events[-1][0] == "result"
    assert events[-1][1]["正确答案"] == "2"


def test_wrong_question_ranges_are_scaled_and_filtered():
    original_events = ai.analyze_image_events
    ai.analyze_image_events = lambda *_: iter([
        ("result", {"错题区域": [
            {"题号": "1", "上": 100, "下": 300},
            {"题号": "x", "上": 500, "下": 510},
            {"题号": "3", "上": -20, "下": 1100},
        ]})
    ])
    try:
        ranges = ai.identify_wrong_question_ranges("unused.jpg", 1000)
    finally:
        ai.analyze_image_events = original_events
    assert ranges == [(0, 1000), (100, 300)]


if __name__ == "__main__":
    test_transient_connection_error_is_retried()
    test_wrong_question_ranges_are_scaled_and_filtered()
    print("AI retry tests passed")
