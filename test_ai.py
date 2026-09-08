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
    original_open = ai.urllib.request.urlopen
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
        ai.urllib.request.urlopen = fake_open
        ai.time.sleep = lambda _: None
        events = list(ai.analyze_image_events("unused.jpg"))
    finally:
        ai.get_api_key = original_key
        ai._encode_image = original_encode
        ai.urllib.request.urlopen = original_open
        ai.time.sleep = original_sleep

    assert calls == 2
    assert any(kind == "reasoning" and "自动重连" in value for kind, value in events)
    assert events[-1][0] == "result"
    assert events[-1][1]["正确答案"] == "2"


if __name__ == "__main__":
    test_transient_connection_error_is_retried()
    print("AI retry tests passed")
