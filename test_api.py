"""端到端 API 验证脚本：增、查、改、删、照片访问"""
import base64
import json
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8010"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def req(method, path, body=None, headers=None):
    r = urllib.request.Request(BASE + path, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(r) as resp:
        data = resp.read()
        return resp.status, json.loads(data) if data else None


# 1. POST multipart 上传
boundary = "----testboundary1234"
fields = {
    "subject": "数学",
    "knowledge_point": "分数计算",
    "note": "正确答案 3/4",
}
parts = b""
for k, v in fields.items():
    parts += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
    ).encode()
parts += (
    f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; '
    f'filename="wrong.png"\r\nContent-Type: image/png\r\n\r\n'
).encode() + PNG + f"\r\n--{boundary}--\r\n".encode()

status, created = req(
    "POST", "/api/questions", parts, {"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
assert status == 201, created
assert created["subject"] == "数学" and created["knowledge_point"] == "分数计算"
print(f"1. POST 创建成功 id={created['id']}")

qid = created["id"]

# 2. 列表 + 过滤 + 搜索
status, lst = req("GET", "/api/questions")
assert any(q["id"] == qid for q in lst)
qs = urllib.parse.urlencode({"subject": "数学"})
status, lst = req("GET", f"/api/questions?{qs}")
assert len(lst) == 1, lst
qs = urllib.parse.urlencode({"keyword": "分数"})
status, lst = req("GET", f"/api/questions?{qs}")
assert len(lst) == 1, lst
qs = urllib.parse.urlencode({"keyword": "不存在的词"})
status, lst = req("GET", f"/api/questions?{qs}")
assert lst == []
print("2. 列表/科目过滤/关键词搜索 通过")

# 3. 详情 + 照片访问
status, detail = req("GET", f"/api/questions/{qid}")
assert detail["note"] == "正确答案 3/4"
r = urllib.request.urlopen(BASE + detail["image_url"])
assert r.status == 200 and r.read() == PNG
print("3. 详情 + 照片原图访问 通过")

# 4. PUT 编辑
body = json.dumps({"knowledge_point": "约分", "note": "已订正"}).encode()
status, updated = req("PUT", f"/api/questions/{qid}", body, {"Content-Type": "application/json"})
assert updated["knowledge_point"] == "约分" and updated["note"] == "已订正"
print("4. PUT 编辑 通过")

# 5. DELETE 删除
status, _ = req("DELETE", f"/api/questions/{qid}")
assert status == 204
try:
    req("GET", f"/api/questions/{qid}")
    raise SystemExit("删除后仍能查到，失败")
except urllib.error.HTTPError as e:
    assert e.code == 404
import pathlib
leftover = list(pathlib.Path("uploads").glob("*.png"))
assert not leftover, leftover  # 测试图已清理；残留的是 curl 乱码测试数据
print("5. DELETE 删除（记录+照片）通过")

print("\n全部端到端测试通过 ✔")
