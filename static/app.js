// 常见错误原因
const ERROR_REASONS = ["计算错误", "审题不清", "概念不清", "公式记错"];

// 五年级数学知识点（人教版 2026 新教材，五年级上册），输入时自动联想
const MATH_KNOWLEDGE_POINTS = [
    "观察简单组合体", "三视图辨认",
    // 第2单元 小数乘法
    "小数乘法", "小数乘整数", "小数乘小数", "积的近似数", "运算定律推广到小数",
    // 第3单元 小数除法
    "小数除法", "除数是整数的小数除法", "一个数除以小数",
    "商的近似数", "循环小数", "用估算解决问题",
    // 第4单元 图形的运动
    "图形的运动", "平移", "旋转", "轴对称",
    // 第5单元 用字母表示数和数量关系
    "用字母表示数", "数量关系", "等式的性质",
    // 第6单元 多边形的面积
    "多边形的面积", "平行四边形的面积", "三角形的面积",
    "梯形的面积", "组合图形的面积", "公顷和平方千米",
    // 第7单元 有趣的密铺
    "密铺",
    // 第8单元 可能性
    "可能性",
];

const photoInput = document.getElementById("photoInput");
const pickPhotoBtn = document.getElementById("pickPhotoBtn");
const photoPreview = document.getElementById("photoPreview");
const previewImg = document.getElementById("previewImg");
const kpList = document.getElementById("kpList");
const submitBtn = document.getElementById("submitBtn");
const uploadMsg = document.getElementById("uploadMsg");
const questionList = document.getElementById("questionList");
const emptyHint = document.getElementById("emptyHint");

let editingId = null;

// AI 分析队列状态条
const analyzeStatus = document.getElementById("analyzeStatus");
let pendingAnalyses = 0;
let statusTimer = null;
let statusStart = 0;

function statusTick() {
    if (pendingAnalyses <= 0) {
        analyzeStatus.hidden = true;
        clearInterval(statusTimer);
        statusTimer = null;
        return;
    }
    const secs = Math.floor((Date.now() - statusStart) / 1000);
    analyzeStatus.textContent = `AI 正在分析错题…（队列剩余 ${pendingAnalyses} 道，已等待 ${secs} 秒，请稍候）`;
}

function beginAnalyze() {
    pendingAnalyses++;
    if (!statusTimer) {
        statusStart = Date.now();
        analyzeStatus.hidden = false;
        statusTick();
        statusTimer = setInterval(statusTick, 1000);
    } else {
        statusTick();
    }
}

function endAnalyze() {
    pendingAnalyses = Math.max(0, pendingAnalyses - 1);
    statusTick();
}

function showMsg(text, ok) {
    uploadMsg.textContent = text;
    uploadMsg.className = "msg " + (ok ? "ok" : "err");
    uploadMsg.hidden = false;
    setTimeout(() => (uploadMsg.hidden = true), 4000);
}

// 知识点候选列表，供编辑时输入联想（含错误原因）
function initKpList() {
    kpList.innerHTML = "";
    [...ERROR_REASONS, ...MATH_KNOWLEDGE_POINTS].forEach((kp) => {
        const opt = document.createElement("option");
        opt.value = kp;
        kpList.appendChild(opt);
    });
}

function formatTime(iso) {
    const d = new Date(iso);
    return d.toLocaleString("zh-CN", { hour12: false });
}

// AI 思考过程弹窗
const thinkingModal = document.getElementById("thinkingModal");
const thinkingText = document.getElementById("thinkingText");
const thinkingCloseBtn = document.getElementById("thinkingCloseBtn");

function openThinking() {
    thinkingText.textContent = "";
    thinkingCloseBtn.hidden = true;
    thinkingModal.hidden = false;
}

function appendThinking(text) {
    // 防止过长撑爆页面，思考过程最多显示 8000 字
    if (thinkingText.textContent.length > 8000) return;
    thinkingText.textContent += text;
    thinkingText.scrollTop = thinkingText.scrollHeight;
}

function finishThinking(ok) {
    appendThinking(ok ? "\n\n✅ 分析完成！" : "\n\n❌ 分析失败");
    thinkingCloseBtn.hidden = false;
}

thinkingCloseBtn.onclick = () => (thinkingModal.hidden = true);

function analyzeQuestion(id) {
    beginAnalyze();
    openThinking();
    let sawReasoning = false;
    let gotEvent = false;
    const waitHint = setTimeout(() => {
        if (!gotEvent) {
            appendThinking("正在连接模型…（思考型模型推理较慢，通常需要 2~6 分钟）\n\n");
        }
    }, 8000);
    return fetch(`/api/questions/${id}/analyze/stream`, { method: "POST" })
        .then((r) => {
            if (!r.ok || !r.body) throw new Error("分析请求失败");
            const reader = r.body.getReader();
            const decoder = new TextDecoder();
            let buf = "";
            const pump = () =>
                reader.read().then(({ done, value }) => {
                    if (done) return null;
                    gotEvent = true;
                    buf += decoder.decode(value, { stream: true });
                    const lines = buf.split("\n");
                    buf = lines.pop();
                    for (const line of lines) {
                        if (!line.startsWith("data:")) continue;
                        let evt;
                        try {
                            evt = JSON.parse(line.slice(5).trim());
                        } catch (e) {
                            continue;
                        }
                        if (evt.type === "reasoning") {
                            sawReasoning = true;
                            appendThinking(evt.text);
                        } else if (evt.type === "content") {
                            if (!sawReasoning) appendThinking(evt.text);
                        } else if (evt.type === "done") {
                            finishThinking(true);
                            return evt.question;
                        } else if (evt.type === "error") {
                            finishThinking(false);
                            throw new Error(evt.message);
                        }
                    }
                    return pump();
                });
            return pump();
        })
        .finally(() => {
            clearTimeout(waitHint);
            endAnalyze();
        });
}

function aiSection(q) {
    const box = document.createElement("div");
    box.className = "ai-box";

    const add = (label, text) => {
        if (!text) return;
        const row = document.createElement("div");
        row.className = "ai-row";
        const lab = document.createElement("span");
        lab.className = "ai-label";
        lab.textContent = label;
        const val = document.createElement("div");
        val.className = "ai-text";
        val.textContent = text.replace(/\\n/g, "\n");
        row.appendChild(lab);
        row.appendChild(val);
        box.appendChild(row);
    };

    add("题目", q.question_text);
    add("孩子的作答", q.child_answer);
    add("错因", q.error_analysis);
    add("正确答案", q.correct_answer);
    add("讲解", q.explanation);

    let similar = [];
    try {
        similar = JSON.parse(q.similar_questions || "[]");
    } catch (e) {
        similar = [];
    }
    if (similar.length) {
        const det = document.createElement("details");
        det.className = "ai-similar";
        const sum = document.createElement("summary");
        sum.textContent = `举一反三（${similar.length} 道类似题）`;
        det.appendChild(sum);
        similar.forEach((s, i) => {
            const item = document.createElement("div");
            item.className = "ai-text";
            item.textContent = `${i + 1}. ${(s["题目"] || "").replace(/\\n/g, "\n")}`;
            det.appendChild(item);
            const figVal = s["图"];
            if (typeof figVal === "string" && figVal) {
                const fig = document.createElement("pre");
                fig.className = "ai-fig";
                fig.textContent = figVal.replace(/\\n/g, "\n");
                det.appendChild(fig);
            } else if (figVal && figVal["立体"]) {
                const img = document.createElement("img");
                img.className = "ai-fig-img";
                img.src = "/api/stack-figure?h=" + encodeURIComponent(JSON.stringify(figVal["立体"]));
                img.alt = "立体图形";
                det.appendChild(img);
            } else if (figVal && figVal["字符画"]) {
                const fig = document.createElement("pre");
                fig.className = "ai-fig";
                fig.textContent = figVal["字符画"].replace(/\\n/g, "\n");
                det.appendChild(fig);
            }
            if (s["答案"]) {
                const ans = document.createElement("div");
                ans.className = "ai-answer";
                ans.textContent = `答案：${s["答案"]}`;
                det.appendChild(ans);
            }
        });
        box.appendChild(det);
    }
    return box;
}

function renderList() {
    fetch("/api/questions")
        .then((r) => r.json())
        .then((list) => {
            questionList.innerHTML = "";
            emptyHint.hidden = list.length > 0;
            list.forEach((q) => {
                const card = document.createElement("div");
                card.className = "q-card";

                const img = document.createElement("img");
                img.src = q.image_url;
                img.alt = "错题照片";
                img.onclick = () => {
                    document.getElementById("lightboxImg").src = q.image_url;
                    document.getElementById("lightbox").hidden = false;
                };

                const body = document.createElement("div");
                body.className = "q-card-body";
                if (q.knowledge_point) {
                    const tags = document.createElement("div");
                    tags.className = "q-tags";
                    const tagKp = document.createElement("span");
                    tagKp.className = "tag tag-kp";
                    tagKp.textContent = q.knowledge_point;
                    tags.appendChild(tagKp);
                    body.appendChild(tags);
                }
                if (q.note) {
                    const noteEl = document.createElement("div");
                    noteEl.className = "q-note";
                    noteEl.textContent = q.note;
                    body.appendChild(noteEl);
                }
                if (q.analyzed) {
                    body.appendChild(aiSection(q));
                }
                const time = document.createElement("div");
                time.className = "q-time";
                time.textContent = formatTime(q.created_at);
                body.appendChild(time);

                const actions = document.createElement("div");
                actions.className = "q-actions";
                const analyzeBtn = document.createElement("button");
                analyzeBtn.className = "btn btn-small";
                analyzeBtn.textContent = q.analyzed ? "重新分析" : "分析";
                analyzeBtn.onclick = () => {
                    analyzeBtn.disabled = true;
                    analyzeBtn.textContent = "分析中…";
                    analyzeQuestion(q.id)
                        .then(() => renderList())
                        .catch((e) => {
                            alert(e.message);
                            renderList();
                        });
                };
                actions.appendChild(analyzeBtn);
                const editBtn = document.createElement("button");
                editBtn.className = "btn btn-small";
                editBtn.textContent = "编辑";
                editBtn.onclick = () => openEdit(q);
                const delBtn = document.createElement("button");
                delBtn.className = "btn btn-small";
                delBtn.textContent = "删除";
                delBtn.onclick = () => deleteQuestion(q.id);
                actions.appendChild(editBtn);
                actions.appendChild(delBtn);

                card.appendChild(img);
                card.appendChild(body);
                card.appendChild(actions);
                questionList.appendChild(card);
            });
        });
}

function openEdit(q) {
    editingId = q.id;
    document.getElementById("editKnowledgePoint").value = q.knowledge_point;
    document.getElementById("editNote").value = q.note;
    document.getElementById("editModal").hidden = false;
}

function deleteQuestion(id) {
    if (!confirm("确定删除这道错题吗？照片也会一并删除。")) return;
    fetch("/api/questions/" + id, { method: "DELETE" }).then((r) => {
        if (r.ok) {
            renderList();
        } else {
            alert("删除失败");
        }
    });
}

pickPhotoBtn.onclick = () => photoInput.click();
photoInput.onchange = () => {
    if (!photoInput.files.length) return;
    previewImg.src = URL.createObjectURL(photoInput.files[0]);
    photoPreview.hidden = false;
};

submitBtn.onclick = () => {
    if (!photoInput.files.length) {
        showMsg("请先拍照或选择照片", false);
        return;
    }
    submitBtn.disabled = true;
    const fd = new FormData();
    fd.append("subject", "数学");
    fd.append("photo", photoInput.files[0]);

    fetch("/api/questions", { method: "POST", body: fd })
        .then(async (r) => {
            if (!r.ok) {
                const err = await r.json().catch(() => ({}));
                throw new Error(err.detail || "保存失败");
            }
            photoInput.value = "";
            photoPreview.hidden = true;
            renderList();
            // 保存成功后自动进行 AI 分析
            const created = await r.json().catch(() => null);
            if (created && created.id) {
                showMsg("已保存，AI 分析中…", true);
                try {
                    await analyzeQuestion(created.id);
                    showMsg("AI 分析完成！", true);
                } catch (e) {
                    showMsg(e.message, false);
                }
                renderList();
            }
        })
        .catch((e) => showMsg(e.message, false))
        .finally(() => (submitBtn.disabled = false));
};

document.getElementById("editSaveBtn").onclick = () => {
    fetch("/api/questions/" + editingId, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            knowledge_point: document.getElementById("editKnowledgePoint").value,
            note: document.getElementById("editNote").value,
        }),
    }).then((r) => {
        if (r.ok) {
            document.getElementById("editModal").hidden = true;
            renderList();
        } else {
            alert("保存失败");
        }
    });
};
document.getElementById("editCancelBtn").onclick = () =>
    (document.getElementById("editModal").hidden = true);
document.getElementById("lightboxCloseBtn").onclick = () =>
    (document.getElementById("lightbox").hidden = true);
document.getElementById("lightbox").onclick = (e) => {
    if (e.target.id === "lightbox") e.target.hidden = true;
};

initKpList();
renderList();
