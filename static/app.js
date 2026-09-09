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
    const box = document.createElement("details");
    box.className = "ai-box";
    const summary = document.createElement("summary");
    const errorText = q.error_analysis || "";
    const errorPreview = errorText.replace(/\s+/g, " ").slice(0, 24);
    summary.textContent = errorPreview ? `查看分析 · ${errorPreview}${errorText.length > 24 ? "…" : ""}` : "查看分析";
    box.appendChild(summary);

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

    let redo = {};
    try {
        redo = JSON.parse(q.original_redo || "{}");
    } catch (e) {
        redo = {};
    }
    if (redo["题目"]) {
        const det = document.createElement("details");
        det.className = "ai-similar";
        const sum = document.createElement("summary");
        sum.textContent = "生成的干净版原题";
        det.appendChild(sum);
        const text = document.createElement("div");
        text.className = "ai-text";
        text.textContent = redo["题目"].replace(/\\n/g, "\n");
        det.appendChild(text);
        const figVal = redo["图"];
        if (typeof figVal === "string" && figVal) {
            const fig = document.createElement("pre");
            fig.className = "ai-fig";
            fig.textContent = figVal.replace(/\\n/g, "\n");
            det.appendChild(fig);
        } else if (figVal && figVal["立体"]) {
            const img = document.createElement("img");
            img.className = "ai-fig-img";
            img.src = "/api/stack-figure?h=" + encodeURIComponent(JSON.stringify(figVal["立体"]));
            img.alt = "干净版原题立体图";
            det.appendChild(img);
        }
        box.appendChild(det);
    }

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
            const groups = new Map();
            list.forEach((q) => {
                const key = q.source_image_url || `single-${q.id}`;
                if (!groups.has(key)) groups.set(key, []);
                groups.get(key).push(q);
            });
            groups.forEach((items) => {
                const representative = items[0];
                const group = document.createElement("section");
                group.className = "worksheet-group";
                const header = document.createElement("div");
                header.className = "worksheet-header";
                const heading = document.createElement("div");
                const title = document.createElement("h3");
                title.textContent = representative.source_image_url
                    ? `一张原卷 · ${items.length} 道错题`
                    : "单题照片";
                const dates = items.map((item) => new Date(item.created_at));
                const date = new Date(Math.min(...dates));
                const meta = document.createElement("span");
                meta.textContent = date.toLocaleString("zh-CN", { hour12: false });
                heading.appendChild(title);
                heading.appendChild(meta);

                const groupActions = document.createElement("div");
                groupActions.className = "worksheet-actions";
                if (representative.source_image_url) {
                    const sourceBtn = document.createElement("button");
                    sourceBtn.className = "btn btn-small";
                    sourceBtn.textContent = "查看原卷";
                    sourceBtn.onclick = () => openLightbox(representative.source_image_url);
                    groupActions.appendChild(sourceBtn);
                }
                const recorrectBtn = document.createElement("button");
                recorrectBtn.className = "btn btn-small";
                recorrectBtn.textContent = "校正并切割";
                recorrectBtn.onclick = () => recorrectPage(representative, recorrectBtn);
                groupActions.appendChild(recorrectBtn);
                header.appendChild(heading);
                header.appendChild(groupActions);
                group.appendChild(header);

                const grid = document.createElement("div");
                grid.className = "worksheet-question-grid";
                items.forEach((q, itemIndex) => {
                const card = document.createElement("div");
                card.className = "q-card";

                const img = document.createElement("img");
                img.src = q.image_url;
                img.alt = `错题 ${itemIndex + 1}`;
                img.onclick = () => openLightbox(q.image_url);

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
                grid.appendChild(card);
                });
                group.appendChild(grid);
                questionList.appendChild(group);
            });
        });
}

function openLightbox(imageUrl) {
    document.getElementById("lightboxImg").src = imageUrl;
    document.getElementById("lightbox").hidden = false;
}

const manualCorrectModal = document.getElementById("manualCorrectModal");
const manualCorrectCanvas = document.getElementById("manualCorrectCanvas");
const manualCorrectHint = document.getElementById("manualCorrectHint");
const manualCorrectApplyBtn = document.getElementById("manualCorrectApplyBtn");
const manualCorrectContext = manualCorrectCanvas.getContext("2d");
const cornerNames = ["左上角", "右上角", "右下角", "左下角"];
let manualCorrectQuestion = null;
let manualCorrectImage = null;
let manualCorrectPoints = [];

function redrawManualCorrect() {
    if (!manualCorrectImage) return;
    manualCorrectContext.clearRect(0, 0, manualCorrectCanvas.width, manualCorrectCanvas.height);
    manualCorrectContext.drawImage(
        manualCorrectImage, 0, 0, manualCorrectCanvas.width, manualCorrectCanvas.height
    );
    if (manualCorrectPoints.length) {
        manualCorrectContext.strokeStyle = "#f44336";
        manualCorrectContext.fillStyle = "#f44336";
        manualCorrectContext.lineWidth = Math.max(3, manualCorrectCanvas.width / 300);
        manualCorrectContext.beginPath();
        manualCorrectPoints.forEach((point, index) => {
            if (index === 0) manualCorrectContext.moveTo(point[0], point[1]);
            else manualCorrectContext.lineTo(point[0], point[1]);
        });
        if (manualCorrectPoints.length === 4) manualCorrectContext.closePath();
        manualCorrectContext.stroke();
        manualCorrectPoints.forEach((point, index) => {
            manualCorrectContext.beginPath();
            manualCorrectContext.arc(
                point[0], point[1], Math.max(8, manualCorrectCanvas.width / 90), 0, Math.PI * 2
            );
            manualCorrectContext.fill();
            manualCorrectContext.fillStyle = "#fff";
            manualCorrectContext.font = `bold ${Math.max(14, manualCorrectCanvas.width / 45)}px sans-serif`;
            manualCorrectContext.textAlign = "center";
            manualCorrectContext.textBaseline = "middle";
            manualCorrectContext.fillText(String(index + 1), point[0], point[1]);
            manualCorrectContext.fillStyle = "#f44336";
        });
    }
    manualCorrectApplyBtn.disabled = manualCorrectPoints.length !== 4;
    manualCorrectHint.textContent = manualCorrectPoints.length < 4
        ? `请点击${cornerNames[manualCorrectPoints.length]}（第 ${manualCorrectPoints.length + 1} 个点）`
        : "四个角已选好；红线内的区域将被拉成矩形";
}

function openManualCorrect(q) {
    manualCorrectQuestion = q;
    manualCorrectPoints = [];
    manualCorrectImage = new Image();
    manualCorrectImage.onload = () => {
        manualCorrectCanvas.width = manualCorrectImage.naturalWidth;
        manualCorrectCanvas.height = manualCorrectImage.naturalHeight;
        redrawManualCorrect();
    };
    manualCorrectImage.onerror = () => {
        showMsg("无法载入原卷图片", false);
        manualCorrectModal.hidden = true;
    };
    manualCorrectImage.src = q.source_image_url || q.image_url;
    manualCorrectHint.textContent = "正在载入原卷…";
    manualCorrectModal.hidden = false;
}

manualCorrectCanvas.onclick = (event) => {
    if (!manualCorrectImage || manualCorrectPoints.length >= 4) return;
    const rect = manualCorrectCanvas.getBoundingClientRect();
    manualCorrectPoints.push([
        (event.clientX - rect.left) * manualCorrectCanvas.width / rect.width,
        (event.clientY - rect.top) * manualCorrectCanvas.height / rect.height,
    ]);
    redrawManualCorrect();
};

document.getElementById("manualCorrectResetBtn").onclick = () => {
    manualCorrectPoints = [];
    redrawManualCorrect();
};
document.getElementById("manualCorrectCancelBtn").onclick = () => {
    manualCorrectModal.hidden = true;
};
manualCorrectApplyBtn.onclick = async () => {
    if (!manualCorrectQuestion || manualCorrectPoints.length !== 4) return;
    manualCorrectApplyBtn.disabled = true;
    manualCorrectApplyBtn.textContent = "校正中…";
    try {
        const response = await fetch(
            `/api/questions/${manualCorrectQuestion.id}/recorrect/preview`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ points: manualCorrectPoints }),
            }
        );
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.detail || "手动校正失败");
        manualCorrectModal.hidden = true;
        openCropReview(result, manualCorrectQuestion);
    } catch (error) {
        showMsg(error.message, false);
    } finally {
        manualCorrectApplyBtn.textContent = "确认放平";
        manualCorrectApplyBtn.disabled = manualCorrectPoints.length !== 4;
    }
};

const cropReviewModal = document.getElementById("cropReviewModal");
const cropReviewCanvas = document.getElementById("cropReviewCanvas");
const cropReviewContext = cropReviewCanvas.getContext("2d");
const cropRangeList = document.getElementById("cropRangeList");
const cropConfirmBtn = document.getElementById("cropConfirmBtn");
let cropReviewQuestion = null;
let cropReviewImage = null;
let cropReviewFilename = "";
let cropRanges = [];
let cropDragStart = null;

function drawCropReview() {
    if (!cropReviewImage) return;
    cropReviewContext.clearRect(0, 0, cropReviewCanvas.width, cropReviewCanvas.height);
    cropReviewContext.drawImage(
        cropReviewImage, 0, 0, cropReviewCanvas.width, cropReviewCanvas.height
    );
    cropRanges.forEach((range, index) => {
        const hue = (index * 67) % 360;
        cropReviewContext.fillStyle = `hsla(${hue}, 90%, 55%, 0.09)`;
        cropReviewContext.strokeStyle = `hsl(${hue}, 85%, 45%)`;
        cropReviewContext.lineWidth = Math.max(3, cropReviewCanvas.width / 350);
        cropReviewContext.fillRect(0, range[0], cropReviewCanvas.width, range[1] - range[0]);
        cropReviewContext.strokeRect(1, range[0], cropReviewCanvas.width - 2, range[1] - range[0]);
    });
    cropRangeList.innerHTML = "";
    cropRanges.forEach((range, index) => {
        const chip = document.createElement("button");
        chip.className = "crop-range-chip";
        chip.textContent = `错题 ${index + 1} ×`;
        chip.title = "删除这个候选范围";
        chip.onclick = () => {
            cropRanges.splice(index, 1);
            drawCropReview();
        };
        cropRangeList.appendChild(chip);
    });
    cropConfirmBtn.disabled = cropRanges.length === 0;
}

function cropCanvasY(event) {
    const rect = cropReviewCanvas.getBoundingClientRect();
    return Math.max(
        0,
        Math.min(
            cropReviewCanvas.height,
            (event.clientY - rect.top) * cropReviewCanvas.height / rect.height
        )
    );
}

function openCropReview(result, question) {
    cropReviewQuestion = question;
    cropReviewFilename = result.preview_filename;
    cropRanges = (result.ranges || []).map((range) => [range[0], range[1]]);
    cropReviewImage = new Image();
    cropReviewImage.onload = () => {
        cropReviewCanvas.width = cropReviewImage.naturalWidth;
        cropReviewCanvas.height = cropReviewImage.naturalHeight;
        drawCropReview();
    };
    cropReviewImage.onerror = () => showMsg("无法载入放平预览", false);
    cropReviewImage.src = result.preview_url;
    cropReviewModal.hidden = false;
}

cropReviewCanvas.onpointerdown = (event) => {
    cropDragStart = cropCanvasY(event);
    cropReviewCanvas.setPointerCapture(event.pointerId);
};
cropReviewCanvas.onpointerup = (event) => {
    if (cropDragStart === null) return;
    const end = cropCanvasY(event);
    const top = Math.round(Math.min(cropDragStart, end));
    const bottom = Math.round(Math.max(cropDragStart, end));
    cropDragStart = null;
    if (bottom - top < 80) {
        showMsg("框选范围太小，请从题目上方向下拖到题目底部", false);
        return;
    }
    cropRanges.push([top, bottom]);
    cropRanges.sort((a, b) => a[0] - b[0]);
    drawCropReview();
};

document.getElementById("cropClearBtn").onclick = () => {
    cropRanges = [];
    drawCropReview();
};

function closeCropReview() {
    cropReviewModal.hidden = true;
    if (cropReviewFilename) {
        fetch(`/api/recorrect/preview/${encodeURIComponent(cropReviewFilename)}`, {
            method: "DELETE",
        });
    }
    cropReviewFilename = "";
}

document.getElementById("cropCancelBtn").onclick = closeCropReview;
cropConfirmBtn.onclick = async () => {
    if (!cropReviewQuestion || !cropRanges.length) return;
    cropConfirmBtn.disabled = true;
    cropConfirmBtn.textContent = "保存中…";
    try {
        const response = await fetch(
            `/api/questions/${cropReviewQuestion.id}/recorrect/confirm`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    preview_filename: cropReviewFilename,
                    ranges: cropRanges,
                }),
            }
        );
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.detail || "保存切割结果失败");
        cropReviewFilename = "";
        cropReviewModal.hidden = true;
        renderList();
        showMsg(`已按确认范围保存 ${result.questions.length} 道错题`, true);
    } catch (error) {
        showMsg(error.message, false);
    } finally {
        cropConfirmBtn.textContent = "确认保存";
        cropConfirmBtn.disabled = cropRanges.length === 0;
    }
};

async function recorrectPage(q, button) {
    button.disabled = true;
    button.textContent = "生成预览…";
    try {
        const response = await fetch(
            `/api/questions/${q.id}/recorrect/preview`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ points: null }),
            }
        );
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.detail || "生成校正预览失败");
        if (!result.page_detected) {
            fetch(`/api/recorrect/preview/${encodeURIComponent(result.preview_filename)}`, {
                method: "DELETE",
            });
            showMsg("没有可靠识别出纸张四角，请在原图上手动选择四角", false);
            openManualCorrect(q);
            return;
        }
        openCropReview(result, q);
    } catch (error) {
        showMsg(error.message, false);
    } finally {
        button.disabled = false;
        button.textContent = "校正并切割";
    }
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

    fetch("/api/pages", { method: "POST", body: fd })
        .then(async (r) => {
            if (!r.ok) {
                const err = await r.json().catch(() => ({}));
                throw new Error(err.detail || "保存失败");
            }
            photoInput.value = "";
            photoPreview.hidden = true;
            const result = await r.json();
            const created = result.questions || [];
            renderList();
            if (created.length) {
                const detection = result.red_mark_count
                    ? `检测到 ${result.red_mark_count} 组明显红叉/圈叉，已切成 ${created.length} 道候选错题`
                    : "未检测到红色批改，已按完整照片保存";
                showMsg(detection + "，AI 将逐题分析…", true);
                for (const question of created) {
                    try {
                        await analyzeQuestion(question.id);
                    } catch (e) {
                        showMsg(`第 ${question.id} 题分析失败：${e.message}`, false);
                    }
                    renderList();
                }
                showMsg(`已完成 ${created.length} 道候选错题的处理`, true);
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
