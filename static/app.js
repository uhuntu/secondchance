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
const DRAFT_KEY = "secondchance-draft";
let questionsCache = [];
let libraryOpen = false;
let draft = { candidates: [], selected: [] };
try {
    // Fall back to the pre-rename key so an in-progress session survives the rename.
    const saved = JSON.parse(sessionStorage.getItem(DRAFT_KEY) ?? sessionStorage.getItem("seconddraft-draft"));
    if (saved && Array.isArray(saved.candidates) && Array.isArray(saved.selected)) draft = saved;
} catch (_) { /* Storage may be unavailable; the current session still works. */ }
let candidateIds = new Set(draft.candidates);
let selectedIds = new Set(draft.selected);

function updateSelection() {
    document.getElementById("selectionCount").textContent = `已选 ${selectedIds.size} 道题`;
    document.getElementById("printSelectedBtn").disabled = !selectedIds.size;
    try {
        sessionStorage.setItem(DRAFT_KEY, JSON.stringify({ candidates: [...candidateIds], selected: [...selectedIds] }));
    } catch (_) { /* Keep the in-memory draft. */ }
}

function replaceDraftGroup(previous, replacements) {
    const old = questionsCache.filter(q => q.source_image_url === previous.source_image_url);
    const inDraft = old.some(q => candidateIds.has(q.id));
    old.forEach(q => { candidateIds.delete(q.id); selectedIds.delete(q.id); });
    if (inDraft) replacements.forEach(q => candidateIds.add(q.id));
    updateSelection();
}

document.getElementById("libraryBtn").onclick = () => { libraryOpen = !libraryOpen; renderList(); };
document.getElementById("clearSelectionBtn").onclick = () => {
    candidateIds.clear(); selectedIds.clear(); libraryOpen = false; renderList();
};
document.getElementById("printSelectedBtn").onclick = () => {
    if (selectedIds.size) window.open(`/static/paper.html?ids=${[...selectedIds].join(",")}`, "_blank");
};
let previewObjectUrl = null;  // 预览用的 blob URL，必须及时 revoke，否则每选一张照片都泄漏一份

function showMsg(text, ok) {
    uploadMsg.textContent = text;
    uploadMsg.className = "msg " + (ok ? "ok" : "err");
    uploadMsg.hidden = false;
    clearTimeout(showMsg.timer);
    showMsg.timer = setTimeout(() => (uploadMsg.hidden = true), 4000);
}

function initKpList() {
    kpList.innerHTML = "";
    [...ERROR_REASONS, ...MATH_KNOWLEDGE_POINTS].forEach((kp) => {
        const opt = document.createElement("option");
        opt.value = kp;
        kpList.appendChild(opt);
    });
}

// ---------- 列表 ----------

function renderList(list) {
    if (list) questionsCache = list;
    const existing = new Set(questionsCache.map(q => q.id));
    candidateIds = new Set([...candidateIds].filter(id => existing.has(id)));
    selectedIds = new Set([...selectedIds].filter(id => existing.has(id)));
    updateSelection();
    document.getElementById("listTitle").textContent = libraryOpen ? "从以前的题里选" : "本次重练";
    document.getElementById("libraryBtn").textContent = libraryOpen ? "返回本次重练" : "从以前的题里选";
    const visible = libraryOpen ? questionsCache : questionsCache.filter(q => candidateIds.has(q.id));
    questionList.innerHTML = "";
    emptyHint.hidden = visible.length > 0;

    const groups = new Map();
    visible.forEach((q) => {
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
            ? `${items.length} 道题`
            : "单题照片";
        const dates = items.map((item) => new Date(item.created_at));
        const meta = document.createElement("span");
        meta.textContent = new Date(Math.min(...dates)).toLocaleString("zh-CN", { hour12: false });
        heading.append(title, meta);

        const actions = document.createElement("div");
        actions.className = "worksheet-actions";
        if (representative.source_image_url) {
            actions.append(
                smallButton("看整页", () => openLightbox(representative.source_image_url)),
                smallButton("AI 识别错题", () => judgePage(representative)),
                smallButton(`删除整页（${items.length} 题）`, () => deletePage(items))
            );
        }
        header.append(heading, actions);
        group.appendChild(header);

        const grid = document.createElement("div");
        grid.className = "worksheet-question-grid";
        items.forEach((q, index) => grid.appendChild(buildCard(q, index + 1)));
        group.appendChild(grid);
        questionList.appendChild(group);
    });
}

function smallButton(text, onclick) {
    const btn = document.createElement("button");
    btn.className = "btn btn-small";
    btn.textContent = text;
    btn.onclick = onclick;
    return btn;
}

function buildCard(q, index) {
    const card = document.createElement("div");
    card.className = "q-card";

    const img = document.createElement("img");
    img.src = q.image_url;
    img.alt = `错题 ${index}`;
    img.onclick = () => openLightbox(q.image_url);

    const body = document.createElement("div");
    body.className = "q-card-body";

    const choice = document.createElement("label");
    choice.className = "question-choice";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedIds.has(q.id);
    card.classList.toggle("selected", checkbox.checked);
    checkbox.onchange = () => {
        candidateIds.add(q.id);
        if (checkbox.checked) selectedIds.add(q.id); else selectedIds.delete(q.id);
        card.classList.toggle("selected", checkbox.checked);
        updateSelection();
    };
    choice.append(checkbox, document.createTextNode(`第 ${index} 题 · 加入本次重练`));
    if (q.suspected_wrong) {
        const badge = document.createElement("span");
        badge.className = "suspect-badge";
        badge.textContent = "疑似错题";
        badge.title = "AI 根据批改痕迹判断，仅供参考；做对了就取消勾选";
        choice.append(badge);
    }
    if (q.practice_count > 0) {
        const practiced = document.createElement("span");
        practiced.className = "practice-meta";
        practiced.textContent = `练过 ${q.practice_count} 次`;
        practiced.title = q.last_practiced_at
            ? `上次练习：${new Date(q.last_practiced_at).toLocaleDateString("zh-CN", { month: "long", day: "numeric" })}`
            : "上次练习时间未知";
        choice.append(practiced);
    }
    body.appendChild(choice);

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "修正 / 备注";
    details.appendChild(summary);
    const kp = document.createElement("input");
    kp.type = "text";
    kp.className = "kp-inline";
    kp.list = "kpList";
    kp.placeholder = "＋ 知识点 / 错因";
    kp.value = q.knowledge_point || "";
    kp.onclick = (e) => e.stopPropagation();
    kp.onchange = () => saveField(q, "knowledge_point", kp.value.trim(), kp);
    details.appendChild(kp);

    if (q.note) {
        const noteEl = document.createElement("div");
        noteEl.className = "q-note";
        noteEl.textContent = q.note;
        details.appendChild(noteEl);
    }

    const actions = document.createElement("div");
    actions.className = "q-actions";
    let savedMasks = [];
    try {
        savedMasks = JSON.parse(q.answer_masks || "[]");
    } catch (e) {
        savedMasks = [];
    }
    actions.append(
        smallButton(savedMasks.length ? `遮答案 ${savedMasks.length}` : "遮答案", () => openMask(q)),
        smallButton("备注", () => openEdit(q)),
        smallButton("删除", () => deleteQuestion(q.id))
    );
    if (q.source_image_url) actions.append(smallButton("调整整页切题", () => openSplit(q)));
    details.appendChild(actions);
    body.appendChild(details);
    card.append(img, body);
    return card;
}

function saveField(q, field, value, inputEl) {
    const previous = q[field] || "";
    if (value === previous) return;
    inputEl.classList.add("saving");
    fetch(`/api/questions/${q.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [field]: value }),
    })
        .then((r) => {
            inputEl.classList.remove("saving");
            if (!r.ok) {
                inputEl.value = previous;
                showMsg("保存失败", false);
                return;
            }
            q[field] = value;
        })
        .catch(() => {
            inputEl.classList.remove("saving");
            inputEl.value = previous;
            showMsg("保存失败", false);
        });
}

function openLightbox(imageUrl) {
    document.getElementById("lightboxImg").src = imageUrl;
    document.getElementById("lightbox").hidden = false;
}

function openEdit(q) {
    editingId = q.id;
    document.getElementById("editKnowledgePoint").value = q.knowledge_point;
    document.getElementById("editNote").value = q.note;
    document.getElementById("editModal").hidden = false;
}

function deleteQuestion(id) {
    if (!confirm("确定删除这道错题吗？照片也会一并删除。")) return;
    fetch("/api/questions/" + id, { method: "DELETE" })
        .then((r) => {
            if (r.ok) refresh();
            else showMsg("删除失败", false);
        })
        .catch(() => showMsg("删除失败", false));
}

function judgePage(representative) {
    fetch(`/api/questions/${representative.id}/judge`, { method: "POST" })
        .then((r) => r.json().catch(() => ({})))
        .then((data) => {
            if (!data.ai_active) {
                showMsg("AI 没参与（未配置 key 或网络失败），没有更新标记", false);
                return;
            }
            if (!data.updated) {
                showMsg("这一页的切法和 AI 识别的题数对不上，请在编辑器里重新自动切后再识别", false);
                return;
            }
            const tagged = data.tagged ? `，顺带给 ${data.tagged} 道题补了知识点` : "";
            showMsg(
                data.suspected
                    ? `识别出 ${data.suspected} 道疑似错题，已标记${tagged}`
                    : `AI 看了一遍：这页没有发现明显的错题${tagged}`,
                true,
            );
            refresh();
        })
        .catch(() => showMsg("识别失败", false));
}

function deletePage(items) {
    const count = items.length;
    if (!confirm(`确定删除这一页的全部 ${count} 道题吗？整页原图和所有切片会一并删除。`)) return;
    fetch(`/api/questions/${items[0].id}/page`, { method: "DELETE" })
        .then((r) => {
            if (r.ok) {
                showMsg(`已删除整页 ${count} 道题`, true);
                refresh();
            } else {
                showMsg("删除整页失败", false);
            }
        })
        .catch(() => showMsg("删除整页失败", false));
}

function refresh() {
    return fetch("/api/questions")
        .then((r) => { if (!r.ok) throw new Error("加载失败"); return r.json(); })
        .then(renderList)
        .catch(() => showMsg("加载错题列表失败", false));
}

// ---------- 分割线编辑器 ----------
//
// 划线代替画框：一道题 = 两条线之间的部分。
// 画框要从题的上边缘一路拖到下边缘（N 次长距离精确拖动）；
// 划线只在题目的空隙里划一下（N-1 次短距离滑动），精度要求低得多。

const splitModal = document.getElementById("splitModal");
const splitCanvas = document.getElementById("splitCanvas");
const splitContext = splitCanvas.getContext("2d");
const splitCount = document.getElementById("splitCount");
const splitLineChips = document.getElementById("splitLineChips");
const splitSaveBtn = document.getElementById("splitSaveBtn");

const LINE_HIT_PX = 22;      // 命中判定半径（屏幕像素，与图缩放无关）
const MIN_LINE_GAP_PX = 14;  // 两条线之间的最小间隔（屏幕像素）

let splitQuestionRef = null;
let splitImage = null;
let splitLines = [];
let splitSkipFirst = false;  // 第一段是页眉（AI 重新自动切后为真），保存时丢弃
let splitBoxes = [];         // AI 标注的手写痕迹框（整页坐标），随保存透传
let splitDragIndex = -1;
let splitDragMoved = false;
let splitChipsRendered = -1;

function splitCanvasY(event) {
    const rect = splitCanvas.getBoundingClientRect();
    return Math.max(0, Math.min(splitCanvas.height, (event.clientY - rect.top) * splitCanvas.height / rect.height));
}

// 画布是按原始像素绘制的，但显示时会被 CSS 缩小。抓手和命中半径必须换算成
// "屏幕像素 → 画布像素"，否则图一小，抓手就只剩几个像素，几乎抓不住。
function screenToCanvasScale() {
    const rect = splitCanvas.getBoundingClientRect();
    if (rect.height <= 0) return 1;
    return splitCanvas.height / rect.height;
}

function drawSplit() {
    if (!splitImage) return;
    const { width, height } = splitCanvas;
    const scale = screenToCanvasScale();
    const px = (screenPixels) => Math.max(2, screenPixels * scale);

    splitContext.clearRect(0, 0, width, height);
    splitContext.drawImage(splitImage, 0, 0, width, height);

    // 交替底色，让"第几道题"一眼可数；第一段是页眉时灰显并单独标注
    let top = 0;
    splitLines.concat([height]).forEach((bottom, index) => {
        if (bottom <= top) return;
        const isPreamble = index === 0 && splitSkipFirst;
        const questionNumber = splitSkipFirst ? index : index + 1;
        splitContext.fillStyle = isPreamble
            ? "rgba(120,120,120,0.18)"
            : index % 2 ? "rgba(74,108,247,0.10)" : "rgba(74,108,247,0.03)";
        splitContext.fillRect(0, top, width, bottom - top);
        const label = isPreamble ? "页眉" : `${questionNumber}`;
        splitContext.font = `bold ${px(18)}px sans-serif`;
        splitContext.textAlign = "center";
        splitContext.textBaseline = "middle";
        const cx = width / 2;
        const cy = (top + bottom) / 2;
        splitContext.fillStyle = isPreamble ? "rgba(90,90,90,0.75)" : "rgba(74,108,247,0.85)";
        splitContext.beginPath();
        splitContext.arc(cx, cy, px(16), 0, Math.PI * 2);
        splitContext.fill();
        splitContext.fillStyle = "#fff";
        splitContext.fillText(label, cx, cy);
        top = bottom;
    });

    // 分割线本身：一条横线 + 中间一个抓手
    splitLines.forEach((line, index) => {
        const active = index === splitDragIndex;
        splitContext.strokeStyle = active ? "#ff8f1f" : "#f44336";
        splitContext.lineWidth = px(active ? 4 : 2.5);
        splitContext.beginPath();
        splitContext.moveTo(0, line);
        splitContext.lineTo(width, line);
        splitContext.stroke();

        const cx = width / 2;
        const radius = px(17);
        splitContext.fillStyle = active ? "#ff8f1f" : "#f44336";
        splitContext.beginPath();
        splitContext.arc(cx, line, radius, 0, Math.PI * 2);
        splitContext.fill();
        splitContext.fillStyle = "#fff";
        splitContext.font = `bold ${px(15)}px sans-serif`;
        splitContext.textAlign = "center";
        splitContext.textBaseline = "middle";
        splitContext.fillText("↕", cx, line);
    });

    splitCount.textContent = `当前 ${splitLines.length + 1 - (splitSkipFirst ? 1 : 0)} 道题`;
    splitSaveBtn.disabled = false;
    // 拖动过程中不重建标签（会闪），只在条数变化或拖动结束时刷新
    if (splitDragIndex < 0 && splitChipsRendered !== splitLines.length) refreshLineChips();
}

function hitLine(event) {
    if (splitCanvas.getBoundingClientRect().height <= 0) return -1;
    const tolerance = LINE_HIT_PX * screenToCanvasScale();
    let best = -1;
    let bestDistance = Infinity;
    splitLines.forEach((line, index) => {
        const distance = Math.abs(splitCanvasY(event) - line);
        if (distance <= tolerance && distance < bestDistance) {
            best = index;
            bestDistance = distance;
        }
    });
    return best;
}

function openSplit(q) {
    splitQuestionRef = q;
    splitModal.hidden = false;
    splitCount.textContent = "正在载入整页原图…";
    fetch(`/api/questions/${q.id}/page`)
        .then(async (r) => {
            const data = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(data.detail || "读取整页原图失败");
            return data;
        })
        .then((data) => {
            // 上次保存的切法原样还原（含"第一段是页眉"的标记）
            splitSkipFirst = !!data.skip_first;
            splitBoxes = [];  // 旧切法没有 AI 手写框，重新自动切才会有
            splitLines = (data.lines || []).slice().sort((a, b) => a - b);
            const img = new Image();
            img.onload = () => {
                splitImage = img;
                splitCanvas.width = img.naturalWidth;
                splitCanvas.height = img.naturalHeight;
                drawSplit();
            };
            img.onerror = () => showMsg("无法载入整页原图", false);
            img.src = data.source_image_url;
        })
        .catch((error) => {
            splitModal.hidden = true;
            showMsg(error.message, false);
        });
}

splitCanvas.addEventListener("pointerdown", (event) => {
    const index = hitLine(event);
    if (index >= 0) {
        splitDragIndex = index;
        splitDragMoved = false;
        splitCanvas.setPointerCapture(event.pointerId);
        drawSplit();
        return;
    }
    // 点在线的空白处 = 在这里加一条线
    const minGap = MIN_LINE_GAP_PX * screenToCanvasScale();
    const y = Math.round(splitCanvasY(event));
    if (y < minGap || y > splitCanvas.height - minGap) return;
    if (splitLines.some((line) => Math.abs(line - y) < minGap)) return;
    if (splitLines.length >= 20) {
        showMsg("一页最多切 21 道题", false);
        return;
    }
    splitLines.push(y);
    splitLines.sort((a, b) => a - b);
    drawSplit();
});

splitCanvas.addEventListener("pointermove", (event) => {
    if (splitDragIndex < 0) return;
    splitDragMoved = true;
    const minGap = MIN_LINE_GAP_PX * screenToCanvasScale();
    const y = Math.round(splitCanvasY(event));
    splitLines[splitDragIndex] = Math.max(minGap, Math.min(splitCanvas.height - minGap, y));
    drawSplit();
});

splitCanvas.addEventListener("pointerup", () => {
    if (splitDragIndex < 0) return;
    const minGap = MIN_LINE_GAP_PX * screenToCanvasScale();
    splitLines.sort((a, b) => a - b);
    const deduped = splitLines.filter((line, index) => index === 0 || line - splitLines[index - 1] >= minGap);
    splitLines = deduped;
    splitDragIndex = -1;
    drawSplit();
});

// 分割线的小标签：删除必须点标签，避免在图上误点就把线弄没了。
function refreshLineChips() {
    splitLineChips.innerHTML = "";
    splitLines.forEach((line, index) => {
        const chip = document.createElement("button");
        chip.className = "line-chip";
        chip.textContent = `线 ${index + 1} ×`;
        chip.title = "删除这条分割线";
        chip.onclick = () => {
            if (index === 0 && splitSkipFirst) splitSkipFirst = false;  // 页眉并入第 1 题
            splitLines.splice(index, 1);
            drawSplit();
        };
        splitLineChips.appendChild(chip);
    });
    if (!splitLines.length) {
        const tip = document.createElement("span");
        tip.className = "line-chip-empty";
        tip.textContent = "还没有分割线，整页就是一道错题。在题目空隙点一下就能加线。";
        splitLineChips.appendChild(tip);
    }
    splitChipsRendered = splitLines.length;
}

function closeSplit() {
    splitModal.hidden = true;
    splitQuestionRef = null;
    splitImage = null;
    splitLines = [];
    splitBoxes = [];
    splitSkipFirst = false;
}

document.getElementById("splitCancelBtn").onclick = closeSplit;

document.getElementById("splitResetBtn").onclick = () => {
    if (!splitQuestionRef) return;
    const button = document.getElementById("splitResetBtn");
    button.disabled = true;
    fetch(`/api/questions/${splitQuestionRef.id}/suggest`, { method: "POST" })
        .then((r) => r.json().catch(() => ({})))
        .then((data) => {
            button.disabled = false;
            if (!data.lines) {
                showMsg(data.detail || "重新自动切分不可用", false);
                return;
            }
            splitSkipFirst = !!data.ai_active;
            splitBoxes = Array.isArray(data.boxes) ? data.boxes : [];
            splitLines = data.lines.slice().sort((a, b) => a - b);
            drawSplit();
            const count = splitLines.length + 1 - (splitSkipFirst ? 1 : 0);
            showMsg(
                count > 1
                    ? `已自动切成 ${count} 道${splitSkipFirst ? "（页眉不保存）" : ""}，可继续拖动调整`
                    : "这一页没找到明显的题间空隙，已按整页一道处理",
                true
            );
        })
        .catch(() => {
            button.disabled = false;
            showMsg("重新自动切分失败", false);
        });
};

splitSaveBtn.onclick = async () => {
    if (!splitQuestionRef) return;
    splitSaveBtn.disabled = true;
    splitSaveBtn.textContent = "保存中…";
    try {
        const response = await fetch(`/api/questions/${splitQuestionRef.id}/split`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lines: splitLines, skip_first: splitSkipFirst, boxes: splitBoxes }),
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.detail || "保存切割结果失败");
        replaceDraftGroup(splitQuestionRef, result.questions);
        closeSplit();
        refresh();
        showMsg(`已切成 ${result.questions.length} 道题，请重新勾选这一页要重练的题`, true);
    } catch (error) {
        showMsg(error.message, false);
    } finally {
        splitSaveBtn.textContent = "保存";
        splitSaveBtn.disabled = false;
    }
};

// ---------- 手动四点摆正 ----------

const cornerModal = document.getElementById("cornerModal");
const cornerCanvas = document.getElementById("cornerCanvas");
const cornerContext = cornerCanvas.getContext("2d");
const cornerHint = document.getElementById("cornerHint");
const cornerApplyBtn = document.getElementById("cornerApplyBtn");
const CORNER_NAMES = ["左上角", "右上角", "右下角", "左下角"];
let cornerPoints = [];
let cornerSourceUrl = "";

function drawCorners() {
    if (!splitImage) return;
    cornerContext.clearRect(0, 0, cornerCanvas.width, cornerCanvas.height);
    cornerContext.drawImage(splitImage, 0, 0, cornerCanvas.width, cornerCanvas.height);
    if (cornerPoints.length) {
        cornerContext.strokeStyle = "#f44336";
        cornerContext.fillStyle = "#f44336";
        cornerContext.lineWidth = Math.max(3, cornerCanvas.width / 300);
        cornerContext.beginPath();
        cornerPoints.forEach((point, index) => {
            if (index === 0) cornerContext.moveTo(point[0], point[1]);
            else cornerContext.lineTo(point[0], point[1]);
        });
        if (cornerPoints.length === 4) cornerContext.closePath();
        cornerContext.stroke();
        cornerPoints.forEach((point, index) => {
            cornerContext.beginPath();
            cornerContext.arc(point[0], point[1], Math.max(8, cornerCanvas.width / 90), 0, Math.PI * 2);
            cornerContext.fill();
            cornerContext.fillStyle = "#fff";
            cornerContext.font = `bold ${Math.max(14, cornerCanvas.width / 45)}px sans-serif`;
            cornerContext.textAlign = "center";
            cornerContext.textBaseline = "middle";
            cornerContext.fillText(String(index + 1), point[0], point[1]);
            cornerContext.fillStyle = "#f44336";
        });
    }
    cornerApplyBtn.disabled = cornerPoints.length !== 4;
    cornerHint.textContent =
        cornerPoints.length < 4
            ? `请点击${CORNER_NAMES[cornerPoints.length]}（第 ${cornerPoints.length + 1} 个点）`
            : "四个角已选好；红线内的区域将被拉成矩形";
}

document.getElementById("splitStraightenBtn").onclick = () => {
    if (!splitQuestionRef) return;
    cornerPoints = [];
    cornerCanvas.width = splitCanvas.width;
    cornerCanvas.height = splitCanvas.height;
    cornerContext.drawImage(splitImage, 0, 0);
    cornerHint.textContent = "正在载入…";
    cornerModal.hidden = false;
    drawCorners();
};

cornerCanvas.onclick = (event) => {
    if (cornerPoints.length >= 4) return;
    const rect = cornerCanvas.getBoundingClientRect();
    cornerPoints.push([
        (event.clientX - rect.left) * cornerCanvas.width / rect.width,
        (event.clientY - rect.top) * cornerCanvas.height / rect.height,
    ]);
    drawCorners();
};

document.getElementById("cornerResetBtn").onclick = () => {
    cornerPoints = [];
    drawCorners();
};
document.getElementById("cornerCancelBtn").onclick = () => (cornerModal.hidden = true);

cornerApplyBtn.onclick = async () => {
    if (!splitQuestionRef || cornerPoints.length !== 4) return;
    cornerApplyBtn.disabled = true;
    cornerApplyBtn.textContent = "摆正中…";
    try {
        const response = await fetch(`/api/questions/${splitQuestionRef.id}/straighten`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ points: cornerPoints, lines: [] }),
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.detail || "手动摆正失败");
        replaceDraftGroup(splitQuestionRef, result.questions);
        splitQuestionRef = result.questions[0];
        cornerModal.hidden = true;
        splitImage = null;
        const img = new Image();
        img.onload = () => {
            splitImage = img;
            splitCanvas.width = img.naturalWidth;
            splitCanvas.height = img.naturalHeight;
            splitLines = (result.lines || []).slice().sort((a, b) => a - b);
            drawSplit();
            refresh();
            showMsg("已摆正，请在切题完成后重新勾选这一页要重练的题", true);
        };
        img.onerror = () => showMsg("无法载入摆正后的图片", false);
        img.src = result.source_image_url;
    } catch (error) {
        showMsg(error.message, false);
    } finally {
        cornerApplyBtn.textContent = "确认摆正";
        cornerApplyBtn.disabled = cornerPoints.length !== 4;
    }
};

// ---------- 答案遮挡 ----------

const maskModal = document.getElementById("maskModal");
const maskCanvas = document.getElementById("maskCanvas");
const maskContext = maskCanvas.getContext("2d");
const maskList = document.getElementById("maskList");
const maskSaveBtn = document.getElementById("maskSaveBtn");
let maskQuestionRef = null;
let maskImage = null;
let masks = [];
let maskStart = null;

function maskPoint(event) {
    const rect = maskCanvas.getBoundingClientRect();
    return [
        Math.max(0, Math.min(maskCanvas.width, (event.clientX - rect.left) * maskCanvas.width / rect.width)),
        Math.max(0, Math.min(maskCanvas.height, (event.clientY - rect.top) * maskCanvas.height / rect.height)),
    ];
}

function drawMasks() {
    if (!maskImage) return;
    maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskContext.drawImage(maskImage, 0, 0);
    masks.forEach((mask) => {
        const x = (mask[0] * maskCanvas.width) / 1000;
        const y = (mask[1] * maskCanvas.height) / 1000;
        const width = (mask[2] * maskCanvas.width) / 1000;
        const height = (mask[3] * maskCanvas.height) / 1000;
        maskContext.fillStyle = "rgba(255,255,255,0.82)";
        maskContext.strokeStyle = "#f56c6c";
        maskContext.lineWidth = Math.max(2, maskCanvas.width / 400);
        maskContext.fillRect(x, y, width, height);
        maskContext.strokeRect(x, y, width, height);
    });
    maskList.innerHTML = "";
    masks.forEach((mask, index) => {
        const chip = document.createElement("button");
        chip.className = "mask-chip";
        chip.textContent = `遮挡 ${index + 1} ×`;
        chip.onclick = () => {
            masks.splice(index, 1);
            drawMasks();
        };
        maskList.appendChild(chip);
    });
}

function openMask(q) {
    maskQuestionRef = q;
    try {
        masks = JSON.parse(q.answer_masks || "[]");
    } catch (e) {
        masks = [];
    }
    const img = new Image();
    img.onload = () => {
        maskImage = img;
        maskCanvas.width = img.naturalWidth;
        maskCanvas.height = img.naturalHeight;
        drawMasks();
    };
    img.onerror = () => showMsg("无法载入错题图片", false);
    img.src = q.image_url;
    maskModal.hidden = false;
}

maskCanvas.addEventListener("pointerdown", (event) => {
    maskStart = maskPoint(event);
    maskCanvas.setPointerCapture(event.pointerId);
});

maskCanvas.addEventListener("pointerup", (event) => {
    if (!maskStart) return;
    const end = maskPoint(event);
    const left = Math.min(maskStart[0], end[0]);
    const top = Math.min(maskStart[1], end[1]);
    const width = Math.abs(end[0] - maskStart[0]);
    const height = Math.abs(end[1] - maskStart[1]);
    maskStart = null;
    if (width < 8 || height < 8) {
        showMsg("遮挡框太小，请重新拖动", false);
        return;
    }
    masks.push([
        (left * 1000) / maskCanvas.width,
        (top * 1000) / maskCanvas.height,
        (width * 1000) / maskCanvas.width,
        (height * 1000) / maskCanvas.height,
    ]);
    drawMasks();
});

document.getElementById("maskClearBtn").onclick = () => {
    masks = [];
    drawMasks();
};
document.getElementById("maskCancelBtn").onclick = () => (maskModal.hidden = true);

maskSaveBtn.onclick = async () => {
    if (!maskQuestionRef) return;
    maskSaveBtn.disabled = true;
    maskSaveBtn.textContent = "保存中…";
    try {
        const response = await fetch(`/api/questions/${maskQuestionRef.id}/answer-masks`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ masks }),
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.detail || "保存遮挡失败");
        maskModal.hidden = true;
        refresh();
        showMsg("遮挡已保存，只会应用到练习卷", true);
    } catch (error) {
        showMsg(error.message, false);
    } finally {
        maskSaveBtn.disabled = false;
        maskSaveBtn.textContent = "保存遮挡";
    }
};

// ---------- 上传 ----------

pickPhotoBtn.onclick = () => photoInput.click();

photoInput.onchange = () => {
    if (!photoInput.files.length) return;
    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = URL.createObjectURL(photoInput.files[0]);
    previewImg.src = previewObjectUrl;
    photoPreview.hidden = false;
    submitBtn.hidden = false;
};

// 上传一整页并把切出的题放进本次重练；手机照片和高拍仪共用
async function uploadPage(photo, filename) {
    const fd = new FormData();
    fd.append("subject", "数学");
    fd.append("photo", photo, filename);
    const r = await fetch("/api/pages", { method: "POST", body: fd });
    const result = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(result.detail || "保存失败");
    const questions = result.questions || [];
    questions.forEach(q => candidateIds.add(q.id));
    libraryOpen = false;
    refresh();
    return questions.length;
}

function pageSavedText(count) {
    return count > 1
        ? `已整理出 ${count} 道题，请勾选要重练的题`
        : "已保存整页，可在“修正”中调整切题，再勾选";
}

submitBtn.onclick = () => {
    if (!photoInput.files.length) {
        showMsg("请先拍照或选择照片", false);
        return;
    }
    submitBtn.disabled = true;
    submitBtn.textContent = "识别中…";
    const file = photoInput.files[0];

    uploadPage(file, file.name)
        .then((count) => {
            photoInput.value = "";
            photoPreview.hidden = true;
            submitBtn.hidden = true;
            if (previewObjectUrl) {
                URL.revokeObjectURL(previewObjectUrl);
                previewObjectUrl = null;
            }
            showMsg(pageSavedText(count), true);
        })
        .catch((e) => showMsg(e.message, false))
        .finally(() => {
            submitBtn.disabled = false;
            submitBtn.textContent = "用这张照片";
        });
};

// ---------- 高拍仪 ----------
//
// 成者 / 极匠等高拍仪是标准 UVC 摄像头，浏览器直接取流即可，不需要厂商 SDK。
// getUserMedia 只在安全上下文可用：电脑上用 localhost 打开时显示按钮，
// 手机经局域网 http 访问时没有这个 API，按钮自动隐藏，继续用手机相机。

const scannerModal = document.getElementById("scannerModal");
const scannerContent = document.getElementById("scannerContent");
const scannerVideo = document.getElementById("scannerVideo");
const scannerSelect = document.getElementById("scannerSelect");
const scannerInfo = document.getElementById("scannerInfo");
const scannerStatus = document.getElementById("scannerStatus");
const scannerShotBtn = document.getElementById("scannerShotBtn");
const SCANNER_KEY = "secondchance-scanner";
// 长边上限：A4 300dpi 约 3508px，切题和打印都够用；也让 JPEG 稳定在上传上限以内
const SCANNER_MAX_SIDE = 3600;
const mediaDevices = globalThis.navigator?.mediaDevices;
let scannerStream = null;
let scannerPrefs = { deviceId: "", rotation: 0 };
try {
    scannerPrefs = { ...scannerPrefs, ...JSON.parse(localStorage.getItem(SCANNER_KEY)) };
} catch (_) { /* 没有记忆也能用，只是每次要重选摄像头 */ }
let scannerShots = 0;
let scannerPending = 0;
let scannerQuestions = 0;
let scannerQueue = Promise.resolve();

function saveScannerPrefs() {
    try { localStorage.setItem(SCANNER_KEY, JSON.stringify(scannerPrefs)); } catch (_) { /* 忽略 */ }
}

function stopScannerStream() {
    if (scannerStream) scannerStream.getTracks().forEach(t => t.stop());
    scannerStream = null;
    scannerVideo.srcObject = null;
}

function updateScannerStatus(error) {
    if (error) {
        scannerStatus.textContent = error;
        scannerStatus.className = "scanner-status err";
        return;
    }
    scannerStatus.className = "scanner-status";
    if (!scannerShots) {
        scannerStatus.textContent = "";
    } else if (scannerPending) {
        scannerStatus.textContent = `已拍 ${scannerShots} 页，正在切题 ${scannerPending} 页…`;
    } else {
        scannerStatus.textContent = `已拍 ${scannerShots} 页，共切出 ${scannerQuestions} 道题`;
    }
}

async function openScannerCamera(deviceId) {
    stopScannerStream();
    scannerShotBtn.disabled = true;
    scannerInfo.textContent = "正在打开…";
    // 要一个远大于实际的分辨率，浏览器会落到设备支持的最高档
    const video = { width: { ideal: 4896 }, height: { ideal: 3672 } };
    let stream;
    try {
        stream = await mediaDevices.getUserMedia({ video: deviceId ? { ...video, deviceId: { exact: deviceId } } : video });
    } catch (error) {
        // 记住的那台拔掉了：退回系统默认摄像头
        if (!deviceId || !["OverconstrainedError", "NotFoundError"].includes(error.name)) throw error;
        stream = await mediaDevices.getUserMedia({ video });
    }
    if (scannerModal.hidden) {  // 摄像头还没打开，弹窗已经关了
        stream.getTracks().forEach(t => t.stop());
        return;
    }
    scannerStream = stream;
    scannerVideo.srcObject = scannerStream;
    const settings = scannerStream.getVideoTracks()[0].getSettings();
    scannerPrefs.deviceId = settings.deviceId || deviceId || "";
    saveScannerPrefs();
    scannerInfo.textContent = settings.width ? `${settings.width}×${settings.height}` : "";
    scannerShotBtn.disabled = false;
    scannerContent.focus();
}

async function listScannerCameras() {
    const cameras = (await mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
    scannerSelect.innerHTML = "";
    cameras.forEach((camera, index) => {
        const option = document.createElement("option");
        option.value = camera.deviceId;
        option.textContent = camera.label || `摄像头 ${index + 1}`;
        scannerSelect.appendChild(option);
    });
    return cameras;
}

async function openScanner() {
    scannerModal.hidden = false;
    scannerShots = scannerPending = scannerQuestions = 0;
    updateScannerStatus();
    applyScannerRotation();
    scannerContent.focus();
    const wanted = scannerPrefs.deviceId;
    try {
        // 先拿一次权限，枚举出来的设备才有名字
        await openScannerCamera(wanted);
        const cameras = await listScannerCameras();
        const remembered = wanted && cameras.some(c => c.deviceId === wanted);
        const czur = cameras.find(c => /CZUR|成者|极匠/i.test(c.label));
        if (!remembered && czur && czur.deviceId !== scannerPrefs.deviceId && !scannerModal.hidden) {
            // 第一次用：笔记本自带摄像头往往排在前面，优先换到高拍仪
            await openScannerCamera(czur.deviceId);
        }
        scannerSelect.value = scannerPrefs.deviceId;
    } catch (error) {
        scannerInfo.textContent = "";
        const name = error && error.name;
        updateScannerStatus(
            name === "NotAllowedError" ? "浏览器没有获得摄像头权限，请在地址栏左侧允许后重试"
            : name === "NotFoundError" ? "没有找到摄像头，请检查高拍仪的数据线，换个 USB 口试试"
            : name === "NotReadableError" ? "摄像头被占用，请先关掉成者客户端或其他用摄像头的程序"
            : "打开高拍仪失败：" + ((error && error.message) || error)
        );
    }
}

function closeScanner() {
    stopScannerStream();
    scannerModal.hidden = true;
    if (scannerShots) {
        showMsg(scannerPending ? "剩余几页还在切题，完成后会出现在本次重练" : `高拍仪共拍 ${scannerShots} 页，请勾选要重练的题`, true);
    }
}

function applyScannerRotation() {
    scannerVideo.style.transform = scannerPrefs.rotation ? `rotate(${scannerPrefs.rotation}deg)` : "";
    scannerVideo.classList.toggle("sideways", scannerPrefs.rotation % 180 !== 0);
}

// 从视频流抓一帧，按当前旋转角度转正、缩到长边上限，编码成 JPEG
function grabScannerFrame() {
    const width = scannerVideo.videoWidth;
    const height = scannerVideo.videoHeight;
    if (!width || !height) return Promise.resolve(null);
    const scale = Math.min(1, SCANNER_MAX_SIDE / Math.max(width, height));
    const w = Math.round(width * scale);
    const h = Math.round(height * scale);
    const sideways = scannerPrefs.rotation % 180 !== 0;
    const canvas = document.createElement("canvas");
    canvas.width = sideways ? h : w;
    canvas.height = sideways ? w : h;
    const context = canvas.getContext("2d");
    context.translate(canvas.width / 2, canvas.height / 2);
    context.rotate((scannerPrefs.rotation * Math.PI) / 180);
    context.drawImage(scannerVideo, -w / 2, -h / 2, w, h);
    return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.92));
}

async function scannerShoot() {
    if (scannerShotBtn.disabled) return;
    const blob = await grabScannerFrame();
    if (!blob) return;
    scannerShots += 1;
    scannerPending += 1;
    updateScannerStatus();
    scannerVideo.classList.remove("flash");
    void scannerVideo.offsetWidth;  // 重新触发闪一下的动画
    scannerVideo.classList.add("flash");
    const stamp = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
    // 一页一页排队上传：切题吃 CPU，并发上去只会更慢
    scannerQueue = scannerQueue
        .then(() => uploadPage(blob, `scanner-${stamp}-${scannerShots}.jpg`))
        .then((count) => { scannerQuestions += count; scannerPending -= 1; updateScannerStatus(); })
        .catch((error) => {
            scannerPending -= 1;
            if (scannerModal.hidden) showMsg(error.message, false);
            else updateScannerStatus(error.message);
        });
}

document.getElementById("scannerBtn").hidden = !mediaDevices?.getUserMedia;
document.getElementById("scannerBtn").onclick = openScanner;
document.getElementById("scannerDoneBtn").onclick = closeScanner;
scannerShotBtn.onclick = scannerShoot;
scannerSelect.onchange = () => {
    openScannerCamera(scannerSelect.value).catch(e => updateScannerStatus("切换摄像头失败：" + e.message));
};
document.getElementById("scannerRotateBtn").onclick = () => {
    scannerPrefs.rotation = (scannerPrefs.rotation + 90) % 360;
    saveScannerPrefs();
    applyScannerRotation();
};
// 空格 = 拍这一页，不管焦点停在哪个按钮上（按钮的空格默认在 keyup 时触发，一并拦掉）
const isScannerSpace = event => event.code === "Space" && event.target.tagName !== "SELECT";
scannerModal.onkeydown = (event) => {
    if (isScannerSpace(event)) {
        event.preventDefault();
        if (!event.repeat) scannerShoot();
    } else if (event.key === "Escape") {
        closeScanner();
    }
};
scannerModal.onkeyup = (event) => { if (isScannerSpace(event)) event.preventDefault(); };

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
            refresh();
        } else {
            showMsg("保存失败", false);
        }
    }).catch(() => showMsg("保存失败", false));
};
document.getElementById("editCancelBtn").onclick = () => (document.getElementById("editModal").hidden = true);
document.getElementById("lightboxCloseBtn").onclick = () => (document.getElementById("lightbox").hidden = true);
document.getElementById("lightbox").onclick = (e) => {
    if (e.target.id === "lightbox") e.target.hidden = true;
};

// 出卷历史：打印时 paper.html 记的那笔，在这里回看。加载失败就静默——历史是附赠品。
function loadHistory() {
    const list = document.getElementById("historyList");
    if (!list) return;
    fetch("/api/prints")
        .then((r) => (r.ok ? r.json() : []))
        .then((papers) => {
            list.innerHTML = "";
            if (!papers.length) {
                const empty = document.createElement("p");
                empty.className = "history-empty";
                empty.textContent = "还没有打印过练习卷。每次打印会自动记录在这里，不用做任何操作。";
                list.appendChild(empty);
                return;
            }
            papers.forEach((paper) => {
                const row = document.createElement("div");
                row.className = "history-item";

                const meta = document.createElement("div");
                meta.className = "history-meta";
                const day = document.createElement("strong");
                day.textContent = new Date(paper.created_at).toLocaleDateString("zh-CN", { month: "long", day: "numeric" });
                const count = document.createElement("span");
                count.textContent = `${paper.count} 题`;
                meta.append(day, count);
                const kps = [...new Set(paper.questions.map((q) => q.knowledge_point).filter(Boolean))];
                if (kps.length) {
                    const kpLine = document.createElement("span");
                    kpLine.className = "history-kps";
                    kpLine.textContent = kps.join("、");
                    meta.append(kpLine);
                }

                const thumbs = document.createElement("div");
                thumbs.className = "history-thumbs";
                paper.questions.slice(0, 6).forEach((q) => {
                    const thumb = document.createElement("img");
                    thumb.src = q.image_url;
                    thumb.alt = "错题缩略图";
                    thumb.loading = "lazy";
                    thumbs.appendChild(thumb);
                });

                row.append(meta, thumbs);
                list.appendChild(row);
            });
        })
        .catch(() => {});
}

initKpList();
loadHistory();
refresh();
