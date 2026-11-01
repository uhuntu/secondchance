const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function harness(script, search = '') {
    const elements = new Map();
    const element = () => ({
        children: [], value: '', innerHTML: '',
        append(...items) { this.children.push(...items); },
        appendChild(item) { this.children.push(item); },
        classList: { toggle() {}, add() {}, remove() {} },
        addEventListener() {}, getContext() { return {}; },
        querySelectorAll() { return []; },
    });
    const document = {
        getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
        createElement: element, createTextNode: text => text,
    };
    let opened;
    const context = vm.createContext({ document, URLSearchParams, location: { search },
        sessionStorage: { getItem() { return null; }, setItem() {} },
        window: { open(url) { opened = url; }, addEventListener() {} },
        fetch: () => new Promise(() => {}), setTimeout, clearTimeout,
    });
    vm.runInContext(script, context);
    return { run: code => vm.runInContext(code, context), elements, opened: () => opened, context };
}

const app = fs.readFileSync('static/app.js', 'utf8');
const paper = fs.readFileSync('static/paper.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];

test('draft starts empty; selection prints only chosen IDs; new draft keeps library', () => {
    const h = harness(app);
    h.run(`renderList([{id: 1}, {id: 2}]);`);
    assert.equal(h.elements.get('printSelectedBtn').disabled, true);
    h.run(`const card = buildCard(questionsCache[1], 2); const check = card.children[1].children[0].children[0]; check.checked = true; check.onchange();`);
    h.elements.get('printSelectedBtn').onclick();
    assert.equal(h.opened(), '/static/paper.html?ids=2');
    h.elements.get('clearSelectionBtn').onclick();
    assert.equal(h.run('selectedIds.size'), 0);
    assert.equal(h.run('questionsCache.length'), 2);
});

test('recutting a page clears its selection, keeps other pages, and exposes new crops', () => {
    const h = harness(app);
    h.run(`questionsCache = [{id:1, source_image_url:'a'}, {id:2, source_image_url:'b'}]; candidateIds = new Set([1,2]); selectedIds = new Set([1,2]); replaceDraftGroup(questionsCache[0], [{id:3}, {id:4}]);`);
    assert.equal(h.run('JSON.stringify([...selectedIds])'), '[2]');
    assert.equal(h.run('JSON.stringify([...candidateIds])'), '[2,3,4]');
});

test('print scope never falls back to entire library for missing or empty selection', () => {
    for (const [query, expected] of [['?ids=2', '[2]'], ['?ids=999', '[]'], ['?ids=', '[]']]) {
        const h = harness(paper, query);
        h.run('allQuestions = [{id:1}, {id:2}];');
        assert.equal(h.run('JSON.stringify(selectedQuestions().map(q => q.id))'), expected);
        assert.equal(h.elements.get('rangeFilter').hidden, true);
    }
});

test('scanner button stays hidden where the browser offers no camera API (phone over LAN http)', () => {
    const h = harness(app);
    assert.equal(h.elements.get('scannerBtn').hidden, true);
});

test('printing records exactly the questions on the paper; button click and Ctrl+P record once', () => {
    const h = harness(paper, '?ids=2');
    h.run('allQuestions = [{id:1}, {id:2}];');
    const posts = [];
    h.context.fetch = (url, options) => {
        posts.push({ url, body: JSON.parse(options.body) });
        return Promise.resolve({ ok: true });
    };
    // beforeprint（Ctrl+P）和按钮点击都会触发 recordPrint，只允许记一笔
    h.run('recordPrint(); recordPrint();');
    assert.equal(posts.length, 1);
    assert.equal(posts[0].url, '/api/prints');
    assert.deepEqual(posts[0].body, { question_ids: [2] });
});

test('card shows practice count only when the question has been practiced', () => {
    const h = harness(app);
    h.run(`questionsCache = [
        {id: 5, practice_count: 3, last_practiced_at: '2026-11-01T10:00:00+00:00'},
        {id: 6},
    ];`);
    h.run('var cards = questionsCache.map((q, i) => buildCard(q, i + 1));');
    const texts = h.run('cards.map(card => [...card.children[1].children[0].children] '
        + '.map(c => typeof c === "string" ? c : c.textContent).filter(Boolean))');
    // 跨 vm realm 的数组原型不同，deepStrictEqual 会拒绝：用序列化比较
    assert.equal(JSON.stringify(texts[0]), JSON.stringify(['第 1 题 · 加入本次重练', '练过 3 次']));
    assert.equal(JSON.stringify(texts[1]), JSON.stringify(['第 2 题 · 加入本次重练']));
});
