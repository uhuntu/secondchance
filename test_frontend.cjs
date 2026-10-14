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
        window: { open(url) { opened = url; } },
        fetch: () => new Promise(() => {}), setTimeout, clearTimeout,
    });
    vm.runInContext(script, context);
    return { run: code => vm.runInContext(code, context), elements, opened: () => opened };
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
