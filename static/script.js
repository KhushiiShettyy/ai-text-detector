let caseCounter = 1;

document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = document.getElementById('inputText').value.trim();
    const resultBox = document.getElementById('result');
    const errorBox = document.getElementById('errorBox');

    errorBox.classList.add('hidden');
    resultBox.classList.add('hidden');

    if (text.length < 20) {
        errorBox.textContent = "Please enter at least a few sentences.";
        errorBox.classList.remove('hidden');
        return;
    }

    const btn = document.getElementById('analyzeBtn');
    const originalLabel = btn.innerHTML;
    btn.innerHTML = "<span>Analyzing…</span>";
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (!response.ok) {
            errorBox.textContent = data.error || "Something went wrong.";
            errorBox.classList.remove('hidden');
            return;
        }

        const isAI = data.verdict === "AI-Generated";

        document.getElementById('ticketId').textContent =
            "CASE-" + String(caseCounter++).padStart(4, '0');

        const stamp = document.getElementById('stampVerdict');
        stamp.textContent = isAI ? "AI-GENERATED" : "HUMAN-WRITTEN";
        stamp.classList.toggle('ai', isAI);

        document.getElementById('confidenceValue').textContent = data.confidence;
        const bar = document.getElementById('confidenceBar');
        bar.style.width = data.confidence + "%";
        bar.classList.toggle('ai', isAI);

        const reasoningList = document.getElementById('reasoningList');
        reasoningList.innerHTML = "";
        const r = data.reasoning;

        const items = [
            { name: "Average sentence length", value: r.avg_sentence_length + " words", pct: Math.min(r.avg_sentence_length / 30 * 100, 100) },
            { name: "Burstiness (sentence variation)", value: r.burstiness, pct: Math.min(r.burstiness / 15 * 100, 100) },
            { name: "Vocabulary richness", value: r.vocabulary_richness, pct: Math.min(r.vocabulary_richness * 100, 100) },
            { name: "Readability score", value: r.readability_score, pct: Math.min(Math.max(r.readability_score, 0) / 100 * 100, 100) }
        ];

        items.forEach(item => {
            const li = document.createElement('li');
            li.className = 'evidence-item';
            li.innerHTML = `
                <span class="evidence-name">${item.name}</span>
                <span class="evidence-val">${item.value}</span>
                <span class="evidence-bar-bg"><span class="evidence-bar-fill" style="width:${item.pct}%"></span></span>
            `;
            reasoningList.appendChild(li);
        });

        resultBox.classList.remove('hidden');
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
        errorBox.textContent = "Failed to connect to server.";
        errorBox.classList.remove('hidden');
    } finally {
        btn.innerHTML = originalLabel;
        btn.disabled = false;
    }
});