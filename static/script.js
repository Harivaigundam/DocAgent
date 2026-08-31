document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');
    const docTypeSelect = document.getElementById('docType');
    const loadingDiv = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');
    const errorSection = document.getElementById('errorSection');
    const refreshCostsBtn = document.getElementById('refreshCosts');
    const refreshQueueBtn = document.getElementById('refreshQueue');

    uploadForm.addEventListener('submit', handleUpload);
    refreshCostsBtn.addEventListener('click', loadCosts);
    refreshQueueBtn.addEventListener('click', loadReviewQueue);

    loadCosts();
    loadModels();
    loadReviewQueue();

    async function handleUpload(e) {
        e.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            showError('Please select a file');
            return;
        }

        showLoading();
        hideError();
        hideResult();

        const formData = new FormData();
        formData.append('file', file);

        const docType = docTypeSelect.value;
        const url = docType ? `/api/extract/${docType}` : '/api/extract';

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                let errorMessage;
                try {
                    const errorJson = JSON.parse(errorText);
                    errorMessage = errorJson.error || errorJson.detail || 'Unknown error';
                } catch {
                    errorMessage = errorText || `HTTP ${response.status}`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            showResult(data);
            loadCosts();
            loadReviewQueue();
        } catch (error) {
            showError(error.message || 'An unknown error occurred');
        } finally {
            hideLoading();
        }
    }

    function showLoading() {
        loadingDiv.classList.remove('hidden');
    }

    function hideLoading() {
        loadingDiv.classList.add('hidden');
    }

    function showResult(data) {
        resultSection.classList.remove('hidden');

        document.getElementById('docType').textContent = data.document_type;
        document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById('modelUsed').textContent = data.model_used;
        document.getElementById('cost').textContent = `$${data.cost.toFixed(4)}`;
        document.getElementById('guardrails').textContent = data.guardrail_passed ? 'Passed' : 'Failed';
        document.getElementById('requiresReview').textContent = data.requires_review ? 'Yes' : 'No';

        const extractionResult = document.getElementById('extractionResult');
        extractionResult.textContent = JSON.stringify(data.extraction, null, 2);
    }

    function hideResult() {
        resultSection.classList.add('hidden');
    }

    function showError(message) {
        errorSection.classList.remove('hidden');
        document.getElementById('errorMessage').textContent = message;
    }

    function hideError() {
        errorSection.classList.add('hidden');
    }

    async function loadCosts() {
        try {
            const response = await fetch('/api/costs');
            const data = await response.json();

            document.getElementById('totalCost').textContent = `$${data.total_cost.toFixed(4)}`;
            document.getElementById('docsProcessed').textContent = data.docs_processed;
        } catch (error) {
            console.error('Failed to load costs:', error);
        }
    }

    async function loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            const modelList = document.getElementById('modelList');
            modelList.innerHTML = '';

            data.models.forEach(model => {
                const div = document.createElement('div');
                div.className = 'model-item';
                div.innerHTML = `
                    <span class="model-name">${model.name} (${model.model})</span>
                    <span class="model-cost">$${model.cost_per_doc.toFixed(4)}/doc</span>
                `;
                modelList.appendChild(div);
            });
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    async function loadReviewQueue() {
        try {
            const response = await fetch('/api/review/queue');
            const data = await response.json();

            const reviewQueue = document.getElementById('reviewQueue');
            reviewQueue.innerHTML = '';

            if (data.queue.length === 0) {
                reviewQueue.innerHTML = '<p>No items in review queue</p>';
                return;
            }

            data.queue.forEach(item => {
                const div = document.createElement('div');
                div.className = 'review-item';
                div.innerHTML = `
                    <span>${item.doc_id}</span>
                    <button class="btn-secondary" onclick="reviewItem('${item.doc_id}')">Review</button>
                `;
                reviewQueue.appendChild(div);
            });
        } catch (error) {
            console.error('Failed to load review queue:', error);
        }
    }
});

async function reviewItem(docId) {
    const review = {
        status: 'approved',
        timestamp: new Date().toISOString(),
    };

    try {
        const response = await fetch(`/api/review/${docId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(review),
        });

        if (response.ok) {
            loadReviewQueue();
        }
    } catch (error) {
        console.error('Failed to submit review:', error);
    }
}
