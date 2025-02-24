document.getElementById('urlForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const urlInput = document.getElementById('urlInput');
    const submitBtn = document.getElementById('submitBtn');
    const progressBarContainer = document.getElementById('progressBarContainer');
    const progressBar = document.getElementById('progressBar');
    const resultsDiv = document.getElementById('results');

    // Validate URL
    if (!urlInput.value.match(/^https?:\/\/.+/)) {
        resultsDiv.innerHTML = '<p class="error">Please enter a valid URL starting with http:// or https://</p>';
        return;
    }

    // Disable form and show progress
    urlInput.disabled = true;
    submitBtn.disabled = true;
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '50%';
    resultsDiv.innerHTML = '<p class="loading">Extracting URLs, please wait...</p>';

    try {
        const response = await fetch('', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            },
            body: `url=${encodeURIComponent(urlInput.value)}`,
        });

        progressBar.style.width = '100%';
        const data = await response.json();

        if (data.status === 'success') {
            const uniqueUrls = [...new Set(data.urls)].sort();
            resultsDiv.innerHTML = `
                <h2>Extracted URLs (${uniqueUrls.length})</h2>
                <div class="url-list">
                    ${uniqueUrls.map(url => `
                        <div class="url-item">
                            <a href="${url}" target="_blank" rel="noopener noreferrer">
                                ${url}
                            </a>
                            <button onclick="navigator.clipboard.writeText('${url}')" class="copy-btn">
                                <i class="fas fa-copy"></i>
                            </button>
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            resultsDiv.innerHTML = `<p class="error">Error: ${data.message}</p>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    } finally {
        // Re-enable form
        urlInput.disabled = false;
        submitBtn.disabled = false;
        progressBarContainer.style.display = 'none';
        progressBar.style.width = '0%';
    }
});


