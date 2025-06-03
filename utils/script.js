
document.getElementById("askBtn").addEventListener("click", async () => {
    const question = document.getElementById("question").value.trim();
    const answerText = document.getElementById("answerText");
    const webSearchResultsContainer = document.getElementById('webSearchResultsContainer');
    const pubmedSearchResultsContainer = document.getElementById('pubmedSearchResultsContainer');
    function sanitize(input) {
        return DOMPurify.sanitize(input);
    }
     function renderSearchResults(containerElement, resultsArray, title) {
            let html = '';
            if (resultsArray && resultsArray.length > 0) {
                html += `<p class="text-sm text-gray-600 mb-2">Found ${resultsArray.length} ${title}:</p>`;
                resultsArray.forEach(result => {
                    html += `
                        <div class="border-b border-gray-200 pb-3 mb-3 last:border-b-0 last:mb-0">
                            <h3 class="text-base font-semibold text-[#1993e5] hover:underline mb-1">
                                <a href="${sanitize(result.url)}" target="_blank" rel="noopener noreferrer">
                                    ${sanitize(result.title)}
                                </a>
                            </h3>
                            <p class="text-sm text-gray-700 mb-1">${sanitize(result.snippet)}</p>
                            <p class="text-xs text-gray-500">Source: ${sanitize(result.source)}</p>
                            <p class="text-xs text-gray-500">URL: <a href="${sanitize(result.url)}" target="_blank" rel="noopener noreferrer" class="text-[#1993e5]">${DOMPurify.sanitize(result.url)}</a></p>
                        </div>
                    `;
                });
            } else {
                html = sanitize(`No ${title} found for this query.`);
            }
            containerElement.innerHTML = html;
        }

    console.log("Question submitted:", question);
    if (!question) {
        answerText.innerHTML = sanitize('<p style="color: red;">Please enter a question.</p>');
        webSearchResultsContainer.textContent = 'No web search results yet.';
        pubmedSearchResultsContainer.textContent = 'No PubMed search results yet.';
        return;
    }
    answerText.textContent = "Thinking...";
    answerText.innerHTML = sanitize('<p>Thinking...</p>');
    webSearchResultsContainer.textContent = 'Loading web search results...';
    pubmedSearchResultsContainer.textContent = 'Loading PubMed results...';

    try {
        const response = await fetch("/mcp", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: question })
        });
        const data = await response.json();
        if (!response.ok) {
            answerText.textContent = "Error: " + (data.error || "Failed to fetch answer.");
        }
        console.log("Response received:", data, "Final answer:", data.final_answer);
        if (data.web_search_results) {
            const parsedWebResults = JSON.parse(data.web_search_results);
            console.log("parsed", parsedWebResults);
            renderSearchResults(webSearchResultsContainer, parsedWebResults, 'Web results')
        } else {
            webSearchResultsContainer.innerHTML = DOMPurify.sanitize('No web search results available.');
        }

        if (data.pubmed_results) {
            const parsedPubmedResults = JSON.parse(data.pubmed_results);
            renderSearchResults(pubmedSearchResultsContainer, parsedPubmedResults, 'PubMed results')
            // pubmedSearchResultsContainer.innerHTML = sanitize( data.pubmed_results || '')
        } else {
            pubmedSearchResultsContainer.textContent = 'No PubMed search results found.';
        }
        const cleanHtml = sanitize(data.final_answer || "").replace(/^```html|```$/g, "").trim();
        answerText.innerHTML = cleanHtml || "No answer returned.";
    } catch (err) {
        console.error(err);
        answerText.innerHTML = sanitize(`<p style="color: red;">Error: ${err.message}. Please try again.</p>`);
         webSearchResultsContainer.textContent = 'Error loading web search results.';
        pubmedSearchResultsContainer.textContent = 'Error loading PubMed search results.';

    }
});
