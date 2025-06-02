
document.getElementById("askBtn").addEventListener("click", async () => {
    const question = document.getElementById("question").value.trim();
    const answerText = document.getElementById("answerText");
    console.log("Question submitted:", question);
    if (!question) {
        answerText.textContent = "Please enter a question.";
        return;
    }
    answerText.textContent = "Searching...";
    try {
        const response = await fetch("/mcp", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: question })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Failed to fetch answer.");
        }
        console.log("Response received:", data, "Final answer:", data.final_answer);
        const cleanHtml = DOMPurify.sanitize(data.final_answer || "",);
        answerText.innerHTML = cleanHtml || "No answer returned.";
    } catch (err) {
        console.error(err);
        answerText.textContent = "An error occurred while fetching the answer.";
    }
});
