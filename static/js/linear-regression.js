const form = document.querySelector("#sales-prediction-form");
const statusMessage = document.querySelector("#prediction-status");
const resultPanel = document.querySelector("#prediction-result");

if (form && statusMessage && resultPanel) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resultPanel.hidden = true;
    statusMessage.textContent = "";

    if (!form.reportValidity()) {
      return;
    }

    const submitButton = form.querySelector('button[type="submit"]');
    const payload = Object.fromEntries(new FormData(form));
    for (const [name, value] of Object.entries(payload)) {
      payload[name] = Number(value);
    }

    submitButton.disabled = true;
    submitButton.textContent = "Estimating…";
    statusMessage.textContent = "Sending inputs to the trained model…";

    try {
      const response = await fetch("/api/predict/linear-regression", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "The model could not make a prediction.");
      }

      resultPanel.replaceChildren();
      const heading = document.createElement("h3");
      heading.textContent = `Estimated sales: $${data.prediction.sales_amount.toLocaleString(
        undefined,
        { minimumFractionDigits: 2, maximumFractionDigits: 2 },
      )}`;
      const explanation = document.createElement("p");
      explanation.textContent = data.interpretation;
      resultPanel.append(heading, explanation);
      resultPanel.hidden = false;
      statusMessage.textContent = "Prediction complete.";
    } catch (error) {
      statusMessage.textContent = error.message;
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Estimate sales";
    }
  });
}

const evaluationElement = document.querySelector("#evaluation-data");
const chartCanvas = document.querySelector("#sales-evaluation-chart");

if (evaluationElement && chartCanvas && window.Chart) {
  const evaluation = JSON.parse(evaluationElement.textContent);
  const values = [...evaluation.actual, ...evaluation.predicted];
  const maximum = Math.max(...values);

  new Chart(chartCanvas, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Held-out test records",
          data: evaluation.actual.map((actual, index) => ({
            x: actual,
            y: evaluation.predicted[index],
          })),
          backgroundColor: "rgba(23, 184, 144, 0.65)",
          pointRadius: 4,
        },
        {
          label: "Perfect prediction",
          data: [{ x: 0, y: 0 }, { x: maximum, y: maximum }],
          type: "line",
          borderColor: "#62727d",
          borderDash: [6, 5],
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Actual sales (USD)" } },
        y: { title: { display: true, text: "Predicted sales (USD)" } },
      },
    },
  });
}
