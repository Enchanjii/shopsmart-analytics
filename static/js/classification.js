function renderClassificationCharts() {
  const dataElement = document.querySelector("[id^='classification-data-']");
  if (!dataElement || !window.Chart) {
    return;
  }

  const slug = dataElement.id.replace("classification-data-", "");
  const result = JSON.parse(dataElement.textContent);
  const confusionCanvas = document.querySelector(`#confusion-chart-${slug}`);
  if (confusionCanvas) {
    new Chart(confusionCanvas, {
      type: "bar",
      data: {
        labels: result.labels,
        datasets: result.labels.map((actualLabel, index) => ({
          label: `Actual ${actualLabel}`,
          data: result.matrix[index],
          backgroundColor: [
            "rgba(23, 184, 144, 0.72)",
            "rgba(53, 116, 179, 0.7)",
            "rgba(223, 163, 57, 0.78)",
          ][index % 3],
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            stacked: true,
            title: { display: true, text: "Predicted class" },
          },
          y: {
            stacked: true,
            beginAtZero: true,
            title: { display: true, text: "Test records" },
          },
        },
      },
    });
  }

  if (slug === "neural-network" && result.history) {
    const epochs = result.history.loss.map((_, index) => index + 1);
    const lossCanvas = document.querySelector(`#loss-chart-${slug}`);
    const validationCanvas = document.querySelector(`#validation-chart-${slug}`);
    if (lossCanvas) {
      new Chart(lossCanvas, {
        type: "line",
        data: {
          labels: epochs,
          datasets: [{
            label: "Training loss",
            data: result.history.loss,
            borderColor: "#087c63",
            pointRadius: 0,
            tension: 0.2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: "Training epoch" } },
            y: { title: { display: true, text: "Training loss" } },
          },
        },
      });
    }
    if (validationCanvas && result.history.validation_accuracy.length) {
      new Chart(validationCanvas, {
        type: "line",
        data: {
          labels: epochs.slice(0, result.history.validation_accuracy.length),
          datasets: [{
            label: "Validation accuracy",
            data: result.history.validation_accuracy,
            borderColor: "#3574b3",
            pointRadius: 0,
            tension: 0.2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: "Training epoch" } },
            y: {
              min: 0,
              max: 1,
              title: { display: true, text: "Validation accuracy" },
            },
          },
        },
      });
    }
  }
}

for (const form of document.querySelectorAll(".classification-form")) {
  const status = form.querySelector('[role="status"]');
  const resultPanel = form.querySelector(".prediction-result");
  const submitButton = form.querySelector('button[type="submit"]');
  const slug = form.dataset.modelSlug;
  const originalButtonText = submitButton.textContent.trim();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resultPanel.hidden = true;
    status.textContent = "";
    if (!form.reportValidity()) {
      return;
    }

    const payload = Object.fromEntries(new FormData(form));
    for (const [name, value] of Object.entries(payload)) {
      if (name !== "review_text") {
        payload[name] = Number(value);
      }
    }
    submitButton.disabled = true;
    submitButton.textContent = "Predicting…";
    status.textContent = "Sending inputs to the trained model…";

    try {
      const response = await fetch(`/api/predict/${slug}`, {
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
      heading.textContent = `Predicted ${data.target.replaceAll("_", " ")}: ${data.prediction.label.replaceAll("_", " ")}`;
      resultPanel.append(heading);
      if (data.prediction.probabilities) {
        const probabilityText = document.createElement("p");
        probabilityText.textContent = `Model class probabilities: ${Object.entries(data.prediction.probabilities)
          .map(([label, probability]) => `${label.replaceAll("_", " ")} ${(probability * 100).toFixed(1)}%`)
          .join(" · ")}`;
        resultPanel.append(probabilityText);
      }
      const explanation = document.createElement("p");
      explanation.textContent = data.interpretation;
      resultPanel.append(explanation);
      resultPanel.hidden = false;
      status.textContent = "Prediction complete.";

      const neighborTable = document.querySelector(`#neighbors-${slug}`);
      if (neighborTable && data.prediction.similar_profiles) {
        const dataElement = document.querySelector(`#classification-data-${slug}`);
        const config = JSON.parse(dataElement.textContent);
        const tbody = neighborTable.querySelector("tbody");
        tbody.replaceChildren();
        for (const profile of data.prediction.similar_profiles) {
          const row = document.createElement("tr");
          for (const field of config.features) {
            const cell = document.createElement("td");
            cell.textContent = profile[field];
            row.append(cell);
          }
          const classCell = document.createElement("td");
          classCell.textContent = profile[config.target];
          row.append(classCell);
          const distanceCell = document.createElement("td");
          distanceCell.textContent = profile.distance.toFixed(3);
          row.append(distanceCell);
          tbody.append(row);
        }
        neighborTable.hidden = false;
      }
    } catch (error) {
      status.textContent = error.message;
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = originalButtonText;
    }
  });
}

renderClassificationCharts();
