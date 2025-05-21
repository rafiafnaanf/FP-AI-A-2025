document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('similarityForm');
    const predictBtn = document.getElementById('predictBtn');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const loadingDiv = document.getElementById('loading');

    const predictionText = document.getElementById('predictionText');
    const distanceText = document.getElementById('distanceText');
    const thresholdText = document.getElementById('thresholdText');
    const modelText = document.getElementById('modelText');
    const metricText = document.getElementById('metricText'); // Get the new element
    const errorText = document.getElementById('errorText');

    const image1Input = document.getElementById('image1');
    const image2Input = document.getElementById('image2');
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');

    // Model and Metric selectors
    const modelNameSelect = document.getElementById('model_name');
    const distanceMetricSelect = document.getElementById('distance_metric');

    function displayPreview(input, previewElement) {
        input.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewElement.src = e.target.result;
                    previewElement.style.display = 'block';
                }
                reader.readAsDataURL(file);
            } else {
                previewElement.src = '#';
                previewElement.style.display = 'none';
            }
        });
    }

    displayPreview(image1Input, preview1);
    displayPreview(image2Input, preview2);

    form.addEventListener('submit', async function (event) {
        event.preventDefault();

        resultDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        loadingDiv.style.display = 'block';
        predictBtn.disabled = true;

        const formData = new FormData(form);
        formData.append('model_name', modelNameSelect.value);
        formData.append('distance_metric', distanceMetricSelect.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData, // FormData now includes image1, image2, model_name, distance_metric
            });

            const data = await response.json();

            loadingDiv.style.display = 'none';
            predictBtn.disabled = false;

            if (response.ok) {
                predictionText.textContent = data.verified ? 'Result: Same Person (Verified)' : 'Result: Different Persons (Not Verified)';
                predictionText.className = data.verified ? 'verified' : 'not-verified';
                distanceText.textContent = `Distance: ${data.distance}`;
                thresholdText.textContent = `Threshold: ${data.threshold} (Distance < Threshold for verification)`;
                modelText.textContent = `Model Used: ${data.model}`;
                metricText.textContent = `Distance Metric Used: ${data.similarity_metric}`; // Display the metric
                resultDiv.style.display = 'block';
            } else {
                errorText.textContent = data.error || 'An unknown error occurred.';
                errorDiv.style.display = 'block';
            }
        } catch (err) {
            loadingDiv.style.display = 'none';
            predictBtn.disabled = false;
            errorText.textContent = 'Failed to connect to the server or an unexpected error occurred: ' + err.message;
            errorDiv.style.display = 'block';
        }
    });
});
