// ==========================================
// Fruit Image Classification Frontend
// ==========================================

// ------------------------------------------
// DOM Elements
// ------------------------------------------

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");

const predictBtn = document.getElementById("predictBtn");

const loading = document.getElementById("loading");

const resultCard = document.getElementById("resultCard");

const fruitName = document.getElementById("fruitName");
const confidence = document.getElementById("confidence");

const dropArea = document.getElementById("drop-area");

// ------------------------------------------
// API Endpoint
// ------------------------------------------

const API_URL = "http://127.0.0.1:8000/predict";

let selectedFile = null;

// ==========================================
// File Selection
// ==========================================

imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {

        alert("Please select a valid image.");

        return;

    }

    selectedFile = file;

    showPreview(file);

});

// ==========================================
// Drag & Drop
// ==========================================

dropArea.addEventListener("dragover", (event) => {

    event.preventDefault();

    dropArea.classList.add("dragover");

});

dropArea.addEventListener("dragleave", () => {

    dropArea.classList.remove("dragover");

});

dropArea.addEventListener("drop", (event) => {

    event.preventDefault();

    dropArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {

        alert("Please upload an image.");

        return;

    }

    selectedFile = file;

    showPreview(file);

});

// ==========================================
// Image Preview
// ==========================================

function showPreview(file) {

    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src = event.target.result;

        previewImage.style.display = "block";

        resultCard.style.display = "none";

    };

    reader.readAsDataURL(file);

}

// ==========================================
// Predict
// ==========================================

predictBtn.addEventListener("click", async () => {

    if (!selectedFile) {

        alert("Please select an image.");

        return;

    }

    loading.style.display = "block";

    resultCard.style.display = "none";

    const formData = new FormData();

    formData.append("file", selectedFile);

    try {

        const response = await fetch(API_URL, {

            method: "POST",

            body: formData,

        });

        if (!response.ok) {

            throw new Error(
                `Server Error: ${response.status}`
            );

        }

        const result = await response.json();

        console.log(result);

        displayResult(result);

    }

    catch (error) {

        console.error(error);

        alert("Unable to connect to the prediction server.");

    }

    finally {

        loading.style.display = "none";

    }

});

// ==========================================
// Display Result
// ==========================================

function displayResult(result) {

    /*
        FastAPI Response

        {
            "class": "apple",
            "class_index": 0,
            "confidence": 91.63
        }
    */

    fruitName.textContent = result.class;

    confidence.textContent =
        result.confidence.toFixed(2) + "%";

    resultCard.style.display = "block";

}

// ==========================================
// Keyboard Shortcut
// ==========================================

document.addEventListener("keydown", (event) => {

    if (event.key === "Enter") {

        predictBtn.click();

    }

});