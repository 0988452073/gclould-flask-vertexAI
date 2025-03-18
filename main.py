from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np

app = Flask(__name__)

# Vertex AI API 端點
VERTEX_AI_ENDPOINT = "https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/YOUR_ENDPOINT_ID:predict"

# Google Cloud 認證
HEADERS = {"Authorization": "Bearer YOUR_ACCESS_TOKEN"}

@app.route("/")
def home():
    return "Cloud Run + Vertex AI YOLO API is running!"

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 轉換影像為 Tensor
    img_resized = cv2.resize(img, (640, 640))  # YOLOv8 標準輸入尺寸
    payload = {"instances": [{"image": img_resized.tolist()}]}

    # 發送請求到 Vertex AI
    response = requests.post(VERTEX_AI_ENDPOINT, json=payload, headers=HEADERS)

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Vertex AI error", "details": response.text}), 500

if __name__ == "__main__":
    from os import environ
    port = int(environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)