import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from ultralytics import YOLO
import os

# Initialize FastAPI app
app = FastAPI(title="YOLOv8 Nano Object Detection API")

MODEL_PATH = "/app/models/yolov8n.pt"
model = None

def load_model():
    global model
    if model is None:
        try:
            model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Critical Error: Could not load model: {e}")
            model = None
    return model

@app.get("/", response_class=HTMLResponse)
async def index():
    """Simple UI to capture a photo from the camera."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 Camera Detection</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 20px; background: #f4f4f9; }
            #container { max-width: 640px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            video, canvas { width: 100%; border-radius: 5px; margin-bottom: 10px; }
            button { padding: 12px 24px; font-size: 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; }
            button:disabled { background: #ccc; }
            #results { text-align: left; margin-top: 20px; padding: 10px; background: #eee; border-radius: 5px; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>YOLOv8 Nano Detector</h1>
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas" style="display:none;"></canvas>
            <button id="capture">Capture & Detect</button>
            <h3>Results:</h3>
            <div id="results">Waiting for capture...</div>
        </div>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const captureButton = document.getElementById('capture');
            const resultsDiv = document.getElementById('results');
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { resultsDiv.innerText = "Error accessing camera: " + err; captureButton.disabled = true; });

            captureButton.onclick = async () => {
                const context = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('file', blob, 'capture.jpg');
                    resultsDiv.innerText = "Detecting...";
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        resultsDiv.innerText = JSON.stringify(data, null, 2);
                    } catch (err) { resultsDiv.innerText = "Error: " + err; }
                }, 'image/jpeg');
            };
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint for Render monitoring."""
    load_model()
    return {"status": "healthy" if model else "unhealthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Object detection endpoint using YOLOv8 nano."""
    model_instance = load_model()
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((640, 640))  # Resize for YOLOv8

        with torch.no_grad():
            results = model_instance.predict(source=image, imgsz=640, device="cpu", verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model_instance.names.get(cls, str(cls))
                detections.append({
                    "class": name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": round(x1,2), "y1": round(y1,2), "x2": round(x2,2), "y2": round(y2,2)}
                })

        return {"detections": detections, "count": len(detections)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), workers=1)