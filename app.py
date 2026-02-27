import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from ultralytics import YOLO
import os

# Fix for PyTorch 2.6+ weights_only=True security restriction
# This allows YOLO models to load safely
try:
    from ultralytics.nn.tasks import DetectionModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

# Initialize FastAPI app
app = FastAPI(title="YOLOv8 Object Detection & Identification")

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

def load_model():
    global model
    if model is None:
        try:
            # Check for model in multiple potential locations
            paths_to_check = [MODEL_PATH, "yolov8n.pt", "/app/yolov8n.pt"]
            final_path = next((p for p in paths_to_check if os.path.exists(p)), "yolov8n.pt")
            
            # Use weights_only=False for local model loading if needed via environment
            # but usually YOLO() handles this if classes are in safe_globals
            model = YOLO(final_path)
            print(f"Successfully loaded YOLO model from {final_path}")
        except Exception as e:
            print(f"Critical Error: Could not load model: {e}")
            model = None
    return model

@app.get("/", response_class=HTMLResponse)
async def index():
    """Simple UI to capture a photo and detect objects."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 Object Identification</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 20px; background: #f0f2f5; color: #333; }
            #container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
            .video-wrap { position: relative; margin-bottom: 20px; background: #000; border-radius: 10px; overflow: hidden; line-height: 0; }
            video, #display-canvas { width: 100%; height: auto; border-radius: 10px; }
            #display-canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
            .controls { margin-bottom: 20px; }
            button { padding: 15px 40px; font-size: 18px; font-weight: bold; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 8px; transition: transform 0.1s, background 0.2s; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            button:hover { background: #0056b3; }
            button:active { transform: scale(0.98); }
            button:disabled { background: #ccc; cursor: not-allowed; }
            #results-panel { text-align: left; margin-top: 10px; padding: 15px; background: #f8f9fa; border-left: 5px solid #28a745; border-radius: 5px; min-height: 60px; }
            .object-label { display: inline-block; background: #28a745; color: white; padding: 4px 10px; border-radius: 20px; font-size: 14px; margin: 3px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>🔍 YOLOv8 Automatic Identification</h1>
            <div class="video-wrap">
                <video id="video" autoplay playsinline></video>
                <canvas id="display-canvas"></canvas>
            </div>
            <div class="controls">
                <button id="capture">IDENTIFY OBJECTS</button>
            </div>
            <div id="results-panel">
                <strong>Status:</strong> <span id="status">Point camera and click Identify</span>
            </div>
            <canvas id="hidden-canvas" style="display:none;"></canvas>
        </div>
        <script>
            const video = document.getElementById('video');
            const displayCanvas = document.getElementById('display-canvas');
            const hiddenCanvas = document.getElementById('hidden-canvas');
            const captureButton = document.getElementById('capture');
            const statusSpan = document.getElementById('status');
            const displayCtx = displayCanvas.getContext('2d');

            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
                .then(stream => { 
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {
                        displayCanvas.width = video.videoWidth;
                        displayCanvas.height = video.videoHeight;
                    };
                })
                .catch(err => { 
                    statusSpan.innerText = "Error accessing camera: " + err; 
                    captureButton.disabled = true; 
                });

            captureButton.onclick = async () => {
                const hiddenCtx = hiddenCanvas.getContext('2d');
                hiddenCanvas.width = video.videoWidth;
                hiddenCanvas.height = video.videoHeight;
                hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

                hiddenCanvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('file', blob, 'capture.jpg');
                    
                    statusSpan.innerText = "Analyzing image...";
                    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
                    
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        
                        if (data.detections && data.detections.length > 0) {
                            let labelsHtml = '';
                            const uniqueNames = [...new Set(data.detections.map(d => d.class.toUpperCase()))];
                            uniqueNames.forEach(name => {
                                labelsHtml += `<span class="object-label">${name}</span> `;
                            });
                            
                            statusSpan.innerHTML = `Identified: ${labelsHtml}`;
                            
                            // Draw bounding boxes
                            displayCtx.strokeStyle = '#28a745';
                            displayCtx.lineWidth = 4;
                            displayCtx.fillStyle = '#28a745';
                            displayCtx.font = 'bold 20px Arial';
                            
                            data.detections.forEach(d => {
                                const {x1, y1, x2, y2} = d.bbox;
                                displayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                                // Draw background for label
                                const label = `${d.class} ${(d.confidence * 100).toFixed(0)}%`;
                                const textWidth = displayCtx.measureText(label).width;
                                displayCtx.fillRect(x1, y1 > 25 ? y1 - 25 : y1, textWidth + 10, 25);
                                displayCtx.fillStyle = 'white';
                                displayCtx.fillText(label, x1 + 5, y1 > 25 ? y1 - 5 : y1 + 20);
                                displayCtx.fillStyle = '#28a745';
                            });
                        } else {
                            statusSpan.innerText = "No recognizable objects found.";
                        }
                    } catch (err) { statusSpan.innerText = "Error: " + err; }
                }, 'image/jpeg');
            };
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    load_model()
    return {"status": "healthy" if model else "unhealthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Automatic object detection and identification."""
    model_instance = load_model()
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

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

@app.get("/health")
async def health_check():
    """Health check endpoint for Render monitoring."""
    load_model()
    return {"status": "healthy" if model else "unhealthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...), target: str = None):
    """Object detection endpoint using YOLOv8 nano."""
    model_instance = load_model()
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Use CPU for inference on Render (Free tier doesn't have GPU)
        # ultralytics handles resizing internally, preserving aspect ratio
        with torch.no_grad():
            results = model_instance.predict(source=image, imgsz=640, device="cpu", verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model_instance.names.get(cls, str(cls))
                
                # Filter by target if provided
                if target and target.lower() not in name.lower():
                    continue

                detections.append({
                    "class": name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": round(x1,2), "y1": round(y1,2), "x2": round(x2,2), "y2": round(y2,2)}
                })

        return {
            "detections": detections, 
            "count": len(detections),
            "target_filtered": target is not None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), workers=1)