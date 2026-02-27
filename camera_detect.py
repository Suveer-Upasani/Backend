import cv2
from ultralytics import YOLO

def main():
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        results = model.predict(source=frame, stream=True, imgsz=640, verbose=False)

        for r in results:
            annotated_frame = r.plot()
            detected_classes = [model.names[int(box.cls[0])] for box in r.boxes]
            if detected_classes:
                print(f"Detected: {', '.join(set(detected_classes))}")
            cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()