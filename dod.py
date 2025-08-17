"""
from ultralytics import YOLO
import json

model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt or yolov8m.pt
results = model("test1.jpg", show=True)  # Change to your image path

# Extract predictions
detections = []
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    label = results[0].names[cls_id]
    confidence = float(box.conf[0])
    bbox = box.xyxy[0].tolist()
    detections.append({"label": label, "confidence": confidence, "box": bbox})

# Save or print
print(json.dumps(detections, indent=2))

"""
from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")  # or yolov8s.pt for better accuracy

results = model("test1.png", show=True)

# To get all detected labels:
for r in results:
    print(r.names)
    print(r.boxes.cls)
