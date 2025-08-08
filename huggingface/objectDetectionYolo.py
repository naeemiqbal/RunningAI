# download the model from https://huggingface.co/ultralytics/yolo12l.pt
# >>yolo task=detect mode=predict model=yolov12l.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=True

import cv2
from ultralytics import YOLO
import supervision as sv


image_path = "dog.jpeg" #In parent foler, replace with your image path
image = cv2.imread(image_path)

model = YOLO('yolo12l.pt')

results = model(image, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = image.copy()
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
print(detections)
sv.plot_image(annotated_image)