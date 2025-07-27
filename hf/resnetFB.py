from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url= "https://staging.d1qeksakky7fvu.amplifyapp.com/NaeemIqbal.JPG"
image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("nmi.jpg")

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.5
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

print(results)
fig, ax = plt.subplots()
ax.imshow(image)
  
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    labelName = model.config.id2label[label.item()]
    print(  f"Detected {labelName} with confidence {round(score.item(), 3)} at location {box}")
    x0, y0, x1, y1 = box
    w,h  = x1-x0, y1-y0
    box = patches.Rectangle((x0, y0), w, h, edgecolor="red", facecolor="none")
    ax.add_patch(box)
    if labelName is not None:
        ax.annotate(labelName, (x0, y0 - 10), fontsize=8, color='white', weight='bold', bbox=dict(facecolor='red', alpha=0.5))
plt.axis("off")
plt.show()

n = len(results["scores"])
print (f"Total items found  {n} \nDone.")