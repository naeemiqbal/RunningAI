import torch
import requests

from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor

import matplotlib.pyplot as plt
import matplotlib.patches as patches

#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("nmi.jpg")

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-nano-coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-nano-coco")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)



fig, ax = plt.subplots()
ax.imshow(image)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), model.config.id2label[label_id.item()]
        box = [round(i, 2) for i in box.tolist()]
        print(f"{label}: {score:.2f} {box}")
        x0, y0, x1, y1 = box
        w,h  = x1-x0, y1-y0
        ibox = patches.Rectangle((x0, y0), w, h, edgecolor="red", facecolor="none")
        ax.add_patch(ibox)
        if label is not None:
            ax.annotate(label, (x0, y0 - 10), fontsize=8, color='white', weight='bold', bbox=dict(facecolor='red', alpha=0.5))
plt.axis("off")
plt.show()
