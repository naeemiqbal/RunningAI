import io
import requests
from PIL import Image
import torch
import numpy

from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# forward pass
outputs = model(**inputs)

# use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
results = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
print(results)


import matplotlib.pyplot as plt
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

# the segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(results["png_string"]))
panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
# retrieve the ids corresponding to each mask
panoptic_seg_id = rgb_to_id(panoptic_seg)

