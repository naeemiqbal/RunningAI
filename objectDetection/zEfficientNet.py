#Bad

import torch
from datasets import load_dataset
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from PIL import Image

image = Image.open("nmi.jpg")
preprocessor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b7")
model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")

inputs = preprocessor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
print (f"Predicted class: {predicted_label} - {model.config.id2label[predicted_label]}")
print( f"{predicted_label} \n {logits} \nEnd.")
