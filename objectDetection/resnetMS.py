from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("nmi.jpg")
plt.imshow(image)
plt.show()

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
print (f"Predicted class: {predicted_label} - {model.config.id2label[predicted_label]}")
print( f"{predicted_label} \n {logits} \nEnd.")