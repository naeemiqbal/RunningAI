from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
#print ( image.size, image.mode   )
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
#print(f"Detected {model.config.id2label[label.item()]} with confidence "          f"{round(score.item(), 3)} at location {box}")    

"""
imgs = dataset["test"]["image"]


for img  in imgs:
        plt.subplot(6, 5, j*5+ i + 1)
        plt.imshow(img)
        plt.title(sub_class)
            plt.axis('off')
    j += 1
plt.tight_layout()
plt.show()
"""
