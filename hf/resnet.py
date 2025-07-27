import os
import base64
from huggingface_hub import InferenceClient

token = os.environ.get("HF_TOKEN")
client = InferenceClient(model="facebook/detr-resnet-50", token=token)

with open("nmi.jpg", "rb") as f:
    image_bytes = f.read()

image_b64 = base64.b64encode(image_bytes).decode("utf-8")
print("Image size:", len(image_b64))
#output = client.object_detection(image_b64)

#output = client.object_detection("nmi.jpg")
output = client.object_detection("https://staging.d1qeksakky7fvu.amplifyapp.com/NaeemIqbal.JPG")
print(output)