"""
FAILURE:   When passing image_bytes, URL, filepath or image_b64: Bad request: 'NoneType' object has no attribute 'lower'

"""
import os
import base64
from huggingface_hub import InferenceClient

token = os.environ.get("HF_TOKEN")
print("Token:", token)
client = InferenceClient(model="facebook/detr-resnet-50", token=token)

with open("nmi.jpg", "rb") as f:
    image_bytes = f.read()

image_b64 = base64.b64encode(image_bytes)
#output = client.object_detection(image_b64)
output = client.object_detection("nmi.jpg")
#output = client.object_detection("https://staging.d1qeksakky7fvu.amplifyapp.com/NaeemIqbal.JPG")
print(output)