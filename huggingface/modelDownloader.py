# Use a pipeline as a high-level helper
from transformers import pipeline
import time
start=time.time()
pipe = pipeline("object-detection", model="facebook/detr-resnet-50")
print("Pipeline loaded in", time.time() - start, "seconds")