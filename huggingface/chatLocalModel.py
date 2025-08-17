# Use a pipeline as a high-level helper
from transformers import pipeline
import time
start = time.time()
pipe = pipeline("text-generation", model="moonshotai/Kimi-K2-Instruct", trust_remote_code=True)
messages = [
    {"role": "user", "content": "How to pray Witr?"},
]
pipe(messages)
print(f"Pipeline took {time.time() - start:.2f} seconds")   
