import os
from google import genai

API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set.")

client =  genai.client(api_key=API_KEY)   
resp = client.models.generate_content(    model="gemini-1.5-flash-001",    contents="What is the dua Qunut?")
print(resp.text)
print(resp)
