"""

import google.generativeai as genai

genai.configure(api_key="AIzaSyDbDr_ryHU75gbVRUo2oTVOUUEi3qvHNUg")
# Initialize the Generative AI client
model = genai.GenerativeModel( "gemini-1.5-flash-001")


#rslt= model.generate_text(    prompt="What is the dua Qunut?",    max_output_tokens=50,    temperature=0.5).result()

rslt = model.generate_content("What is the dua Qunut?")
print(rslt.text)
print(rslt)

"""

from google import genai


client =  genai.client(api_key="AIzaSyDbDr_ryHU75gbVRUo2oTVOUUEi3qvHNUg"    )   
resp = client.models.generate_content(    model="gemini-1.5-flash-001",    contents="What is the dua Qunut?")
print(resp.text)
print(resp)
