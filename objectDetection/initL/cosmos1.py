import requests
import streamlit as st
from dotenv import load_dotenv
import os

invoke_url = "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos-1.0-7b-diffusion-text2world"
fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json",
}

# Streamlit UI
st.title("NVIDIA Text2World")
prompt = st.text_area("Enter your prompt:", "A first person view from the perspective from a human sized robot as it works in a chemical plant. The robot has many boxes and supplies nearby on the industrial shelves. The camera on moving forward, at a height of 1m above the floor. Photorealistic")

if st.button("Generate"):
    payload = {
      "inputs": [
        {
          "name": "command",
          "shape": [1],
          "datatype": "BYTES",
          "data": [
            f"text2world --prompt=\"{prompt}\""
          ]
        }
      ],
      "outputs": [
        {
          "name": "status",
          "datatype": "BYTES",
          "shape": [1]
        }
      ]
    }

    # re-use connections
    session = requests.Session()

    response = session.post(invoke_url, headers=headers, json=payload)

    while response.status_code == 202:
        request_id = response.headers.get("NVCF-REQID")
        fetch_url = fetch_url_format + request_id
        response = session.get(fetch_url, headers=headers)
    
    response.raise_for_status()

    with open('result.zip', 'wb') as f:
        f.write(response.content)
    
    st.success("Generation complete! Check the result.zip file.")