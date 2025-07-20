import google.generativeai as genai

# Initialize the Generative AI client
model = genai.GenerativeModel( "gemini-1.5-flash-001")  # or genai.GenerativeAI
#.Model("gemini-1.5-flash-001")

# Upload the image file
with open("test0.png", "rb") as f:
    file_upload = model.files.upload(file=("test0.png", f, "image/png"))

model.files.wait_for_upload(file_upload.id)
# Use the uploaded file in a message
message = model.chat.completions.create(
    model="gemini-1.5-flash-001",   
    max_output_tokens=1024,
    messages=[
        {   
            "role": "user",
            "content": [
                {
                    "type": "image",    
                    "source": {
                        "type": "file",
                        "file_id": file_upload.id   
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }   
            ]
        }
    ],
)
print(message.candidates[0].content)
