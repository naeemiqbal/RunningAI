import anthropic

client = anthropic.Anthropic()

# Upload the image file
with open("test0.png", "rb") as f:
    file_upload = client.beta.files.upload(file=("test0.png", f, "image/png"))

# Use the uploaded file in a message
message = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    betas=["files-api-2025-04-14"],
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

print(message.content)