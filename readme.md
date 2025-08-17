# llm1

A collection of deep learning, computer vision, and LLM experiments using local and remote models from Hugging Face, Google Gemini, YOLO, and more.

## Project Structure

- `classify_digit.py`  
  Classify handwritten digits using a PyTorch CNN (see `mnist_cnn.py`).

- `mnist_cnn.py`  
  PyTorch CNN for MNIST digit classification. Trains and saves a model.

- `huggingface/digitSiglip.py`  
  Gradio app for digit classification using SigLIP model from Hugging Face.

- `objectDetection/objectDetectionYolo.py`, `objectDetection/liveOD.py`, `dod.py`  
  Object detection using YOLOv8/v12 and Ultralytics, with live webcam and static image support.

- `objectDetection/resnet.py`, `objectDetection/resnetFB.py`, `objectDetection/resnetMS.py`, `objectDetection/fb2.py`, `objectDetection/resnetpFB.py`  
  Object detection and segmentation using Hugging Face Transformers (DETR, ResNet, panoptic segmentation).

- `huggingface/chat.py`, `huggingface/chatLocalModel.py`, `huggingface/chatNvidia.py`  
  LLM chat and text generation using Hugging Face endpoints and local models.

- `google/objectDetection.py`, `google/chat.py`, `google/google2.py`  
  Google Gemini API demos for object detection and chat (requires API key).

- `claudeChat.py`  
  Example of using Anthropic Claude API for image and text analysis.

- Data and model files:  
  - `data/MNIST/raw/` (MNIST dataset)  
  - `yolov8s.pt`, `yolo12l.pt`, etc. (YOLO weights)  
  - `nmi.jpg`, `test0.png`, etc. (test images)

## Requirements

- Python 3.8+
- torch, torchvision, transformers, huggingface_hub
- matplotlib, numpy, pillow, gradio, supervision, opencv-python
- ultralytics, datasets, openai, google-generativeai, anthropic

Install dependencies:
```powershell
pip install torch torchvision transformers huggingface_hub matplotlib numpy pillow gradio supervision opencv-python ultralytics datasets openai google-generativeai anthropic
```

## Usage Examples

### Digit Classification (PyTorch)
```powershell
python mnist_cnn.py         # Train and save CNN
python classify_digit.py    # Predict digit from image
```

### Gradio Digit Classifier (SigLIP)
```powershell
python huggingface/digitSiglip.py
```

### YOLO Object Detection
```powershell
python objectDetection/objectDetectionYolo.py   # Static image
python objectDetection/liveOD.py                # Live webcam
```

### Hugging Face DETR/ResNet Detection
```powershell
python objectDetection/resnetFB.py
python objectDetection/fb2.py
python objectDetection/resnetMS.py
python objectDetection/resnetpFB.py
```

### Google Gemini API (Object Detection/Chat)
```powershell
python google/objectDetection.py
python google/chat.py
python google/google2.py
```

### Anthropic Claude API
```powershell
python claudeChat.py
```

## Notes

- For Hugging Face API usage, set your token as an environment variable:
  ```powershell
  $env:HF_TOKEN="your_hf_token"
  ```
- For Google Gemini API usage, set your key as:
  ```powershell
  $env:GENAI_API_KEY="your_gemini_api_key"
  ```
- Place your test images (e.g., `nmi.jpg`, `test0.png`) in the project directory.
- Some scripts require specific model weights (e.g., YOLO). Download from Hugging Face or Ultralytics as needed.

## License

This project is for self learning and sharing.