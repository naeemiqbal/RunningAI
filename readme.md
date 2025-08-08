# llm1

A collection of deep learning and computer vision experiments using pre-trained models from  TensorFlow, PyTorch, and Hugging Face Transformers.

## Project Structure

- `numberTest.py`  
  Test and visualize digit classification models (Keras/TensorFlow).

- `numFashion.py`, `numFashion.keras`  
  Fashion MNIST model and weights.

- `mnist_cnn.py`  
  PyTorch CNN for MNIST digit classification.

- `hf/resnet.py`, `hf/resnetFB2.py`  
  Object detection and segmentation using Hugging Face Transformers (DETR, ResNet).

## Requirements

- Python 3.8+
- TensorFlow
- PyTorch
- torchvision
- transformers
- huggingface_hub
- matplotlib
- numpy
- pillow
- torchinfo (for model summaries)

Install dependencies:
```bash
pip install tensorflow torch torchvision transformers huggingface_hub matplotlib numpy pillow torchinfo
```

## Usage

### Digit Classification (Keras)
```bash
python numberTest.py
```
- Loads a Keras model and tests it on sample images.
- Displays predictions and accuracy.

### MNIST CNN (PyTorch)
```bash
python mnist_cnn.py
```
- Trains a simple CNN on MNIST and saves the model.

### Object Detection (Hugging Face)
```bash
python hf/resnet.py
```
- Runs object detection on a local image using Hugging Face Inference API.

### DETR Object Detection/Segmentation (Transformers)
```bash
python hf/resnetFB2.py
```
- Runs DETR-based object detection or segmentation on an image.
- Visualizes results with bounding boxes and labels.

## Notes

- For Hugging Face API usage, set your token as an environment variable:
  ```powershell
  $env:HF_TOKEN="your_hf_token"
  ```
- Place your test images (e.g., `nmi.jpg`) in the project directory.

## License

This project is for self learning and sharing. 