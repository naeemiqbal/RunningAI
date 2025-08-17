from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from mnist_cnn import MNISTModel

# Load image
img = Image.open("test0.png").convert("L")

# Resize and invert (black background, white digit)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img_tensor = transform(img).unsqueeze(0)

# Load model
model = MNISTModel()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

# Predict
with torch.no_grad():
    output = model(img_tensor)
    pred = output.argmax(dim=1, keepdim=True)
    print(f"Predicted Digit: {pred.item()}")
