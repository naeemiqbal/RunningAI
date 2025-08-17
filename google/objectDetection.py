import os
import cv2
import json
import numpy as np
import google.generativeai as genai
from PIL import Image
import io

# --- Configuration ---
# Ensure your API key is set as an environment variable
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)

# Define the expected JSON response structure for the Gemini API
detection_schema = {\
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "name": {
                "type": "STRING",
                "description": "The name of the detected object (e.g., 'laptop', 'cup')."
            },
            "box": {
                "type": "OBJECT",
                "description": "Normalized bounding box [0-1].",
                "properties": {
                    "x": {"type": "NUMBER"},
                    "y": {"type": "NUMBER"},
                    "width": {"type": "NUMBER"},
                    "height": {"type": "NUMBER"},
                },
                "required": ["x", "y", "width", "height"]
            }
        },
        "required": ["name", "box"]
    }
}

# Use a capable model that supports JSON output
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": detection_schema,
    }
)

# Colors for drawing bounding boxes (in BGR format for OpenCV)
BOUNDING_BOX_COLORS = [
    (246, 130, 59),  # Blue
    (94, 197, 34),   # Green
    (68, 68, 239),   # Red
    (8, 179, 234),   # Yellow
    (247, 85, 168),  # Purple
]

# --- Main Functions ---

def detect_objects_from_frame(frame: np.ndarray):
    """Encodes a frame, sends it to Gemini, and parses the result."""
    # Convert numpy array (OpenCV frame) to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Save image to a byte stream to send to the API
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    
    # Prepare API request parts
    image_part = {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
    prompt_part = "Identify all distinct objects in this image. For each object, provide its name and a normalized bounding box."
    
    print("🤖 Sending image to Gemini for analysis...")
    try:
        response = model.generate_content([prompt_part, image_part])
        # The API is configured to return JSON, so we can load it directly
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        return None

def draw_results_on_image(image: np.ndarray, objects: list):
    """Draws bounding boxes and labels on the image."""
    h, w, _ = image.shape
    for i, obj in enumerate(objects):
        box = obj.get('box', {})
        name = obj.get('name', 'Unknown')
        
        # Denormalize coordinates
        x1 = int(box.get('x', 0) * w)
        y1 = int(box.get('y', 0) * h)
        width = int(box.get('width', 0) * w)
        height = int(box.get('height', 0) * h)
        x2, y2 = x1 + width, y1 + height
        
        # Draw bounding box
        color = BOUNDING_BOX_COLORS[i % len(BOUNDING_BOX_COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(name.capitalize(), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image, name.capitalize(), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    return image

# --- Main Application Loop ---

def main():
    """Starts the camera and handles user interaction."""
    cap = cv2.VideoCapture(0) # 0 is the default camera
    if not cap.isOpened():
        print("❌ Error: Cannot open camera.")
        return

    print("\n📷 Camera feed started.")
    print("   Press 'c' to capture and detect objects.")
    print("   Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Can't receive frame. Exiting...")
            break
        
        # Flip the frame horizontally for a more natural mirror-like view
        frame = cv2.flip(frame, 1)

        cv2.imshow('Live Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        if key == ord('c'):
            print("\nCapturing frame...")
            detected_objects = detect_objects_from_frame(frame)
            
            if detected_objects:
                print("\n✅ Detection Complete! Found objects:")
                for i, obj in enumerate(detected_objects):
                    print(f"  {i+1}. {obj.get('name', 'N/A').capitalize()}")

                result_image = draw_results_on_image(frame.copy(), detected_objects)
                cv2.imshow('Detection Results', result_image)
                print("\n-> Press any key on the 'Detection Results' window to continue.")
                cv2.waitKey(0) # Wait indefinitely for a key press on the result window
                cv2.destroyWindow('Detection Results')
            else:
                print("-> Could not detect any objects. Please try again.")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Application closed.")

if __name__ == '__main__':
    main()