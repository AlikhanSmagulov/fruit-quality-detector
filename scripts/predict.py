import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Detect project root automatically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fruit_quality_mobilenetv2.h5")
VAL_PATH = os.path.join(PROJECT_ROOT, "dataset", "val")

# Load model
model = load_model(MODEL_PATH)

# Get available classes from val folder
classes_in_val = [d for d in os.listdir(VAL_PATH) if os.path.isdir(os.path.join(VAL_PATH, d))]
labels = classes_in_val  # use same order as folder names

# Pick first image from first available class
first_class_folder = os.path.join(VAL_PATH, labels[0])
first_image = os.listdir(first_class_folder)[0]
test_image = os.path.join(first_class_folder, first_image)

# Prediction
def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)[0]
    class_id = np.argmax(pred)
    confidence = pred[class_id]
    return labels[class_id], float(confidence)

# Run prediction
if __name__ == "__main__":
    result, conf = predict_image(test_image)
    print(f"Predicted: {result} | Confidence: {conf:.2f}")
