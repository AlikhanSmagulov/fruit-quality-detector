# 🍎 Fruit Quality Detection Using Computer Vision

This project is a **computer vision-based application** that classifies fruits as **fresh or rotten** based on images. It uses **Transfer Learning with MobileNetV2** and a **Tkinter modern GUI** for an interactive desktop experience.

---

## **📝 Project Description**

The application allows users to:

- Upload or select fruit images.
- Detect fruit quality (fresh or rotten) with high accuracy.
- Visualize predictions with confidence percentages.
- Use a modern, user-friendly GUI with light/dark mode.

The project is built using **Python**, **TensorFlow**, and **Tkinter**, and is designed for **Windows, Linux, and macOS**.

---

## **✨ Features**

- Detect multiple types of fruits: Apple, Banana, Orange (both fresh and rotten).  
- Top-3 prediction results with confidence scores.  
- Image preview before prediction.  
- Responsive and modern GUI using **Tkinter + ttk styling**.  
- Light/Dark mode toggle for comfortable use.  
- Progress indicator during predictions to keep the UI responsive.  

---

## **⚙️ Software Used**

- Python 3.10+  
- TensorFlow 2.x  
- Keras (included in TensorFlow)  
- Pillow (for image processing)  
- NumPy  
- scikit-learn (for dataset splitting)  
- Tkinter (for GUI)  

**Optional (for packaging as .exe):**  
- PyInstaller  

---

## 🚀 How to Use

Follow these steps to set up and use the Fruit Quality Detection project:

---

### 1️⃣ Clone the Repository

Clone the project repository to your local machine:

git clone https://github.com/your-username/fruit-quality-detection.git
cd fruit-quality-detection

### 2️⃣ Install Dependencies

Make sure you have Python 3.10+ installed. Then install the required Python packages:

pip install tensorflow pillow numpy scikit-learn

Optional (for packaging as a desktop app later):

pip install pyinstaller

### 3️⃣ Prepare the Dataset

Organize your dataset in the dataset/train/ folder by class names. Each folder should contain images corresponding to that class:

dataset/train/
    FreshApple/
    RottenApple/
    FreshBanana/
    RottenBanana/
    FreshOrange/
    RottenOrange/


Ensure your folder names match exactly with the class labels you want the model to predict.

### 4️⃣ Create Validation Set

Run the script to automatically split 20% of images from each class into a validation folder (dataset/val/):

python scripts/prepare_dataset.py


After running, your dataset folder structure will look like this:

dataset/
    train/    # 80% of images
    val/      # 20% of images for validation

### 5️⃣ Train the Model

Train the MobileNetV2 model using transfer learning:

python scripts/train_model.py


The script loads images from dataset/train/ and dataset/val/.

Data augmentation is applied: rotation, zoom, and horizontal flip.

Early stopping prevents overfitting.

After training, the model will be saved as:

models/fruit_quality_mobilenetv2.h5

### 6️⃣ Test Command-Line Prediction

You can test a single image using the command-line script:

python scripts/predict.py


By default, the script picks one image from the validation set.

Example output:

Predicted: Fresh Apples | Confidence: 0.998


To predict any other image, update the test_image path inside the script.

### 7️⃣ Run the GUI

Run the Tkinter modern GUI for an interactive desktop experience:

python scripts/gui_modern.py


Steps inside the app:

Click Choose Image → select a fruit image.

Click Analyze → the app will display:

Top-1 predicted class

Top-3 predictions with confidence scores

Confidence bar

Toggle Light/Dark Mode using the button at the top.

The GUI uses a progress spinner while predicting, keeping the interface responsive.


