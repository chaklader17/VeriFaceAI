#from tensorflow.keras.models import load_model
#from tensorflow.keras.applications.resnet50 import preprocess_input
# tensorflow.keras.preprocessing.image import img_to_array
#mport numpy as np

#print("TensorFlow and Keras imports successful.")
import os

real_dir = r"D:\deepfake-detection\frames\real"
fake_dir = r"D:\deepfake-detection\frames\fake"

print("📸 Real frames:", len(os.listdir(real_dir)))
print("🧪 Fake frames:", len(os.listdir(fake_dir)))
