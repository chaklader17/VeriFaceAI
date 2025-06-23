import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def extract_frames_from_video(video_path, num_frames=5):
    """Extract evenly spaced resized frames from a video."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[ERROR] No frames found in: {video_path}")
        return frames

    num_frames = min(num_frames, total_frames)
    frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return frames

def load_data(real_dir, fake_dir, max_videos_per_class=100):
    """Load a fixed number of frames per video and label them."""
    X, y = [], []

    print("[INFO] Loading real videos...")
    real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    real_paths = real_paths[:max_videos_per_class]

    for path in tqdm(real_paths):
        frames = extract_frames_from_video(path, num_frames=5)
        X.extend(frames)
        y.extend([0] * len(frames))  # 0 = real

    print("[INFO] Loading fake videos...")
    fake_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.mp4')]
    fake_paths = fake_paths[:max_videos_per_class]

    for path in tqdm(fake_paths):
        frames = extract_frames_from_video(path, num_frames=5)
        X.extend(frames)
        y.extend([1] * len(frames))  # 1 = fake

    X = np.array([preprocess_input(img_to_array(img)) for img in X])
    y = to_categorical(y, num_classes=2)

    return X, y

def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    real_dir = r"D:\deepfake-detection\Datasets\Celeb-DF-v2\Celeb-real"
    fake_dir = r"D:\deepfake-detection\Datasets\Celeb-DF-v2\Celeb-synthesis"

    print("[INFO] Loading and preprocessing data...")
    X, y = load_data(real_dir, fake_dir, max_videos_per_class=100)  # adjust value based on RAM
    print(f"[INFO] Dataset shape: {X.shape}, Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Building and training model...")
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=8)

    model_path = r"D:\deepfake-detection\Models\cnn_deepfake_model.h5"
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

if __name__ == "__main__":
    main()
