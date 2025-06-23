import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

model_path = r"D:\deepfake-detection\Models\cnn_deepfake_model.h5"
video_folder = r"D:\deepfake-detection\Samples"
output_csv = r"D:\deepfake-detection\Results\video_predictions.csv"
max_frames = 200

real_confident_thresh = 0.35
fake_confident_thresh = 0.70


enable_plot = True

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
def extract_frames(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (224, 224))
        frame_preprocessed = preprocess_input(frame_resized.astype(np.float32))
        frames.append(frame_preprocessed)
        count += 1
    cap.release()
    print(f"[INFO] Extracted {count} frames from {video_path}")
    return np.array(frames)

def evaluate_video(model, frames, video_name):
    print(f"[INFO] Predicting on {len(frames)} frames for {video_name}...")
    preds = model.predict(frames, verbose=0).flatten()

    if enable_plot:
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(preds)), preds, color='skyblue')
        plt.axhline(y=0.5, color='r', linestyle='--', label="0.5 threshold")
        plt.axhline(y=fake_confident_thresh, color='orange', linestyle='--', label="Fake threshold")
        plt.axhline(y=real_confident_thresh, color='green', linestyle='--', label="Real threshold")
        plt.title(f"Frame Predictions: {video_name}")
        plt.xlabel("Frame Index")
        plt.ylabel("Predicted Score")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"[DEBUG] First 10 frame scores:")
    for i, p in enumerate(preds[:10]):
        print(f"  Frame {i+1}: Score = {p:.3f}")

    confident_scores = [p for p in preds if p <= real_confident_thresh or p >= fake_confident_thresh]
    print(f"[DEBUG] Used {len(confident_scores)} confident frames out of {len(preds)} total")

    if len(confident_scores) == 0:
        print(f"⚠️ No confident predictions for {video_name}. Marking as Uncertain.\n")
        return "Uncertain", 0.0, 0, 0


    confident_scores = np.array(confident_scores)
    pred_labels = (confident_scores >= 0.5).astype(int)
    real_count = (pred_labels == 0).sum()
    fake_count = (pred_labels == 1).sum()
    total = len(pred_labels)

    print(f"[DEBUG] Real frames: {real_count}, Fake frames: {fake_count}")

    if fake_count > real_count:
        prediction = "Fake"
        confidence = round(fake_count / total, 2)
    else:
        prediction = "Real"
        confidence = round(real_count / total, 2)

    print(f"📼 {video_name} → Prediction: {prediction} (Confidence: {confidence})\n")
    return prediction, confidence, real_count, fake_count

#MAIN
print(f"[INFO] Loading model from: {model_path}")
model = load_model(model_path)

results = []
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")]

for file_name in video_files:
    full_path = os.path.join(video_folder, file_name)
    frames = extract_frames(full_path, max_frames)
    prediction, confidence, real_frames, fake_frames = evaluate_video(model, frames, file_name)
    results.append([file_name, prediction, confidence, real_frames, fake_frames])

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Video Name", "Prediction", "Confidence", "Real Frames", "Fake Frames"])
    writer.writerows(results)

print(f"\n✅ Predictions saved to: {output_csv}")
