import os
import cv2
import csv
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# === CONFIGURATION ===
cnn_model_path = r"D:\deepfake-detection\Models\cnn_deepfake_model.h5"
rnn_model_path = r"D:\deepfake-detection\Models\rnn_audio_model.pth"
video_folder = r"D:\deepfake-detection\Samples"
output_csv = r"D:\deepfake-detection\Results\video_audio_predictions.csv"
temp_audio_path = r"D:\deepfake-detection\Temp\temp_audio.wav"
max_frames = 200
real_thresh = 0.35
fake_thresh = 0.70
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)

# === AUDIO LSTM MODEL ===
class AudioLSTMModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, output_size=2):
        super(AudioLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# === UTILITY FUNCTIONS ===
def extract_frames(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)
    cap.release()
    print(f"[INFO] Extracted {len(frames)} frames from {video_path}")
    return np.array(frames)

def extract_audio_ffmpeg(video_path, out_path):
    import subprocess
    try:
        command = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{out_path}"'
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(out_path)
    except Exception as e:
        print(f"[ERROR] Audio extraction failed: {e}")
        return False

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram.T[:100, :]  # Cap at 100 timesteps

# === PREDICTION FUNCTIONS ===
def evaluate_video_frames(model, frames, video_name):
    preds = model.predict(frames, verbose=0).flatten()
    print(f"[DEBUG] First 10 frame scores for {video_name}:")
    for i, p in enumerate(preds[:10]):
        print(f"  Frame {i+1}: Score = {p:.3f}")
    confident = [p for p in preds if p <= real_thresh or p >= fake_thresh]
    if len(confident) == 0:
        return "Uncertain", 0.0
    labels = (np.array(confident) >= 0.5).astype(int)
    real = (labels == 0).sum()
    fake = (labels == 1).sum()
    result = "Fake" if fake > real else "Real"
    confidence = round(max(real, fake) / len(labels), 2)
    return result, confidence

def evaluate_audio(rnn_model, video_path):
    if not extract_audio_ffmpeg(video_path, temp_audio_path):
        print(f"[ERROR] Audio extraction failed: {temp_audio_path}")
        return "Uncertain", 0.0
    try:
        spec = audio_to_spectrogram(temp_audio_path)
        input_tensor = torch.tensor(spec).unsqueeze(0).float()  # shape: (1, T, 128)
        rnn_model.eval()
        with torch.no_grad():
            output = rnn_model(input_tensor)
            pred = torch.softmax(output, dim=1)
            class_idx = pred.argmax(dim=1).item()
            confidence = round(pred.max().item(), 2)
            label = "Fake" if class_idx == 1 else "Real"
            return label, confidence
    except Exception as e:
        print(f"[ERROR] Spectrogram/audio error: {e}")
        return "Uncertain", 0.0

# === MAIN SCRIPT ===
print(f"[INFO] Loading CNN model from: {cnn_model_path}")
cnn_model = load_model(cnn_model_path)
print(f"[INFO] Loading RNN model from: {rnn_model_path}")
rnn_model = AudioLSTMModel()
rnn_model.load_state_dict(torch.load(rnn_model_path))
rnn_model.eval()

video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")]
results = []

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    print(f"\n🎥 Evaluating: {video_file}")

    frames = extract_frames(video_path, max_frames)
    vid_result, vid_conf = evaluate_video_frames(cnn_model, frames, video_file)
    aud_result, aud_conf = evaluate_audio(rnn_model, video_path)

    print(f"🎥 Video-Based: {vid_result} (Confidence: {vid_conf})")
    print(f"🎵 Audio-Based: {aud_result} (Confidence: {aud_conf})")

    results.append([video_file, vid_result, vid_conf, aud_result, aud_conf])

# === SAVE TO CSV ===
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Video Name", "Video Prediction", "Video Confidence", "Audio Prediction", "Audio Confidence"])
    writer.writerows(results)

print(f"\n✅ All predictions saved to: {output_csv}")
