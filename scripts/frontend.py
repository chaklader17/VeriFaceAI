import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading

import cv2
import numpy as np
import torch
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch.nn as nn

# === Paths and Models ===
cnn_model_path = r"D:\deepfake-detection\Models\cnn_deepfake_model.h5"
rnn_model_path = r"D:\deepfake-detection\Models\rnn_audio_model.pth"
temp_audio_path = r"D:\deepfake-detection\Temp\temp_audio.wav"
real_thresh = 0.35
fake_thresh = 0.70
max_frames = 200

# === Audio RNN Model ===
class AudioLSTMModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, output_size=2):
        super(AudioLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# === Load Models ===
cnn_model = load_model(cnn_model_path)
rnn_model = AudioLSTMModel()
rnn_model.load_state_dict(torch.load(rnn_model_path))
rnn_model.eval()

# === Processing Functions ===
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
    return np.array(frames)

def extract_audio(video_path):
    import subprocess
    try:
        command = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_audio_path}"'
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(temp_audio_path)
    except:
        return False

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spec = librosa.power_to_db(spec, ref=np.max)
    return log_spec.T[:100, :]

def predict_video(frames):
    preds = cnn_model.predict(frames, verbose=0).flatten()
    confident = [p for p in preds if p <= real_thresh or p >= fake_thresh]
    if not confident:
        return "Uncertain", 0.0
    labels = (np.array(confident) >= 0.5).astype(int)
    real, fake = (labels == 0).sum(), (labels == 1).sum()
    label = "Fake" if fake > real else "Real"
    confidence = round(max(real, fake) / len(labels), 2)
    return label, confidence

def predict_audio(video_path):
    if not extract_audio(video_path): return "Uncertain", 0.0
    try:
        spec = audio_to_spectrogram(temp_audio_path)
        tensor = torch.tensor(spec).unsqueeze(0).float()
        with torch.no_grad():
            out = rnn_model(tensor)
            pred = torch.softmax(out, dim=1)
            label = "Fake" if pred.argmax(dim=1).item() == 1 else "Real"
            return label, round(pred.max().item(), 2)
    except:
        return "Uncertain", 0.0

# === GUI Logic ===
def run_detection(video_path):
    result_text.set("Analyzing... Please wait.")
    frames = extract_frames(video_path, max_frames)
    video_result, video_conf = predict_video(frames)
    audio_result, audio_conf = predict_audio(video_path)

    final = f"""
📼 Video: {video_result} ({video_conf * 100:.1f}%)
🎵 Audio: {audio_result} ({audio_conf * 100:.1f}%)
"""
    result_text.set(final)

def browse_file():
    path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if path:
        video_path.set(path)
        result_text.set("")
        threading.Thread(target=run_detection, args=(path,), daemon=True).start()

# === GUI Setup ===
root = tk.Tk()
root.title("Deepfake Detection System")
root.configure(bg="white")
root.geometry("500x350")

style = ttk.Style(root)
style.configure("TButton", font=("Segoe UI", 10), padding=6)
style.configure("TLabel", font=("Segoe UI", 10), background="white")

tk.Label(root, text="🎬 Select a Video for Deepfake Analysis", font=("Segoe UI", 14, "bold"), bg="white").pack(pady=15)

video_path = tk.StringVar()
result_text = tk.StringVar()

ttk.Button(root, text="Browse Video", command=browse_file).pack(pady=10)
ttk.Label(root, textvariable=video_path, wraplength=450).pack(pady=5)

tk.Label(root, text="Result", font=("Segoe UI", 12, "bold"), bg="white").pack(pady=10)
result_label = tk.Label(root, textvariable=result_text, font=("Segoe UI", 11), bg="white", justify="left")
result_label.pack()

root.mainloop()
