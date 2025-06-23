import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

AUDIO_DIR = "D:/deepfake-detection/Audio"
SPECTROGRAM_DIR = "D:/deepfake-detection/Spectrograms"

os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

def convert_to_spectrogram(audio_path, output_folder):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar()

        audio_name = os.path.basename(audio_path).replace(".wav", ".png")
        plt.savefig(os.path.join(output_folder, audio_name))
        plt.close()
        return True
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return False

audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

if len(audio_files) == 0:
    print("❌ No audio files found! Run `extract_audio.py` first.")
else:
    for audio_file in tqdm(audio_files, desc="Generating spectrograms"):
        convert_to_spectrogram(os.path.join(AUDIO_DIR, audio_file), SPECTROGRAM_DIR)

print("🎉 Spectrogram generation complete!")
