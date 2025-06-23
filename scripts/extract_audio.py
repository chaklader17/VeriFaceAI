import os
import shutil
from tqdm import tqdm

AUDIO_DATASET_PATH = "D:/deepfake-detection/Datasets/Deepfake-Audio/files"  # Updated path
OUTPUT_DIR = "D:/deepfake-detection/Audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def copy_audio_files(src_folder, dest_folder):
    copied = 0
    for subfolder in tqdm(os.listdir(src_folder), desc="Processing subfolders"):
        subfolder_path = os.path.join(src_folder, subfolder)

        if os.path.isdir(subfolder_path):  # Check if it's a subfolder
            wav_files = [f for f in os.listdir(subfolder_path) if f.endswith(".wav")]

            if len(wav_files) != 4:  # Ensure each folder has 4 files
                print(f"⚠️ Warning: {subfolder} has {len(wav_files)} `.wav` files instead of 4!")

            for file_name in wav_files:
                src_file = os.path.join(subfolder_path, file_name)
                dest_file = os.path.join(dest_folder, f"{subfolder}_{file_name}")

                shutil.copy2(src_file, dest_file)
                copied += 1

    if copied == 0:
        print("❌ No `.wav` files found! Check dataset structure.")
    else:
        print(f"✅ Copied {copied} audio files to {OUTPUT_DIR}")

copy_audio_files(AUDIO_DATASET_PATH, OUTPUT_DIR)
