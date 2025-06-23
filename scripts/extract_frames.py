import cv2
import os
from tqdm import tqdm

VIDEO_DIR = "C:/Users/chakl/OneDrive/Desktop/deepfake-detection/Datasets/Celeb-DF-v2"
OUTPUT_DIR = "C:/Users/chakl/OneDrive/Desktop/deepfake-detection/Frames"

folders = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]

def extract_frames(video_path, output_folder, frame_interval=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.basename(video_path).split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"✅ Extracted frames from {video_name}")

for folder in folders:
    folder_path = os.path.join(VIDEO_DIR, folder)
    output_folder = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(output_folder, exist_ok=True)

    for video_file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        if video_file.endswith(".mp4"):
            extract_frames(os.path.join(folder_path, video_file), output_folder)

print("🎉 Frame extraction complete!")
