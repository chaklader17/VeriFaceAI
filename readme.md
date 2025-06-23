<div align="center">

# DeepDetect AI: Deepfake Detection using CNN and RNN

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**DeepDetect AI** is a machine learning-based deepfake detection system developed for the **CSE299 Junior Design** course at **North South University**. The project leverages **Convolutional Neural Networks (CNN)** for image-based detection and **Recurrent Neural Networks (RNN)** for audio-based detection of deepfake media.

</div>

---

<div align="center">

## 📌 Project Overview

</div>

The goal of this project is to build a **robust deepfake detection system** that analyzes both visual and audio cues from videos to determine their authenticity. The system processes video frames using a CNN (ResNet50) model and audio spectrograms using an LSTM-based RNN model to classify content as real or fake.

This is accomplished by:
- **Extracting frames and audio** from video inputs
- **Preprocessing video and audio data**
- **Training and evaluating CNN and RNN models**
- **Aggregating predictions across frames/audio for final decision**

---

<div align="center">

## 🔍 Key Features

</div>

- **Multimodal Detection**: Combines both video (frame-level) and audio (spectrogram-based) classification.
- **Frame Extraction & Analysis**: Uses ResNet50 CNN to process and classify video frames.
- **Spectrogram-Based Audio Detection**: Uses an LSTM model trained on Mel-spectrograms of audio.
- **Prediction Logging**: Detailed frame-wise and overall prediction scores saved to CSV.
- **FFmpeg Integration**: Uses FFmpeg for efficient audio extraction from videos.

---

<div align="center">

## 🧠 Technologies Used

</div>

### Machine Learning & Libraries
- **TensorFlow/Keras** for CNN model (ResNet50)
- **PyTorch** for RNN model (LSTM)
- **OpenCV**, **FFmpeg**, **Librosa** for video/audio preprocessing
- **NumPy**, **Matplotlib**, **Pandas**, **CSV** for data handling

### Development Tools
- Python 3.10+
- PyCharm IDE
- Git & GitHub for version control
- FFmpeg CLI tool for multimedia processing

---

<div align="center">

## 📚 Datasets Used

</div>

- **[Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)**: High-quality deepfake video dataset used for training and evaluation of video-based model.
- **Custom Deepfake Audio Dataset**: Directory-based structure of `.wav` audio samples used for spectrogram generation and training of the RNN model.

---

<div align="center">

## 🧪 Model Files

</div>

| Model                | Path                                        |
|---------------------|---------------------------------------------|
| CNN Model (ResNet50) | `Models/cnn_deepfake_model.h5`              |
| RNN Model (LSTM)     | `Models/rnn_audio_model.pth`                |

---

<div align="center">

## 🧠 Evaluation Results (Example)

</div>

| Video              | Video Prediction | Audio Prediction | Final Verdict |
|--------------------|------------------|------------------|----------------|
| test-real.mp4      | Real (0.74)      | Real (0.81)      | ✅ Real         |
| test-fake.mp4      | Fake (0.86)      | Fake (0.78)      | ❌ Fake         |

---
