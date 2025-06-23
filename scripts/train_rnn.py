import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

SPECTROGRAM_PATH = "D:/deepfake-detection/Spectrograms"
MODEL_PATH = "D:/deepfake-detection/Models/rnn_audio_model.pth"

REAL_LABEL = 0
FAKE_LABEL = 1

spectrograms = []
labels = []

for filename in os.listdir(SPECTROGRAM_PATH):
    if filename.endswith(".png"):
        label = REAL_LABEL if "real" in filename.lower() else FAKE_LABEL
        img_path = os.path.join(SPECTROGRAM_PATH, filename)

        img = Image.open(img_path).convert("L")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img)

        spectrograms.append(img_tensor.numpy())
        labels.append(label)

spectrograms = np.array(spectrograms)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.long)

print(f"✅ Debug: X_train shape before reshaping: {X_train.shape}")

X_train = X_train.squeeze(1)
X_train = X_train.reshape(X_train.shape[0], 128, 128)

print(f"✅ Debug: X_train shape after reshaping: {X_train.shape}")


class LSTMDeepfake(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMDeepfake, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print(f"🔹 LSTM input shape: {x.shape}")
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


input_size = 128
hidden_size = 256
num_layers = 2
output_size = 2

model = LSTMDeepfake(input_size, hidden_size, num_layers, output_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

X_train, y_train = X_train.to(device), y_train.to(device)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), MODEL_PATH)

print(f"✅ RNN Model Training Complete! Model saved at {MODEL_PATH}")
