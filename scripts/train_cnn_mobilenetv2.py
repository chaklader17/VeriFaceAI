import os
import numpy as np
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

# === Paths ===
DATA_DIR = r"D:\deepfake-detection\frames"
MODEL_SAVE_PATH = r"D:\deepfake-detection\Models\cnn_mobilenetv2_model.h5"
LOG_DIR = r"D:\deepfake-detection\Logs\mobilenetv2_" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("[INFO] Class indices:", train_data.class_indices)


labels = train_data.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))
print("[INFO] Computed class weights:", class_weights)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Binary output

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])


callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    TensorBoard(log_dir=LOG_DIR)
]

print("\nStarting training with MobileNetV2...\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

print(f"\n Training complete. Best model saved to: {MODEL_SAVE_PATH}")
print(f" TensorBoard logs saved to: {LOG_DIR}")
