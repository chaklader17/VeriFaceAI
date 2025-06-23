import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


DATA_DIR = r"D:\deepfake-detection\frames"
MODEL_SAVE_PATH = r"D:\deepfake-detection\Models\cnn_deepfake_model.h5"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 5

# === Data generator with validation split ===
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
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

# === Check class mapping ===
print("[INFO] Class indices (label mapping):", train_data.class_indices)

# === Compute class weights ===
labels = train_data.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

print("[INFO] Computed class weights:", class_weights)

# === Build CNN model ===
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# === Compile ===
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# === Callbacks ===
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
    EarlyStopping(patience=3, restore_best_weights=True)
]

# === Train ===
print("\n[INFO] Starting training with class weights...\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,   # ✅ this line fixes the imbalance!
    callbacks=callbacks
)

print("\n✅ Training complete. Best model saved at:")
print(f"📁 {MODEL_SAVE_PATH}")
