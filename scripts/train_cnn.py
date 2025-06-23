import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

FRAME_PATH = "D:/deepfake-detection/Frames"
MODEL_PATH = "D:/deepfake-detection/Models/cnn_deepfake_model.h5"

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    FRAME_PATH,
    target_size=(224, 224),
    batch_size=8,
    class_mode="binary",
    subset="training"
)

model.fit(train_data, epochs=3)

model.save(MODEL_PATH)

print(f"✅ CNN Model Training Complete! Model saved at {MODEL_PATH}")
