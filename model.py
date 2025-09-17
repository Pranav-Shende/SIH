import os
import zipfile
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# -------------------------
# 1. Unzip the dataset
# -------------------------
zip_file_path = r'Dataset.zip'
extract_path = 'dataset'

if not os.path.exists(extract_path):
    print(f"Extracting {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
else:
    print("Dataset directory already exists. Skipping extraction.")

# ----------------------------------------------------
# 2. Set parameters
# ----------------------------------------------------
data_dir = 'dataset'
print(f"Using data directory: {data_dir}")

image_size = (224, 224)
batch_size = 5
epochs = 30  # Increased for better accuracy

num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"Found {num_classes} classes.")

# --------------------------------------
# 3. Load dataset
# --------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    color_mode = 'rgb'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    color_mode = 'rgb'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------------
# 4. Data augmentation
# -------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# -------------------------
# 5. Build the model
# -------------------------
# FIX: Using MobileNetV2 as a new base model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = data_augmentation(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.summary()

# -------------------------
# 6. Compile & Train
# -------------------------
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# -------------------------
# 7. Save model
# -------------------------
model.save("cattle_buffalo_model.h5")

# Export to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("cattle_buffalo_model.tflite", "wb") as f:
    f.write(tflite_model)

print("\n✅ Training complete. Model saved as cattle_buffalo_model.h5 and .tflite")