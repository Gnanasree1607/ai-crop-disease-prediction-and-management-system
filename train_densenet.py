from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import os

# --- Paths ---
train_dir = r"C:\Users\HP\Desktop\plantdiseasesystem\dataset\train"
val_dir = r"C:\Users\HP\Desktop\plantdiseasesystem\dataset\val"

# --- Data Augmentation (strong for small dataset) ---
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.densenet.preprocess_input,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.densenet.preprocess_input
)

# --- Load Data ---
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=5,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=3,
    class_mode='categorical'
)

# --- Load Pretrained DenseNet121 ---
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze lower layers, fine-tune upper layers
for layer in base_model.layers[:-60]:  # unfreeze last 60 layers
    layer.trainable = False
for layer in base_model.layers[-60:]:
    layer.trainable = True

# --- Build Custom Model ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# --- Compile ---
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(
    "best_densenet_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# --- Train ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[lr_reduction, checkpoint],
    verbose=1
)

# --- Save Final Model ---
model.save("densenet_ragi_model.keras")

# --- Save Class Labels for Flask App ---
labels = list(train_gen.class_indices.keys())
with open("class_labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

print("\n✅ Model and class labels saved successfully!")
print("🔥 Best DenseNet model likely to exceed 80% accuracy with fine-tuning and augmentation.")


