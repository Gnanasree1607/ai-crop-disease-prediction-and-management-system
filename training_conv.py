from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# --- Data augmentation ---
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_gen = train_datagen.flow_from_directory('C:/Users/HP/Desktop/plantdiseasesystem/dataset/train', target_size=(224,224), batch_size=5, class_mode='categorical')
val_gen = val_datagen.flow_from_directory('C:/Users/HP/Desktop/plantdiseasesystem/dataset/val', target_size=(224,224), batch_size=3, class_mode='categorical')

# --- Load pretrained MobileNetV2 ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False  # freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train (overfit tiny dataset intentionally) ---
# --- Train ---
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# --- Save model ---
model.save("tiny_dataset_model.keras")

# --- Save class labels ---
labels = list(train_gen.class_indices.keys())
with open("class_labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

print("✅ Model and class labels saved successfully!")

