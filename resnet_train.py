from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ==========================
# 1. Data Augmentation
# ==========================
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    'C:/Users/HP/Desktop/plantdiseasesystem/dataset/train',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    'C:/Users/HP/Desktop/plantdiseasesystem/dataset/val',
    target_size=(224, 224),
    batch_size=3,
    class_mode='categorical',
    shuffle=False
)

# ==========================
# 2. Load Pretrained ResNet50
# ==========================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze most layers (fine-tune top layers later)
for layer in base_model.layers[:-30]:
    layer.trainable = False

# ==========================
# 3. Add Custom Layers
# ==========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)  # helps prevent overfitting
output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ==========================
# 4. Compile Model
# ==========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# 5. Train Model
# ==========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,          # more epochs for better tuning
    verbose=1
)

# ==========================
# 6. Save Model & Class Labels
# ==========================
model.save("resnet50_ragi_model.keras")

labels = list(train_gen.class_indices.keys())
with open("class_labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

print("✅ Model trained and saved successfully!")
