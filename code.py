import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================
# 1. Dataset Path
# ==========================
dataset_path = r"C:/Users/HP/Desktop/plantdiseasesystem/dataset"  # <-- your local dataset folder

# ==========================
# 2. Data Preparation
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ==========================
# 3. Build Model Function
# ==========================
def build_model(model_name="convnexttiny"):
    if model_name == "resnet50":
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    elif model_name == "convnexttiny":
        from tensorflow.keras.applications import ConvNeXtTiny
        base_model = ConvNeXtTiny(weights="imagenet", include_top=False, input_shape=(224,224,3))
    else:
        raise ValueError("Only ResNet50 and ConvNeXtTiny implemented for local speed demo!")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    preds = Dense(len(train_gen.class_indices), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=preds)

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ==========================
# 4. Train Model
# ==========================
model = build_model("convnexttiny")  # or "resnet50"

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=1
)

# ==========================
# 5. Evaluate
# ==========================
loss, acc = model.evaluate(val_gen)
print(f"Validation Accuracy: {acc*100:.2f}%")

# ==========================
# 6. Save Model Locally
# ==========================
model_path = r"C:\Users\HP\Desktop\ragi_model.h5"
model.save(model_path)
print(f"Model saved at: {model_path}")
