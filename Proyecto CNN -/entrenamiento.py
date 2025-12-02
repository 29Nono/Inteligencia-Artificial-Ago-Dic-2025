import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os
import math

# --- 1. CONFIGURACIÓN DEL PROYECTO ---
DATA_DIR = './dataset'
IMG_SIZE = 224
BATCH_SIZE = 32
# Establece el número de clases en 5
NUM_CLASSES = 10
MODEL_FILE_FINAL = 'facial_recognition_5_class.h5'
INITIAL_LR = 0.0001
EPOCHS = 20

print(f"Clases a entrenar: {NUM_CLASSES}")

# --- 2. PREPARACIÓN Y AUMENTO DE DATOS ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.15
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

# Cálculo correcto de pasos
STEPS_PER_EPOCH = math.ceil(train_generator.samples / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(validation_generator.samples / BATCH_SIZE)


# --- 3. DISEÑO DE ARQUITECTURA (Transfer Learning) ---
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

# Clasificador personalizado
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. ENTRENAMIENTO EXTENDIDO ---
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- COMENZANDO ENTRENAMIENTO ---")
print(f"Objetivo: val_accuracy > 0.80")
history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS
)

# Guardar el modelo final
model.save(MODEL_FILE_FINAL)
print(f"\n✅ Modelo FINAL guardado como: {MODEL_FILE_FINAL}")

# --- 5. ETIQUETAS FINALES ---
class_labels = {v: k for k, v in train_generator.class_indices.items()}
print("\n--- DICCIONARIO DE ETIQUETAS PARA CLASIFICACIÓN EN VIVO ---")
print(f"class_labels = {class_labels}")