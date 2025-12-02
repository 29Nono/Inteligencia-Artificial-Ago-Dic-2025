import cv2
import numpy as np
import tensorflow as tf
import os


# --- CLASE PARA ESTABILIZAR PREDICCIONES ---
class PredictionSmoother:
    """Aplica un promedio móvil para suavizar las predicciones de la CNN."""

    def __init__(self, history_size=5):
        # Almacena las predicciones (probabilidades) de las últimas 5 frames
        self.history = []
        self.history_size = history_size

    def add_prediction(self, prediction_vector):
        # Solo almacenar el vector de probabilidades, no el resultado de predict() completo
        self.history.append(prediction_vector)
        # Mantener solo los últimos 'history_size' elementos
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def get_smoothed_prediction(self):
        if len(self.history) < 1:
            return None
        # Calcula el promedio de todas las predicciones almacenadas en el historial
        smoothed_vector = np.mean(self.history, axis=0)
        return smoothed_vector


# --- 1. CONFIGURACIÓN DE CARGA ---
MODEL_FILE = 'facial_recognition_5_class.h5'
IMG_SIZE = 224

# Etiquetas de Clases (ACTUALIZA ESTO con la salida exacta del script de entrenamiento)
class_labels = {
    0: 'Amigo_Brayan',
    1: 'Amigo_Edson',
    2: 'Amigo_Plata',
    3: 'Familia_Patricia',
    4: 'Hugh_Jackman',
    5: 'Jennifer_Lawrence',
    6: 'Leonardo_DiCaprio',
    7: 'Megan_Fox',
    8: 'Yo_Arnoldo',
    9: 'Yo_Sebastian'

    # ASEGÚRATE DE QUE ESTE DICCIONARIO COINCIDA CON TU DATASET
}

# Cargar el modelo de Keras
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Cargar el Detector Facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar el suavizador
smoother = PredictionSmoother(history_size=7)  # Usar las últimas 7 frames para el promedio

# --- 2. APLICACIÓN EN TIEMPO REAL (Bucle de Video) ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección facial
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]

        # Preprocesamiento
        face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_array = np.array(face_resized, dtype="float32") / 255.0
        face_input = np.expand_dims(face_array, axis=0)

        # Clasificación con la CNN (se predice cada frame)
        predictions = model.predict(face_input, verbose=0)[0]  # Obtener solo el vector de probabilidades

        # Aplicar Suavizado y obtener el resultado estable
        smoother.add_prediction(predictions)
        smoothed_predictions = smoother.get_smoothed_prediction()

        if smoothed_predictions is not None:
            # Usar la predicción suavizada
            predicted_class_index = np.argmax(smoothed_predictions)
            confidence = smoothed_predictions[predicted_class_index] * 100
            predicted_label = class_labels[predicted_class_index]
        else:
            # Si el historial no está lleno, usar el frame actual
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[predicted_class_index] * 100
            predicted_label = class_labels[predicted_class_index]

        # --- Mostrar resultados en tiempo real ---
        label = f"{predicted_label}: {confidence:.1f}%"

        # Color: Verde si cumple el umbral del 80% (mínimo del proyecto)
        color = (0, 255, 0) if confidence > 80.0 else (0, 0, 255)

        # Dibujar Bounding Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Mostrar Etiqueta
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mostrar el frame procesado
    cv2.imshow('CNN Facial Recognition - Tiempo Real', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()