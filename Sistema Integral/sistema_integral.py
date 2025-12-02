import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from gtts import gTTS
import os

# --- Configuración de Modelos ---
# 1. HTR (Handwritten Text Recognition)
MODEL_HTR_ID = "microsoft/trocr-large-handwritten"

# --- 1. Inicialización de Componentes del Pipeline ---

print("⏳ Inicializando modelos...")

# Inicializar HTR
try:
    processor_htr = TrOCRProcessor.from_pretrained(MODEL_HTR_ID)
    model_htr = VisionEncoderDecoderModel.from_pretrained(MODEL_HTR_ID)
    print(f" HTR ({MODEL_HTR_ID}) cargado.")
except Exception as e:
    print(f" Error al cargar HTR: {e}")
    model_htr = None



# --- 2. Funciones del Pipeline ---

def detectar_y_segmentar(image_path: str):
    """
    *SIMULACIÓN* de la Detección de Texto.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        print("-> Imagen cargada. (Simulando detección)")
        return [image]
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado en {image_path}")
        return []


def reconocimiento_htr(list_of_image_crops: list) -> str:
    """
    3. Reconocimiento de Texto Manuscrito (HTR)
    """
    if not model_htr or not list_of_image_crops:
        return ""

    full_text = []
    print("-> Iniciando Reconocimiento HTR...")

    for image in list_of_image_crops:
        pixel_values = processor_htr(image, return_tensors="pt").pixel_values

        # Generar la transcripción
        generated_ids = model_htr.generate(pixel_values)

        transcribed_text = processor_htr.decode(generated_ids[0], skip_special_tokens=True)
        full_text.append(transcribed_text)

    # El texto crudo del HTR es ahora el texto final
    return " ".join(full_text)



def sintesis_de_voz_tts(final_text: str, output_filename="audio_manuscrito_final.mp3"):
    """
    4. Síntesis de Voz (TTS) con gTTS.
    """
    if not final_text.strip():
        print(" Texto final vacío. No se puede generar audio.")
        return None

    print(f"-> Generando audio gTTS para: '{final_text}'")

    try:
        # Crear el objeto TTS
        tts = gTTS(text=final_text, lang='es')

        # Guardar el audio como archivo MP3
        tts.save(output_filename)

        print(f" Audio guardado exitosamente en: {output_filename}")
        return os.path.abspath(output_filename)

    except Exception as e:
        print(f" ERROR al generar audio con gTTS: {e}")
        return None


# --- 3. Función Principal de Ejecución ---

def ejecutar_pipeline_htts(image_path: str):
    """
    Flujo de ejecución simplificado: HTR -> TTS.
    """
    print("\n--- INICIO DEL PIPELINE DE HTTS ---")

    # 1. Entrada + Detección
    image_crops = detectar_y_segmentar(image_path)
    if not image_crops:
        print(" Finalizado. No se pudieron procesar las imágenes.")
        return

    # 2. Reconocimiento (HTR)
    # El resultado de HTR es directamente el texto final
    texto_final = reconocimiento_htr(image_crops)
    print(f"\n[Texto a Leer (HTR)]: {texto_final}")

    if not texto_final.strip():
        print(" HTR no detectó texto. Finalizando.")
        return

    # 3. Síntesis y Salida (TTS)
    audio_path = sintesis_de_voz_tts(texto_final, "audio.mp3")

    print("\n--- FIN DEL PIPELINE ---")
    if audio_path:
        print(f"Ruta de salida del audio: {audio_path}")


# --- 4. Ejemplo de Uso ---

if __name__ == "__main__":

    EJEMPLO_IMAGEN_RUTA = "manuscrito_ejemplo.jpg"

    if os.path.exists(EJEMPLO_IMAGEN_RUTA):
        ejecutar_pipeline_htts(EJEMPLO_IMAGEN_RUTA)
    else:
        print(
            f"\n ERROR: Por favor, crea una imagen de ejemplo llamada '{EJEMPLO_IMAGEN_RUTA}' para ejecutar el script.")