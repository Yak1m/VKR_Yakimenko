import os
import warnings
import logging

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")

logging.getLogger('absl').setLevel(logging.ERROR)

from deepface import DeepFace

def calculate_identity_preservation(original_img_path, generated_img_path):
    """
    Вычисляет косинусное сходство между лицами на оригинальной и сгенерированной фотографии.
    Возвращает процент сходства.
    """
    try:
        result = DeepFace.verify(
            img1_path=original_img_path, 
            img2_path=generated_img_path, 
            model_name="Facenet",
            detector_backend="ssd", 
            distance_metric="cosine",
            enforce_detection=False 
        )
        
        similarity_score = (1 - result["distance"]) * 100
        
        return round(similarity_score, 2), result["distance"]
    
    except ValueError as e:
        print(f"Ошибка детекции лица: {e}")
        return None, None
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")
        return None, None

if __name__ == "__main__":
    
    original = r"input_img\2148739334.jpg"
    generated = r"out_val\10_corporate_card_bg_only.png" 
    
    if os.path.exists(original) and os.path.exists(generated):
        print("\n🔍 Вычисление метрики Identity Preservation (FaceNet)...")
        sim_score, distance = calculate_identity_preservation(original, generated)
        
        if sim_score:
            print("-" * 50)
            print(f"Оригинал:  {os.path.basename(original)}")
            print(f"Генерация: {os.path.basename(generated)}")
            print(f"✅ Косинусное расстояние: {round(distance, 4)}")
            print(f"✅ Сохранение идентичности: {sim_score}%")
            if sim_score > 60: 
                print("Идентичность успешно сохранена (порог > 60%).")
            else:
                print("Идентичность нарушена.")
            print("-" * 50)
    else:
        print("Файлы не найдены. Проверьте пути.")