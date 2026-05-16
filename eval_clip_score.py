import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


def calculate_clip_score(image_path, text_prompt, model, processor):
    """
    Вычисляет косинусное сходство между изображением и текстом промпта 
    с помощью модели OpenAI CLIP.
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[text_prompt], 
        images=image, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():
        outputs = model(**inputs)
    
    clip_score = outputs.logits_per_image.item()
    return round(clip_score, 2)

if __name__ == "__main__":
    TARGET_DIR = r"out_val" 
    
    print("⏳ Загрузка модели CLIP (OpenAI)...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    print("✅ Модель загружена!\n")
    
    print(f"🔍 Сканирование папки: {TARGET_DIR}")
    print("-" * 60)
    
    total_score = 0
    valid_pairs_count = 0
    
    for filename in os.listdir(TARGET_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(TARGET_DIR, filename)
            
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(TARGET_DIR, f"{base_name}.txt")
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as file:
                    prompt_text = file.read().strip()
                
                if prompt_text:
                    score = calculate_clip_score(image_path, prompt_text, model, processor)
                    
                    total_score += score
                    valid_pairs_count += 1
                    
                    short_prompt = prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
                    
                    print(f"Картинка: {filename}")
                    print(f"Промпт:   {short_prompt}")
                    print(f"Оценка (CLIP Score): {score}")
                    print("-" * 60)
                else:
                    print(f"Текстовый файл {base_name}.txt пуст. Пропуск.\n" + "-" * 60)
            else:
                print(f"Для картинки {filename} не найден файл {base_name}.txt. Пропуск.\n" + "-" * 60)

    if valid_pairs_count > 0:
        average_score = round(total_score / valid_pairs_count, 2)
        print("\nИТОГОВАЯ СТАТИСТИКА ЭКСПЕРИМЕНТА:")
        print(f"Обработано пар (Картинка+Промпт): {valid_pairs_count}")
        print(f"Средний показатель семантического соответствия (Avg CLIP Score): {average_score}")
    else:
        print("\nНе найдено ни одной пары Картинка + Текстовый файл.")