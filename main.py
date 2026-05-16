import io
import os
import cv2
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import streamlit as st

# --- КОНФИГУРАЦИЯ ---
PROJECT_ID = "glassy-bonsai-479912-q1"  # Мой ID проекта Google Cloud
LOCATION = "us-central1"        
MODEL_NAME = "gemini-2.5-flash-image"
TEXT_MODEL_NAME = "gemini-2.5-flash"
API_KEY = "AQ.Ab8RN6K9T3FehzqElm8_a2NaZgj7IlmZ2738brnZw-tpFdPV8g"
BRAND_COLORS = {
    "gold": "#D2923A",
    "text_main": "#333333"
}

test = False

user_name = "Иван Иванов"
body_text = "Ведущий эксперт по кибербезопасности с 10-летним опытом. Специализируется на защите критической инфраструктуры и предотвращении кибератак. Автор множества публикаций и докладов на международных конференциях."

# Инициализируем хранилище
if 'history' not in st.session_state:
    st.session_state.history = []

def get_current_state():
    """Возвращает текущее изображение и маску из конца истории"""
    if st.session_state.history:
        return st.session_state.history[-1]
    return None


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def pil_to_bytes(pil_image):
    """Конвертирует PIL Image в байты (PNG формат)"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.read()

def translate_to_english(text, client):
    """Переводит пользовательский промпт на английский язык"""
    if not text.strip():
        return ""
    prompt = f"Translate the following text to English. Return ONLY the translation, without any quotes or additional explanations: '{text}'"
    response = client.models.generate_content(
        model=TEXT_MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()

def analyze_photo_context(client, image_part):
    """
    Отправляет исходное фото в Gemini для получения цветового и стилистического контекста.
    Возвращает словарь с атрибутами.
    """
    # Системный промпт-анализатор
    prompt = """
    You are an expert art director. Analyze this portrait photo. 
    Return a valid JSON object with EXACTLY these keys:
    - "clothing_color": The dominant color of the person's clothes (e.g., "Navy Blue", "White").
    - "style": Determine if the clothing is "formal", "smart-casual", or "casual".
    - "bg_accent_color": Suggest 1-2 background accent colors that are strictly complementary to the clothing_color to make the subject pop.
    - "lighting": Determine if the photo has "warm", "cool", or "neutral" lighting.
    """
    
    contents = [
        types.Content(
            role="user",
            parts=[
                image_part, 
                types.Part.from_text(text=prompt)
            ]
        )
    ]

    # Конфигурация: Указываем, что ждем строго JSON
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2, # Низкая температура для большей логичности и стабильного JSON
        response_mime_type="application/json", # Ключевая настройка для структурированного вывода
    )

    print("🔍 Анализируем гардероб и освещение (Gemini 2.5 Flash)...")
    
    # Отправляем запрос (используем обычный flash для текста и анализа)
    response = client.models.generate_content(
        model=TEXT_MODEL_NAME, 
        contents=contents,
        config=generate_content_config,
    )

    # Парсим результат
    try:
        context = json.loads(response.text)
        print(f"✅ Анализ завершен:\n{json.dumps(context, indent=2, ensure_ascii=False)}")
        return context
    except Exception as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        if hasattr(response, 'text'):
            print(f"Сырой ответ модели: {response.text}")
        return None

def build_dynamic_prompt(base_prompt, context):
    """
    Склеивает базовый промпт с динамическими параметрами из JSON.
    """
    if not context:
        return base_prompt # Фолбэк, если анализ сломался
        
    dynamic_add_on = (
        f"The subject is wearing a {context['clothing_color']} outfit in a {context['style']} style. "
        f"Ensure the generated environment lighting matches the {context['lighting']} tone of the subject. "
        f"Incorporate {context['bg_accent_color']} architectural or decorative accents in the background "
        f"to create color harmony and perfect contrast."
    )
    
    # Соединяем базовое описание офиса с индивидуальными требованиями
    final_prompt = f"{base_prompt}. {dynamic_add_on}"
    return final_prompt

def get_person_segmentation_limits(image_input):
    """
    Использует MediaPipe для семантической сегментации человека.
    Возвращает крайние координаты силуэта и бинарную маску.
    """
    # 1. Создаем настройки сегментатора
    base_options = python.BaseOptions(model_asset_path='selfie_segmenter.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True)

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Загружаем изображение в формате MediaPipe

        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        # 2. Переводим PIL Image в формат, который понимает MediaPipe (NumPy array)
        numpy_image = np.array(pil_img)

        # 3. Создаем объект MediaPipe прямо из массива данных (без файлов!)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        
        # Проводим сегментацию
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask.numpy_view()

        # Обычно это бинарная маска, где человек > 0
        mask = np.squeeze(category_mask) < 0.1

        # --- СОХРАНЕНИЕ МАСКИ ДЛЯ СЛАЙДА ---
        # Конвертируем булеву маску (True/False) в картинку (255/0 - белый/черный)
        mask_8bit = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_8bit).save("presentation_mask.png")
        # -----------------------------------

        y_indices, x_indices = np.where(mask)
        
        # Конвертируем булеву маску (True/False) в 8-битную картинку (0/255)
        mask_8bit = (mask * 255).astype(np.uint8)
        

        if len(x_indices) == 0:
            return None
            
        return {
            'bbox_left': float(np.min(x_indices)),
            'bbox_right': float(np.max(x_indices)), 
            'bbox_top': float(np.min(y_indices)),
            'bbox_bottom': float(np.max(y_indices)),
            'mask': mask 
        }


# --- ГЛАВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ---

def generate_card_gemini(person_input, custom_bg=None, custom_clothes=None, edit_prompt=None, test=False):
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    
    if hasattr(person_input, 'convert'):
        person_img = person_input.convert("RGB")
    else:
        person_img = Image.open(person_input).convert("RGB")
    person_bytes = pil_to_bytes(person_img)
    
    if test:
        print("⚠️ Режим тестирования: Gemini не будет вызван, используется заглушка.")
        # Заглушка: просто копируем входное изображение в выходное
        # visual_context = analyze_photo_context(client, person_bytes)
        if hasattr(person_input, 'convert'):
            person_img = person_input.convert("RGB")
        else:
            person_img = Image.open(person_input).convert("RGB")
        # add_text(img, user_name, body_text, output_path)
        return person_img

    # Подготовка контента 
    image_part = types.Part.from_bytes(
        data=person_bytes,
        mime_type="image/png"
    )

    # Если это просто правка существующего кадра (Больше/Меньше/Кастомная правка)
    if edit_prompt:
        smart_prompt = edit_prompt
    else:
        base_prompt = """
        ACTION: Compose a high-end corporate portrait using the reference person.

        COMPOSITION & LAYOUT:
        1. **Person Position:** Place the person in the **BOTTOM-LEFT** area of the frame. Show them from the waist up or full body.
        If the body is not shown enough in the original picture, generate it. The face should be in the **LOWER LEFT** area of the frame, that is, far from the middle. 
        A person should occupy at least a quarter of the image.
        2. **Background Extension:** Do NOT crop the scene abruptly. Extend the background naturally to fill the frame. There should be no empty flat space on the right (walls, etc.)

        QUALITY:
        Photorealistic, 8k, expensive commercial look. No vignetting.
        """

        # 1. Обработка одежды (ОТКЛЮЧАЕТ АНАЛИЗ, ЕСЛИ ЗАДАНА)
        if custom_clothes:
            clothing_section = f"\nCLOTHING STYLE:\n{custom_clothes}. Ensure the subject is wearing this exactly."
            base_prompt += clothing_section
            visual_context = None # Отключаем автоанализ
        else:
            visual_context = analyze_photo_context(client, image_part)

        # 2. Обработка фона
        if custom_bg:
            bg_section = f"\nBACKGROUND STYLE:\n{custom_bg}. Improve the lighting to highlight the person."
        else:
            bg_section = "\nBACKGROUND STYLE:\nA photorealistic, high-end corporate office background. Modern Scandinavian design, blurred depth of field (bokeh)."
        base_prompt += bg_section

        # 3. Финальная сборка
        if visual_context:
            smart_prompt = build_dynamic_prompt(base_prompt, visual_context)
        else:
            smart_prompt = base_prompt


    print(f"Отправляю промпт: {smart_prompt}")
    
    contents = [
        types.Content(
            role="user",
            parts=[
                image_part, # Сначала картинка
                types.Part.from_text(text=smart_prompt) # Потом инструкция
            ]
        )
    ]

    # Конфигурация
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["IMAGE"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        image_config=types.ImageConfig(
            aspect_ratio="1:1",
            image_size="1K",
            output_mime_type="image/png",
        ),
    )
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=generate_content_config,
    )
    
    # Ищем картинку в частях ответа
    generated_img = None
    if response.parts:
        for part in response.parts:
            if part.inline_data: 
                generated_img = Image.open(io.BytesIO(part.inline_data.data))
                break
    
    if not generated_img and response.candidates:
        for part in response.candidates[0].content.parts:
             if part.inline_data:
                generated_img = Image.open(io.BytesIO(part.inline_data.data))
                break

    return generated_img

# --- ФУНКЦИИ ОТРИСОВКИ ТЕКСТА ---

def get_x_start_for_y(y, mask, padding, margin_right):
    """Вспомогательная функция для поиска X для одной точки Y"""
    h_mask, w_mask = mask.shape
    line_idx = int(y)
    if line_idx >= h_mask: line_idx = h_mask - 1
    
    # Берем узкий срез маски в 5 пикселей для стабильности
    sample = mask[line_idx:line_idx+5, :]
    coords = np.where(sample)
    
    if len(coords[1]) > 0:
        return np.max(coords[1]) + padding
    return margin_right

    

def draw_contour_text(img, text, font, mask, start_y, padding=40, right_margin=50, text_overlay=None):
    """
    Рисует текст 'лесенкой', огибая маску человека.
    """
    if text_overlay is None:
        text_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        
    # Создаем временный слой для прозрачных элементов (RGBA)
    draw_ov = ImageDraw.Draw(text_overlay)
    
    # Слой для самого текста (можно рисовать сразу на img, но лучше тоже на overlay)
    draw_img = ImageDraw.Draw(img)
    
    words = text.split()
    current_word_idx = 0
    line_spacing = 12
    img_w, img_h = img.size
    
    # Высота строки
    line_h = font.getbbox("Hg")[3] - font.getbbox("Hg")[1]
    y = start_y
    
    # Настройки подложки
    bg_color = (255, 255, 255, 160)  # Белый с прозрачностью (0-255)
    text_color = (51, 51, 51, 255)   # Темно-серый
    rect_padding = 10                # Внутренний отступ текста от краев подложки

    while current_word_idx < len(words):
        h_mask, w_mask = mask.shape
        y_end = min(int(y + line_h), h_mask)
        
        # Анализ маски для текущей строки
        line_mask_area = mask[int(y):y_end, :]
        person_pixels = np.where(line_mask_area)
        
        if len(person_pixels[1]) > 0:
            x_start = np.max(person_pixels[1]) + padding
        else:
            x_start = right_margin
            
        available_w = img_w - right_margin - x_start
        
        # Формируем строку
        line_content = ""
        while current_word_idx < len(words):
            test_line = line_content + (" " if line_content else "") + words[current_word_idx]
            test_w = draw_img.textlength(test_line, font=font)
            
            if test_w <= available_w:
                line_content = test_line
                current_word_idx += 1
            else:
                break
        
        if line_content:
            line_w = draw_img.textlength(line_content, font=font)
            
            # --- РИСУЕМ ПОДЛОЖКУ ---
            # Рассчитываем координаты прямоугольника под строкой
            rect_coords = [
                x_start - rect_padding, 
                (y - rect_padding // 2) - 2, 
                x_start + line_w + rect_padding, 
                (y + line_h + rect_padding // 2) + 2
            ]
            # Рисуем скругленный прямоугольник на слое оверлея
            draw_ov.rounded_rectangle(rect_coords, radius=10, fill=bg_color)
            
            # --- РИСУЕМ ТЕКСТ ---
            # Рисуем текст прямо на оверлее (чтобы он был поверх подложки)
            draw_ov.text((x_start, y), line_content, font=font, fill=text_color)
            
        y += line_h + line_spacing
        if y > img_h - line_h: break

    # Совмещаем основное изображение с оверлеем
    # Используем метод alpha_composite (требует, чтобы оба изображения были RGBA)
    img_rgba = img.convert('RGBA')
    combined = Image.alpha_composite(img_rgba, text_overlay)
    
    return combined.convert('RGB'), text_overlay

def add_text(img, user_name, body_text, person_pos, font_size_name=40, font_size_body=20):
    try:
        font_name = ImageFont.truetype("ariblk.ttf", font_size_name)
        font_body = ImageFont.truetype("arial.ttf", font_size_body)
    except:
        font_name = ImageFont.load_default()
        font_body = ImageFont.load_default()

    current_y = 60
    margin_right = 50
    padding_from_person = 40
    img_rgba = img.convert('RGBA')
    overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    rect_padding = 15 
    
    text_only_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(text_only_layer)

    # Считаем координаты для заголовка
    name_x = get_x_start_for_y(current_y, person_pos['mask'], padding_from_person, margin_right)
    name_bbox = draw_ov.textbbox((name_x, current_y), user_name.upper(), font=font_name)
    
    name_bg_color = (255, 255, 255, 180)
    
    # name_rect содержит координаты [x_левый, y_верхний, x_правый, y_нижний]
    name_rect = [
        name_bbox[0] - rect_padding, (name_bbox[1] - rect_padding // 2) - 2, 
        name_bbox[2] + rect_padding, (name_bbox[3] + rect_padding // 2) + 2
    ]
    
    # Рисуем подложку и текст заголовка
    draw_ov.rounded_rectangle(name_rect, radius=12, fill=name_bg_color)
    draw_ov.text((name_x, current_y), user_name.upper(), font=font_name, fill="#D2923A")

    # Сливаем слои
    img = Image.alpha_composite(img_rgba, overlay).convert('RGB')
    
    fixed_gap_between_texts = 30
    current_y = name_rect[3] + fixed_gap_between_texts

    # Рисуем основной текст "лесенкой", начиная с нового динамического current_y
    img, final_text_layer = draw_contour_text(
        img=img, 
        text=body_text, 
        font=font_body, 
        mask=person_pos['mask'], 
        start_y=current_y, 
        padding=padding_from_person, 
        right_margin=margin_right,
        text_overlay=text_only_layer
    )

    final_text_layer.save("presentation_text_layer.png")
    
    return img

# --- ИНТЕРФЕЙС СТРАНИЦЫ ---

st.set_page_config(layout="wide")
st.title("Генератор корпоративных открыток")

col_settings, col_preview = st.columns([1, 2])

with col_settings:
    st.header("Настройки")
    uploaded_file = st.file_uploader("Загрузите фото", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        source_img = Image.open(uploaded_file)
        
        # Блок кастомных настроек перед первой генерацией
        with st.expander("⚙️ Тонкая настройка (Опционально)"):
            st.info("Если поля пустые, алгоритм автоматически подберет фон и определит ваш гардероб.")
            custom_bg_ru = st.text_input("Опишите желаемый фон:", placeholder="Например: Офис в Москва-Сити, панорамные окна, вечер...")
            custom_clothes_ru = st.text_input("Изменить одежду:", placeholder="Например: Строгий черный смокинг с бабочкой...")

        # Основная кнопка генерации (сбрасывает историю)
        if st.button("🚀 Сгенерировать", type="primary"):
            st.session_state.history = [] # Очищаем историю при новой генерации
            client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
            
            with st.spinner("Генерируем..."):
                bg_en = translate_to_english(custom_bg_ru, client) if custom_bg_ru else None
                clothes_en = translate_to_english(custom_clothes_ru, client) if custom_clothes_ru else None
                
                gen_img = generate_card_gemini(source_img, custom_bg=bg_en, custom_clothes=clothes_en, test=test)
                
                if gen_img:
                    mask_dict = get_person_segmentation_limits(gen_img)
                    if mask_dict:
                        # Сохраняем первый шаг в историю
                        st.session_state.history.append({'img': gen_img, 'mask': mask_dict})

        # Если хотя бы одно изображение уже сгенерировано (история не пуста)
        current_state = get_current_state()
        
        if current_state:
            st.divider()
            st.subheader("Редактирование")
            
            # Кнопки отмены
            if len(st.session_state.history) > 1:
                if st.button("↩️ Отменить последнее действие"):
                    st.session_state.history.pop() # Удаляем последний кадр
                    st.rerun() # Перерисовываем интерфейс
            
            # Кастомная правка
            custom_edit_ru = st.text_input("Внести правку:", placeholder="Например: Добавь комнатное растение на задний план")
            if st.button("✏️ Применить правку"):
                if custom_edit_ru:
                    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
                    with st.spinner("Применяем изменения..."):
                        edit_prompt_en = translate_to_english(custom_edit_ru, client)
                        # Для правки отправляем ТЕКУЩУЮ картинку и ТЕКСТ правки
                        gen_img = generate_card_gemini(current_state['img'], edit_prompt=edit_prompt_en, test=test)
                        if gen_img:
                            mask_dict = get_person_segmentation_limits(gen_img)
                            st.session_state.history.append({'img': gen_img, 'mask': mask_dict})
                            st.rerun()

            st.divider()
            # Поля для текста (мгновенный рендер)
            user_name = st.text_input("Имя сотрудника", "Иван Иванов")
            body_text = st.text_area("Описание", "Ведущий эксперт по кибербезопасности...")
            font_size_name = st.slider("Размер шрифта заголовка", 20, 100, 40)
            font_size_body = st.slider("Размер шрифта описания", 10, 50, 20)

with col_preview:
    st.header("Результат")
    current_state = get_current_state()
    
    if current_state:
        # Рисуем текст мгновенно на актуальном фоне
        res = add_text(
            current_state['img'], 
            user_name, 
            body_text, 
            current_state['mask'],
            font_size_name,
            font_size_body
        )
        st.image(res, width=700)
        
        st.divider()
        
        # Размещаем кнопки скачивания в две колонки
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            # Скачивание с текстом
            buf_with_text = io.BytesIO()
            res.save(buf_with_text, format="PNG")
            st.download_button(
                label="💾 Скачать открытку",
                data=buf_with_text.getvalue(),
                file_name="corporate_card_with_text.png",
                mime="image/png",
                type="primary",
                use_container_width=True
            )
            
        with dl_col2:
            # Скачивание БЕЗ текста
            buf_no_text = io.BytesIO()
            current_state['img'].save(buf_no_text, format="PNG")
            st.download_button(
                label="🖼️ Скачать открытку (без текста)",
                data=buf_no_text.getvalue(),
                file_name="corporate_card_bg_only.png",
                mime="image/png",
                type="secondary",
                use_container_width=True
            )
    else:
        st.info("Загрузите фото, настройте параметры (если нужно) и нажмите 'Сгенерировать'")