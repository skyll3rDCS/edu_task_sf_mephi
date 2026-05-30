"""
Веб-интерфейс для модели предсказания конверсии
Сервис СберАвтоподписка
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify

# Инициализация приложения
app = Flask(__name__)

# Конфигурация путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.pkl')

# Глобальные переменные для компонентов модели
model = None
scaler = None
feature_columns = None
model_loaded = False

def load_model_components():
    """Загрузка всех компонентов модели"""
    global model, scaler, feature_columns, model_loaded
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        model_loaded = True
        print("Модель и вспомогательные компоненты успешно загружены")
        print(f"Загружено признаков: {len(feature_columns)}")
    except FileNotFoundError as e:
        print(f"Ошибка загрузки компонентов модели: {e}")
        model_loaded = False
    except Exception as e:
        print(f"Непредвиденная ошибка при загрузке модели: {e}")
        model_loaded = False

def prepare_features(data):
    """Подготовка признаков для модели"""
    feature_dict = {}
    
    # Базовые числовые признаки
    feature_dict['visit_number'] = data.get('visit_number', 1)
    feature_dict['visit_hour'] = data.get('visit_hour', 12)
    feature_dict['visit_dayofweek'] = data.get('visit_dayofweek', 3)
    feature_dict['visit_month'] = data.get('visit_month', 6)
    feature_dict['total_hits'] = data.get('total_hits', 10)
    feature_dict['unique_pages'] = data.get('unique_pages', 5)
    feature_dict['car_page_views'] = data.get('car_page_views', 0)
    feature_dict['form_interactions'] = data.get('form_interactions', 0)
    feature_dict['session_duration_sec'] = data.get('session_duration_sec', 60)
    feature_dict['screen_width'] = data.get('screen_width', 1920)
    feature_dict['screen_height'] = data.get('screen_height', 1080)
    
    # Бинарные признаки
    feature_dict['is_organic'] = 1 if data.get('is_organic', False) else 0
    feature_dict['is_social'] = 1 if data.get('is_social', False) else 0
    feature_dict['is_mobile'] = 1 if data.get('device_category', 'desktop') == 'mobile' else 0
    feature_dict['is_tablet'] = 1 if data.get('device_category', 'desktop') == 'tablet' else 0
    feature_dict['is_desktop'] = 1 if data.get('device_category', 'desktop') == 'desktop' else 0
    feature_dict['is_weekend'] = 1 if data.get('visit_dayofweek', 3) >= 5 else 0
    
    # Создаем DataFrame с правильным порядком колонок
    df = pd.DataFrame([feature_dict])
    
    # Приводим к порядку признаков, ожидаемому моделью
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    
    return df

@app.route('/', methods=['GET'])
def index():
    """Главная страница с формой ввода данных"""
    default_payload = json.dumps({
        "visit_number": 2,
        "visit_month": 6,
        "visit_dayofweek": 4,
        "visit_hour": 14,
        "total_hits": 25,
        "unique_pages": 8,
        "car_page_views": 3,
        "form_interactions": 2,
        "session_duration_sec": 180,
        "screen_width": 1920,
        "screen_height": 1080,
        "is_organic": False,
        "is_social": False,
        "device_category": "desktop"
    }, indent=2, ensure_ascii=False)
    
    return render_template(
        'index.html',
        default_payload=default_payload,
        model_status=model_loaded
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Эндпоинт для получения предсказаний"""
    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Модель не загружена. Обратитесь к администратору."
        }), 503
    
    try:
        # Парсинг входных данных
        if request.is_json:
            data = request.get_json()
        elif request.form.get('json_data'):
            data = json.loads(request.form.get('json_data'))
        else:
            return jsonify({
                "status": "error",
                "message": "Неверный формат запроса. Ожидается JSON."
            }), 400
        
        # Подготовка признаков
        features_df = prepare_features(data)
        
        # Масштабирование числовых признаков
        if scaler is not None:
            numerical_cols = features_df.select_dtypes(include=[np.number]).columns
            features_df[numerical_cols] = scaler.transform(features_df[numerical_cols])
        
        # Получение предсказания
        probability = model.predict_proba(features_df)[0, 1]
        predicted_class = int(probability >= 0.5)
        
        if probability >= 0.7:
            conversion_level = "high"
            level_description = "Высокая вероятность конверсии"
        elif probability >= 0.3:
            conversion_level = "medium"
            level_description = "Средняя вероятность конверсии"
        else:
            conversion_level = "low"
            level_description = "Низкая вероятность конверсии"

        return jsonify({
            "status": "success",
            "probability": round(probability, 4),
            "predicted_class": predicted_class,
            "conversion_level": conversion_level,
            "level_description": level_description
        })
        
    except json.JSONDecodeError:
        return jsonify({
            "status": "error",
            "message": "Некорректный JSON формат данных."
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Ошибка при выполнении предсказания: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Эндпоинт для проверки состояния сервиса"""
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "service": "conversion_predictor",
        "version": "1.0.0"
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Возвращает список признаков, используемых моделью"""
    if not model_loaded:
        return jsonify({"error": "Модель не загружена"}), 503
    
    return jsonify({
        "status": "success",
        "feature_count": len(feature_columns) if feature_columns else 0,
        "features": feature_columns
    })

# Загрузка модели при старте приложения
load_model_components()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)