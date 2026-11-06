# 🐶 Dog Breed API

Мощный API для распознавания пород собак по изображению с помощью нейросети EfficientNet.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern-009688)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)

## ✨ Что умеет проект

- распознаёт породу собаки по фотографии;
- возвращает топ-3 наиболее вероятных варианта;
- работает через REST API;
- поддерживает CORS для простого интегрирования с веб-страницей;
- загружает модель и список классов один раз при старте сервера.

## 🧠 Технологии

- Python 3.10+
- FastAPI
- PyTorch
- TorchVision
- timm
- Pillow

## 📁 Структура проекта

```text
 dog-breed-app/
├── main.py
├── requirements.txt
├── README.md
├── models/
│   ├── best_model.pt
│   ├── classes.json
│   └── history.json
└── .vscode/
    └── settings.json
```

## 🚀 Установка

1. Клонируйте проект:

```bash
git clone <ссылка-на-репозиторий>
cd dog-breed-app
```

2. Создайте виртуальное окружение (по желанию):

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Установите зависимости:

```bash
python -m pip install -r requirements.txt
```

## ▶️ Запуск

```bash
python main.py
```

После запуска приложение будет доступно по адресу:

- http://127.0.0.1:8000

Swagger UI:

- http://127.0.0.1:8000/docs

## 📡 API

### GET /health

Проверка состояния сервиса.

Пример ответа:

```json
{
  "status": "ok",
  "breeds": 120,
  "device": "cpu"
}
```

### POST /predict

Отправьте изображение собаки в поле `file`.

Пример запроса через curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@dog.jpg"
```

Пример ответа:

```json
{
  "predictions": [
    {"breed": "Labrador Retriever", "confidence": 0.8734},
    {"breed": "Golden Retriever", "confidence": 0.0912},
    {"breed": "Beagle", "confidence": 0.0201}
  ]
}
```

## ⚠️ Важные замечания

- В папке `models/` должны лежать:
  - `best_model.pt` — обученная модель;
  - `classes.json` — список пород собак;
- Если фото не загружено или файл не является изображением, API вернёт ошибку `400`.

## 🏆 Цель проекта

Проект создан как компактный и удобный пример использования нейросети для классификации пород собак в реальном веб-сервисе на FastAPI.

---

Если хочешь, я могу в следующем сообщении сделать ещё более крутой README с:
- красивым GitHub-баннером;
- секцией "Как это работает";
- примерами фронтенда для загрузки фото;
- разделом "Примеры результатов".
