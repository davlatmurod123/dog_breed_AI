import io
import json

import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# 1. Загружаем названия пород и модель один раз при запуске сервера
with open("models/classes.json", encoding="utf-8") as f:
    CLASSES = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASSES))
model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
model.to(device).eval()  # eval отключает режим обучения

# 2. Та же обработка фото, что была у val/test в Colab
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="Dog Breed API")

# Разрешаем запросы из HTML-страницы (пока открыта для всех, для разработки)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "breeds": len(CLASSES), "device": device}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Проверяем, что прислали картинку
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Нужен файл-изображение")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    x = preprocess(img).unsqueeze(0).to(device)   # добавляем размерность батча
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]  # логиты -> вероятности
    top = probs.topk(3)

    return {
        "predictions": [
            {"breed": CLASSES[i], "confidence": round(p, 4)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]
    }