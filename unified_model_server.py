from fastapi import FastAPI, UploadFile, File, Form
import io
from pydantic import BaseModel
import torch
import time
from PIL import Image
from torchvision import models, transforms
from torchvision.models import get_model_weights

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 1️⃣ 模型加载区
# ===============================

print("🔄 Loading models...")

model_registry = {}

model_registry["mobilenet_v2"] = models.mobilenet_v2(weights="IMAGENET1K_V1")
model_registry["efficientnet_b0"] = models.efficientnet_b0(weights="IMAGENET1K_V1")
model_registry["resnet18"] = models.resnet18(weights="IMAGENET1K_V1")
model_registry["resnet50"] = models.resnet50(weights="IMAGENET1K_V1")
model_registry["vit"] = models.vit_b_16(weights="IMAGENET1K_V1")

for name, model in model_registry.items():
    model.eval()
    model.to(device)

print("✅ All models loaded.")

# ===============================
# 2️⃣ ImageNet Labels
# ===============================

weights = get_model_weights("resnet50").DEFAULT
labels = weights.meta["categories"]

# ===============================
# 3️⃣ 图像预处理
# ===============================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# 4️⃣ 请求结构
# ===============================

class RequestData(BaseModel):
    model_name: str
    image_path: str

# ===============================
# 5️⃣ 推理接口
# ===============================

@app.post("/classify")
async def classify(
    model_name: str = Form(...),
    image: UploadFile = File(...)
):

    if model_name not in model_registry:
        return {
            "business_data": {
                "status": "error",
                "message": f"Model {model_name} not found"
            }
        }

    model = model_registry[model_name]

    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = transform(img).unsqueeze(0).to(device)

        start = time.time()

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()

        end = time.time()

        return {
            "business_data": {
                "status": "success",
                "model_used": model_name,
                "prediction_id": pred_id,
                "english_label": labels[pred_id],
                "confidence": round(confidence, 4),
                "info": "Please translate the label into Chinese."
            },
            "server_telemetry": {
                "inference_latency_ms": round((end - start) * 1000, 2)
            }
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())

        return {
            "business_data": {
                "status": "error",
                "message": str(e)
            }
        }