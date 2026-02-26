from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 2)
model.load_state_dict(torch.load("coloai_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ['cancer', 'normal']

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)

    pred = probs.argmax(dim=1).item()
    confidence = probs[0][pred].item()

    result = f"{classes[pred]} ({confidence*100:.2f}%)"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )