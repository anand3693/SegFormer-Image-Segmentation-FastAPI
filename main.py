from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from PIL import Image
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

app = FastAPI()
templates = Jinja2Templates(directory="templates")

segformer = pipeline(
    "image-segmentation",
    model="nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    width, height = image.size

    results = segformer(image)

    overlay = np.zeros((height, width, 3), dtype=np.float32)
    class_colors = {}

    for r in results:
        label = r["label"]
        mask = np.array(r["mask"], dtype=bool)

        if label not in class_colors:
            class_colors[label] = np.random.rand(3) 

        overlay[mask] = class_colors[label]

    class_colors_hex = {label: "#{:02x}{:02x}{:02x}".format(
        int(c[0]*255), int(c[1]*255), int(c[2]*255))
        for label, c in class_colors.items()
    }

    fig1 = plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(image)
    plt.axis("off")
    img1 = fig_to_base64(fig1)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(overlay)
    plt.axis("off")
    img2 = fig_to_base64(fig2)
    plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.imshow(image)
    ax.imshow(overlay, alpha=0.5)
    ax.axis("off")
    img3 = fig_to_base64(fig3)
    plt.close(fig3)

    return JSONResponse({
        "original": img1,
        "mask": img2,
        "overlay": img3,
        "legend": class_colors_hex 
    })
