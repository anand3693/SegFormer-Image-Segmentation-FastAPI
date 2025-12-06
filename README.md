# SegFormer Image Segmentation Web App 🚀

A lightweight web application built using **FastAPI**, **Hugging Face Transformers**, and **SegFormer** model (`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`) to perform **semantic image segmentation** directly in the browser.

---

## ✨ Features

✔ Upload an image & visualize segmentation results  
✔ Display:
- Original Image
- Segmentation Mask
- Overlay Output (Mask + Original)
✔ Automatically generated class-color legend  
✔ Fast inference through Hugging Face pipelines  
✔ Web UI using Jinja2 Templates  

---

## 🖥️ Demo Preview

| Original | Mask | Overlay |
|---------|------|---------|
| 🖼️ | 🎭 | 🧩 |



<img width="1885" height="913" alt="image seg eg" src="https://github.com/user-attachments/assets/b6f41b24-be8a-47da-ab9f-1f2cf597b527" />

---

## 🧠 Model Used
- **SegFormer B5**
  - Pretrained on CityScapes dataset
  - Hugging Face Model: `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`

---

## 📁 Project Structure

📦SegFormer-Image-Segmentation-FastAPI
┣ 📂templates
┃ ┗ 📜index.html
┣ 📜main.py
┣ 📜requirment.txt
┗ 📜README.md


## Install Dependencies
pip install -r requirment.txt

## Run FastAPI App
uvicorn main:app --reload


