import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import os

# ===== DEVICE =====
device = "cuda" if torch.cuda.is_available() else "cpu"


# ===== PATH MODEL (AUTO FIX PATH) =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "vlm", "model")


# ===== LOAD MODEL =====
processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        processor = DonutProcessor.from_pretrained(MODEL_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()


# ===== PREDICT RAW =====
def predict(image):
    """
    image: PIL.Image hoặc path string
    return: raw string output
    """

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    outputs = model.generate(
        pixel_values,
        max_length=512,
        num_beams=3,
        early_stopping=True
    )

    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return result


# ===== PARSE TAG =====
def extract_tag(text, tag):
    match = re.search(f"<{tag}>(.*?)</{tag}>", text)
    return match.group(1).strip() if match else None


# ===== FALLBACK FIX (cho demo đẹp hơn) =====
def clean_result(parsed):
    # xử lý nếu model miss field

    if not parsed["store"]:
        parsed["store"] = "Unknown"

    if not parsed["date"]:
        parsed["date"] = "Unknown"

    if not parsed["total"]:
        parsed["total"] = "⚠️ Missing"

    return parsed


# ===== FULL PIPELINE =====
def extract_invoice(image):
    raw = predict(image)

    parsed = {
        "store": extract_tag(raw, "store"),
        "date": extract_tag(raw, "date"),
        "total": extract_tag(raw, "total"),
    }

    parsed = clean_result(parsed)

    return {
        "raw": raw,
        "parsed": parsed
    }


# ===== QUICK TEST =====
if __name__ == "__main__":
    load_model()
    test_image = os.path.join(BASE_DIR, "data", "test", "images", "invoice171.jpg")

    result = extract_invoice(test_image)

    print("RAW:", result["raw"])
    print("PARSED:", result["parsed"])