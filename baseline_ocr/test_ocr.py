import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
from baseline_ocr.ocr_infer import run_ocr
from baseline_ocr.extract_total import extract_total_vn

def test_image(path):
    image = Image.open(path)

    text = run_ocr(image)

    print("\n===== OCR TEXT =====\n")
    print(text)

    total = extract_total_vn(text)

    print("\n===== RESULT =====")
    print("Total:", total)


if __name__ == "__main__":
    test_image("data/invoice1.jpg")