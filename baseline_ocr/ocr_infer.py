import pytesseract
from PIL import Image

def run_ocr(image):
    text = pytesseract.image_to_string(image, lang="vie+eng")
    return text