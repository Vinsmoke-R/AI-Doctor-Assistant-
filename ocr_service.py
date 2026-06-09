import pytesseract
from PIL import Image
import io
import base64
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def extract_text(base64_string: str) -> str:
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text
