import pytesseract
try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version: {version}")
except Exception as e:
    print(f"Error: {e}")
    print("Tesseract is not installed or not in PATH")
    print("Please install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
