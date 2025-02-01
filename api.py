from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import pytesseract
import spacy
from fastapi.responses import JSONResponse


app = FastAPI()

# Loading the SciSpacy model (ensure this is installed correctly)
nlp = spacy.load("en_core_sci_sm")


def preprocess_image(image_bytes):
    """Convert image bytes to a preprocessed grayscale image."""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    # Try different preprocessing techniques
    # img = cv2.resize(img, (800, 800))
    # _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return img

# def preprocess_image(image_bytes):
#     """Convert image bytes to a preprocessed grayscale image."""
#     np_img = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (800, 800))
#     _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
#     return img

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """API to process medical prescription images"""
    try:
        image_bytes = await file.read()
        processed_img = preprocess_image(image_bytes)

        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(processed_img, lang="eng")


        # Process extracted text using NLP
        doc = nlp(text)
        medical_terms = [(ent.text, ent.label_) for ent in doc.ents]

        return JSONResponse(content={"extracted_text": text, "medical_data": medical_terms})

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
