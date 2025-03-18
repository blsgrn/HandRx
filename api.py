from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import pytesseract
import spacy
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load SciSpacy model (use medium or large model if available)
try:
    nlp = spacy.load("en_core_sci_md")
except Exception as e:
    raise ImportError(f"Error loading NLP model: {str(e)}")


def preprocess_image(image_bytes):
    """Convert image bytes to a preprocessed grayscale image."""
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (800, 800))
        img = cv2.GaussianBlur(img, (5, 5), 0)

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        return img

    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """API to process medical prescription images"""
    try:
        logger.info("Received file: %s", file.filename)

        image_bytes = await file.read()

        processed_img = preprocess_image(image_bytes)

        custom_config = r'--oem 3 --psm 6'

        text = pytesseract.image_to_string(processed_img, lang="eng", config=custom_config)

        doc = nlp(text)

        medical_terms = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        logger.info("Processing completed successfully.")

        return JSONResponse(content={"extracted_text": text,
                                     "medical_data": medical_terms})

    except ValueError as ve:
        logger.error("Preprocessing failed: %s", str(ve))

        return JSONResponse(status_code=400,
                            content={"error": f"Preprocessing failed: {str(ve)}"})

    except Exception as e:

        logger.error("Internal server error: %s", str(e))

        return JSONResponse(status_code=500,
                            content={"error": f"Internal server error: {str(e)}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
