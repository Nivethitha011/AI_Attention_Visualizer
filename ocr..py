import easyocr
import numpy as np


def load_ocr():
    """
    Load EasyOCR English reader.
    """
    return easyocr.Reader(
        ["en"],
        gpu=False
    )


def extract_text(reader, image):
    """
    Extract text from an image.

    Returns:
        extracted_text
        detection_results
    """

    image_array = np.array(image)

    results = reader.readtext(image_array)

    extracted_text = " ".join(
        result[1]
        for result in results
    )

    return extracted_text, results
