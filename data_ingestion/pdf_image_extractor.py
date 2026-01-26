import fitz #for PyMuPDF
from typing import List, Tuple
from PIL import Image
import io

def extract_images_from_pdf(pdf_path: str) -> List[Tuple[Image.Image, dict]]:
    """Returns:
    List of (PIL.Image, metadata_dict)"""

    doc = fitz.open(pdf_path)
    results = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            metadata = {"page": page_index + 1,
                        "image_index": img_index}
            
            results.append((image, metadata))

    return results
