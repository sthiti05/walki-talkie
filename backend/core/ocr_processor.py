import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import shutil

# CONFIGURATION:
# If Tesseract is not in your PATH, uncomment and set the path manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# If Poppler is not in your PATH, set the path to the bin folder here:
# Download pre-built Windows binaries from: https://github.com/ossamamehmood/Poppler-windows/releases
POPPLER_PATH = r'C:\poppler\Library\bin'  # UPDATE THIS to your actual poppler bin path


def convert_pdf_to_images(pdf_path, temp_folder="data/temp_images"):
    """
    Converts a PDF file into a list of image paths.
    """
    # Create temp folder if it doesn't exist
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    print(f"Converting PDF: {pdf_path} to images...")
    
    # dpi=300 is standard for good OCR results
    images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    
    saved_image_paths = []
    
    for i, image in enumerate(images):
        image_name = f"page_{i + 1}.jpg"
        image_path = os.path.join(temp_folder, image_name)
        image.save(image_path, "JPEG")
        saved_image_paths.append(image_path)
        
    print(f"Converted {len(images)} pages.")
    return saved_image_paths


def extract_text_from_images(image_paths):
    """
    Iterates over image paths and uses Tesseract to extract text.
    """
    full_document_text = ""

    print("Starting OCR processing...")
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing Page {i + 1}...")
        
        # Load image
        img = Image.open(img_path)
        
        # Extract text (lang='eng' for English)
        text = pytesseract.image_to_string(img, lang='eng')
        
        # Append to full text with a page separator
        full_document_text += f"\n--- PAGE {i + 1} ---\n"
        full_document_text += text

    return full_document_text


def clean_up_temp(temp_folder="data/temp_images"):
    """
    Deletes the temporary images to save space.
    """
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
        print("Temporary images cleaned up.")


def process_pdf_pipeline(pdf_path, cleanup=True):
    """
    Main function: PDF -> Images -> Text
    """
    try:
        # 1. PDF -> Images
        image_paths = convert_pdf_to_images(pdf_path)
        
        # 2. Images -> Text
        extracted_text = extract_text_from_images(image_paths)
        
        # 3. Cleanup
        if cleanup:
            clean_up_temp()
            
        return extracted_text
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None


# --- TESTING BLOCK ---
if __name__ == "__main__":
    # Point to your PDF file
    test_pdf = r"C:/Users/LENOVO/Desktop/AI/project/energies-16-07680-v2.pdf"
    
    # Check if file exists before running
    if os.path.exists(test_pdf):
        result = process_pdf_pipeline(test_pdf, cleanup=False)
        print("\n--- FINAL EXTRACTED TEXT ---\n")
        print(result[:500] if result else "No text extracted")
    else:
        print(f"PDF not found at: {test_pdf}")
        print("Please update the 'test_pdf' path to point to your PDF file.")