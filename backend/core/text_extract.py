from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pypdf.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text with page separators
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n--- PAGE ---\n"
    return text


# Test code - can be removed or commented out
if __name__ == "__main__":
    print(extract_text_from_pdf(r"C:\Users\LENOVO\Desktop\New folder\sthiti_resume.pdf"))