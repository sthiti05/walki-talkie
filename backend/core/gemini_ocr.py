
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv(Path(__file__).parent.parent / ".env")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using Gemini's vision capabilities via LangChain.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Extracted text from the PDF
    """
    import base64
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Read and encode PDF as base64
    with open(pdf_path, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    # Create message with PDF
    message = HumanMessage(
        content=[
            {
                "type": "media",
                "mime_type": "application/pdf",
                "data": pdf_data,
            },
            {
                "type": "text",
                "text": """Extract ALL text from this PDF document verbatim. 
Preserve structure and formatting. Separate pages with "--- PAGE X ---".
Return ONLY the extracted text."""
            },
        ]
    )
    
    response = llm.invoke([message])
    return response.content


# --- TESTING ---
if __name__ == "__main__":
    test_pdf = r"C:\Users\LENOVO\Desktop\AI\project\Magnetic_Signature-Based_Model_Using_Machine_Learning_for_Electrical_and_Mechanical_Faults_Classification_of_Wind_Turbine_Drive_Trains.pdf"
    
    if os.path.exists(test_pdf):
        result = extract_text_from_pdf(test_pdf)
        print("\n--- EXTRACTED TEXT (first 1000 chars) ---\n")
        print(result[:1000] if result else "No text extracted")
    else:
        print(f"PDF not found at: {test_pdf}")
