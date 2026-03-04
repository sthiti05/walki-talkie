from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from gemini_ocr import extract_text_from_pdf  # ← Import OCR function

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )



def embed_text(text: str) -> List[float]:
    model = get_embedding_model()
    return model.embed_query(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.embed_documents(texts)
    

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return splitter.split_text(text)


def process_pdf_text(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Tuple[str, List[float]]]:
    """
    Complete flow: PDF → Gemini OCR → Chunk → Embed
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of (chunk_text, embedding) tuples
    """
    # Step 1: Get text from Gemini OCR
    print("  → Extracting text from PDF using Gemini OCR...")
    extracted_text = extract_text_from_pdf(pdf_path)
    print(f"  → Extracted {len(extracted_text)} characters")
    
    # Step 2: Chunk the extracted text
    print("  → Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"  → Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings for each chunk
    print("  → Generating embeddings...")
    embeddings = embed_texts(chunks)
    print(f"  → Generated {len(embeddings)} embeddings")
    
    return list(zip(chunks, embeddings))


# --- TESTING ---
if __name__ == "__main__":
    import os
    
    # Test with a real PDF using Gemini OCR
    test_pdf = r"C:\Users\LENOVO\Desktop\AI\project\Magnetic_Signature-Based_Model_Using_Machine_Learning_for_Electrical_and_Mechanical_Faults_Classification_of_Wind_Turbine_Drive_Trains.pdf"
    
    if os.path.exists(test_pdf):
        print("Testing full pipeline: PDF → OCR → Chunk → Embed\n")
        
        results = process_pdf_text(test_pdf)
        
        print(f"\n✅ Done! Got {len(results)} chunks with embeddings")
        print(f"   First chunk preview: {results[0][0][:100]}...")
        print(f"   Embedding dimensions: {len(results[0][1])}")
    else:
        print(f"PDF not found: {test_pdf}")
        print("Update the path to test with your PDF.")
