from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/geminiembedding-001")


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


# --- TESTING ---
if __name__ == "__main__":
    # Test embedding generation
    test_text = "This is a test sentence for generating embeddings."
    
    print("Testing Gemini embedding-001...")
    embedding = embed_text(test_text)
    
    print(f"✓ Embedding generated successfully")
    print(f"  Dimensions: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    test_texts = [
        "First document about machine learning.",
        "Second document about natural language processing.",
        "Third document about computer vision."
    ]
    
    print("\nTesting batch embedding...")
    embeddings = embed_texts(test_texts)
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    # Test chunking
    long_text = "This is a longer text. " * 100
    chunks = chunk_text(long_text)
    print(f"\n✓ Split text into {len(chunks)} chunks")
