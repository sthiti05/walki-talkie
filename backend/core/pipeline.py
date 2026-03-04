from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from text_extract import extract_text_from_pdf
from embedding import chunk_text, embed_texts
from database import init_db, create_document, create_chunks, get_document, list_documents
from llm_engine import query_pdf, QueryResult, format_result


@dataclass
class ProcessedDocument:
    """Result of processing a PDF."""
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str


def process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> ProcessedDocument:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    print(f"📄 Processing: {pdf_path.name}")
    
    print("  → Extracting text with pypdf...")
    extracted_text = extract_text_from_pdf(str(pdf_path))

    page_count = extracted_text.count("--- PAGE")
    if page_count == 0:
        page_count = 1
    
    print(f"  → Extracted {len(extracted_text)} characters from {page_count} page(s)")
    
    
    print("  → Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  → Created {len(chunks)} chunks")
    

    print("  → Generating embeddings...")
    embeddings = embed_texts(chunks)
    print(f"  → Generated {len(embeddings)} embeddings")
    
    print("  → Storing in database...")
    init_db()

    doc_id = create_document(
        filename=pdf_path.name,
        extracted_text=extracted_text,
        file_path=str(pdf_path),
        page_count=page_count
    )
    
    create_chunks(doc_id, list(zip(chunks, embeddings)))
    print(f"  ✓ Document stored with ID: {doc_id}")
    
    return ProcessedDocument(
        document_id=doc_id,
        filename=pdf_path.name,
        page_count=page_count,
        chunk_count=len(chunks),
        status="ready"
    )


def ask_question(question: str, document_id: Optional[str] = None) -> QueryResult:
    """
    Ask a question about processed PDF(s).
    
    Args:
        question: Your question about the PDF content
        document_id: Optional - query specific document, or all if None
        
    Returns:
        QueryResult with answer and source references
    """
    return query_pdf(question, document_id=document_id)


def show_documents():
    """List all processed documents."""
    docs = list_documents()
    
    if not docs:
        print("No documents found. Process a PDF first!")
        return
    
    print("\n📚 Processed Documents:")
    print("-" * 60)
    for doc in docs:
        print(f"  ID: {doc['id'][:8]}...")
        print(f"  File: {doc['filename']}")
        print(f"  Pages: {doc['page_count']}")
        print(f"  Status: {doc['status']}")
        print(f"  Created: {doc['created_at']}")
        print("-" * 60)


# --- TESTING / CLI ---
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("   PDF Q&A Pipeline - Walki Talkie to PDF")
    print("=" * 60)
    
    # Initialize database
    init_db()
    
    # Example usage
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        
        # Process the PDF
        result = process_pdf(pdf_file)
        print(f"\n✅ PDF processed successfully!")
        print(f"   Document ID: {result.document_id}")
        print(f"   Chunks: {result.chunk_count}")
        
        # Interactive Q&A loop
        print("\n💬 Ask questions about your PDF (type 'quit' to exit):\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            answer = ask_question(question, document_id=result.document_id)
            print(f"\n🤖 Assistant:\n{answer.answer}\n")
    else:
        print("\nUsage:")
        print("  python pipeline.py <path_to_pdf>")
        print("\nExample:")
        print("  python pipeline.py document.pdf")
        print("\nOr import in your code:")
        print("  from pipeline import process_pdf, ask_question")
        print("  doc = process_pdf('my_file.pdf')")
        print("  result = ask_question('What is this about?', doc.document_id)")
