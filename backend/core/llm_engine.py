
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from embedding import embed_text
from database import search_similar_chunks, get_document

load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass
class SourceReference:
    """Reference to a source chunk used in the answer."""
    content: str
    chunk_index: int
    document_id: str
    relevance_score: float


@dataclass
class QueryResult:
    """Result of a RAG query containing answer and sources."""
    answer: str
    sources: List[SourceReference]
    query: str


def get_llm(model: str = "gemini-3-pro", temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Get a configured Gemini LLM instance.
    
    Args:
        model: Model name to use
        temperature: Sampling temperature (0-1)
        
    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature
    )


def retrieve_context(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5
) -> List[SourceReference]:
    """
    Retrieve relevant chunks for a query using vector similarity search.
    
    Args:
        query: The user's question
        document_id: Optional document to search within
        top_k: Number of chunks to retrieve
        
    Returns:
        List of SourceReference objects with relevant content
    """
    # Generate embedding for the query
    query_embedding = embed_text(query)
    
    # Search for similar chunks
    results = search_similar_chunks(
        query_embedding=query_embedding,
        document_id=document_id,
        limit=top_k
    )
    
    # Convert to SourceReference objects
    sources = []
    for result in results:
        sources.append(SourceReference(
            content=result["content"],
            chunk_index=result["chunk_index"],
            document_id=result["document_id"],
            relevance_score=result.get("distance", 0.0)
        ))
    
    return sources


def build_context_string(sources: List[SourceReference]) -> str:
    """Build a formatted context string from source references."""
    context_parts = []
    for i, source in enumerate(sources, 1):
        context_parts.append(f"[Source {i}]\n{source.content}")
    return "\n\n".join(context_parts)


# RAG Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided context from PDF documents.

Guidelines:
- Answer ONLY based on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite your sources using [Source X] notation
- Be concise but comprehensive
- Maintain accuracy - don't make up information"""),
    ("human", """Context from PDF:
{context}

Question: {question}

Please provide a detailed answer based on the context above, citing relevant sources.""")
])


def query_pdf(
    question: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
    model: str = "gemini-3-pro"
) -> QueryResult:
    """
    Answer a question about a PDF using RAG.
    
    This function:
    1. Embeds the question
    2. Retrieves relevant chunks via similarity search
    3. Generates an answer with source citations
    
    Args:
        question: The user's question
        document_id: Optional specific document to query
        top_k: Number of context chunks to retrieve
        model: LLM model to use for generation
        
    Returns:
        QueryResult with answer and source references
    """
    # Step 1: Retrieve relevant context
    sources = retrieve_context(question, document_id, top_k)
    
    if not sources:
        return QueryResult(
            answer="I couldn't find any relevant information in the document(s) to answer your question.",
            sources=[],
            query=question
        )
    
    # Step 2: Build context string
    context = build_context_string(sources)
    
    # Step 3: Generate answer using LLM
    llm = get_llm(model)
    chain = RAG_PROMPT | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    return QueryResult(
        answer=answer,
        sources=sources,
        query=question
    )


def format_result(result: QueryResult) -> str:
    """Format a QueryResult for display."""
    output = []
    output.append("=" * 60)
    output.append(f"Question: {result.query}")
    output.append("=" * 60)
    output.append("\n📝 Answer:\n")
    output.append(result.answer)
    
    if result.sources:
        output.append("\n\n📚 Sources Used:")
        output.append("-" * 40)
        for i, source in enumerate(result.sources, 1):
            preview = source.content[:200] + "..." if len(source.content) > 200 else source.content
            output.append(f"\n[Source {i}] (Chunk {source.chunk_index}, Score: {source.relevance_score:.4f})")
            output.append(f"  {preview}")
    
    return "\n".join(output)


# --- TESTING ---
if __name__ == "__main__":
    from database import init_db, create_document, create_chunks, delete_document
    from embedding import embed_text, embed_texts, chunk_text
    
    print("Testing RAG Query Engine...")
    print("-" * 50)
    
    # Initialize database
    init_db()
    
    # Create test document with sample content
    test_content = """
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    
    Deep learning uses neural networks with multiple layers to process complex patterns.
    Neural networks are inspired by the human brain's structure.
    
    Natural language processing (NLP) helps computers understand human language.
    Common NLP tasks include sentiment analysis, translation, and question answering.
    
    Computer vision enables machines to interpret visual information from images and videos.
    Applications include facial recognition, object detection, and autonomous vehicles.
    """
    
    # Create document
    doc_id = create_document(
        filename="test_ml_document.pdf",
        extracted_text=test_content,
        page_count=1
    )
    print(f"✓ Created test document: {doc_id}")
    
    # Chunk and embed the content
    chunks = chunk_text(test_content, chunk_size=200, chunk_overlap=50)
    chunk_embeddings = embed_texts(chunks)
    create_chunks(doc_id, list(zip(chunks, chunk_embeddings)))
    print(f"✓ Created {len(chunks)} chunks with embeddings")
    
    # Test query
    test_question = "What is deep learning and how does it relate to neural networks?"
    print(f"\n🔍 Testing query: '{test_question}'")
    
    result = query_pdf(test_question, document_id=doc_id)
    print(format_result(result))
    
    # Cleanup
    delete_document(doc_id)
    print("\n✓ Cleaned up test document")
    print("\n✅ RAG Query Engine test complete!")
