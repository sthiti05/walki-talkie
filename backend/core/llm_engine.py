
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


def get_llm(model: str = "gemini-3.1-flash-preview", temperature: float = 0.3) -> ChatGoogleGenerativeAI:
 
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature
    )


def retrieve_context(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5
) -> List[SourceReference]:
    
    query_embedding = embed_text(query)
    
    results = search_similar_chunks(
        query_embedding=query_embedding,
        document_id=document_id,
        limit=top_k
    )
    
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
    context_parts = []
    for i, source in enumerate(sources, 1):
        context_parts.append(f"[Source {i}]\n{source.content}")
    return "\n\n".join(context_parts)


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
    model: str = "gemini-2.5-flash"
) -> QueryResult:     
    sources = retrieve_context(question, document_id, top_k)
    
    if not sources:
        return QueryResult(
            answer="I couldn't find any relevant information in the document(s) to answer your question.",
            sources=[],
            query=question
        )
    
    context = build_context_string(sources)
    
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
    from database import init_db, list_documents
    
    print("Testing RAG Query Engine...")
    print("-" * 50)
    
    init_db()
    
    # Check for existing documents in database
    docs = list_documents()
    
    if not docs:
        print("⚠️  No documents in database!")
        print("First process a PDF using pipeline.py:")
        print("  uv run python core/pipeline.py your_document.pdf")
    else:
        print(f"Found {len(docs)} document(s) in database:\n")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc['filename']} (ID: {doc['id'][:8]}...)")
        
        # Use the first document for testing
        doc_id = docs[0]['id']
        print(f"\n🔍 Testing query on: {docs[0]['filename']}")
        
        # Test query
        test_question = "What is this document about?"
        print(f"   Question: '{test_question}'")
        
        result = query_pdf(test_question, document_id=doc_id)
        print("\n" + format_result(result))
        
        print("\n✅ RAG Query Engine test complete!")

