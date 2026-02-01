import sqlite3
import sqlite_vec
import uuid
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import contextmanager


DB_PATH = Path(__file__).parent / "data" / "app.db"


def get_connection() -> sqlite3.Connection:
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    return conn


@contextmanager
def db_connection():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_connection() as conn:
        # Create documents table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT,
                extracted_text TEXT,
                page_count INTEGER,
                status TEXT DEFAULT 'processing',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(
                id TEXT PRIMARY KEY,
                document_id TEXT,
                content TEXT,
                chunk_index INTEGER,
                embedding FLOAT[768]
            )
        """)
        
        print(f"✓ Database initialized at {DB_PATH}")


def generate_id() -> str:
    return str(uuid.uuid4())



def create_document(
    filename: str,
    extracted_text: str,
    file_path: Optional[str] = None,
    page_count: Optional[int] = None
) -> str:
    doc_id = generate_id()
    
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, file_path, extracted_text, page_count, status)
            VALUES (?, ?, ?, ?, ?, 'ready')
            """,
            (doc_id, filename, file_path, extracted_text, page_count)
        )
    
    return doc_id


def get_document(doc_id: str) -> Optional[dict]:
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,)
        ).fetchone()
        
        return dict(row) if row else None


def list_documents() -> List[dict]:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT id, filename, page_count, status, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
        
        return [dict(row) for row in rows]


def create_chunks(document_id: str, chunks: List[Tuple[str, List[float]]]):
    with db_connection() as conn:
        for idx, (content, embedding) in enumerate(chunks):
            chunk_id = generate_id()
            
            import struct
            embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)
            
            conn.execute(
                """
                INSERT INTO chunks (id, document_id, content, chunk_index, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, content, idx, embedding_bytes)
            )


def search_similar_chunks(
    query_embedding: List[float],
    document_id: Optional[str] = None,
    limit: int = 5
) -> List[dict]:
    """
    Search for similar chunks using vector similarity.
    
    Args:
        query_embedding: The query vector (768 dimensions)
        document_id: Optional filter by document
        limit: Maximum results to return
        
    Returns:
        List of matching chunks with distance scores
    """
    import struct
    
    with db_connection() as conn:
        query_bytes = struct.pack(f'{len(query_embedding)}f', *query_embedding)
        
        if document_id:
            rows = conn.execute(
                """
                SELECT id, document_id, content, chunk_index, distance
                FROM chunks
                WHERE embedding MATCH ? AND document_id = ?
                ORDER BY distance
                LIMIT ?
                """,
                (query_bytes, document_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, document_id, content, chunk_index, distance
                FROM chunks
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (query_bytes, limit)
            ).fetchall()
        
        return [dict(row) for row in rows]


def delete_document(doc_id: str):
    with db_connection() as conn:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# --- TESTING ---
if __name__ == "__main__":
    print("Testing database module...")
    
    # Initialize database
    init_db()
    
    # Test document creation
    doc_id = create_document(
        filename="test.pdf",
        extracted_text="This is test content from the PDF.",
        page_count=1
    )
    print(f"✓ Created document: {doc_id}")
    
    # Test document retrieval
    doc = get_document(doc_id)
    print(f"✓ Retrieved document: {doc['filename']}")
    
    # Test chunk creation with dummy embedding
    dummy_embedding = [0.1] * 768
    create_chunks(doc_id, [
        ("First chunk of text", dummy_embedding),
        ("Second chunk of text", dummy_embedding),
    ])
    print("✓ Created chunks with embeddings")
    
    # Test vector search
    results = search_similar_chunks(dummy_embedding, limit=2)
    print(f"✓ Vector search returned {len(results)} results")
    
    # Cleanup
    delete_document(doc_id)
    print("✓ Deleted test document")
    
    print("\n✅ All database tests passed!")
