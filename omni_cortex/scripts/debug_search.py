#!/usr/bin/env python3
"""
Debug script to trace ChromaDB search issues.
Run from repo root: python -m scripts.debug_search
"""
import os
import sys
import traceback

# Ensure we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("ChromaDB Search Debug")
    print("=" * 60)
    
    # Check environment
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/data/chroma")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    print(f"\n1. Environment:")
    print(f"   CHROMA_PERSIST_DIR: {persist_dir}")
    print(f"   OPENAI_API_KEY set: {bool(openai_key)}")
    print(f"   Key prefix: {openai_key[:20]}..." if openai_key else "   No key!")
    
    # Check if chroma directory exists
    if os.path.exists(persist_dir):
        files = os.listdir(persist_dir)
        print(f"   Chroma dir exists: {files}")
    else:
        print(f"   ERROR: Chroma dir does not exist!")
        return
    
    # Try to initialize embeddings
    print(f"\n2. Initializing Embeddings...")
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=openai_key
        )
        print("   ✓ OpenAIEmbeddings initialized")
        
        # Test embedding a query
        print("   Testing embedding generation...")
        test_embed = embeddings.embed_query("test query")
        print(f"   ✓ Generated embedding of dimension: {len(test_embed)}")
    except Exception as e:
        print(f"   ✗ Embedding error: {e}")
        traceback.print_exc()
        return
    
    # Try to load vectorstore
    print(f"\n3. Loading Vectorstore...")
    try:
        from langchain_chroma import Chroma
        vs = Chroma(
            collection_name="omni-cortex-context",
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print(f"   ✓ Chroma vectorstore loaded")
        
        # Check collection info
        if hasattr(vs, '_collection'):
            coll = vs._collection
            count = coll.count()
            print(f"   Collection name: {coll.name}")
            print(f"   Document count: {count}")
        else:
            print("   Warning: Cannot access _collection attribute")
    except Exception as e:
        print(f"   ✗ Vectorstore error: {e}")
        traceback.print_exc()
        return
    
    # Try a search
    print(f"\n4. Testing Search...")
    test_queries = [
        "framework reasoning",
        "def main",
        "class",
        "import",
        "LangChain"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = vs.similarity_search(query, k=3)
            print(f"   Results: {len(results)}")
            for i, doc in enumerate(results):
                path = doc.metadata.get("path", "unknown")
                preview = doc.page_content[:80].replace("\n", " ")
                print(f"     {i+1}. {path}: {preview}...")
        except Exception as e:
            print(f"   ✗ Search error: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
