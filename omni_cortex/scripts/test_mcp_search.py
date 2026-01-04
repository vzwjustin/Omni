#!/usr/bin/env python3
"""
Test the MCP server's search_documentation tool EXACTLY as it would be called.
"""
import asyncio
import sys
import os

# Set up logging to stderr like MCP server does
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

async def test_mcp_search():
    print("Testing MCP search_documentation flow...", file=sys.stderr)
    
    # Import the same way main.py does
    from app.langchain_integration import search_vectorstore
    from app.collection_manager import get_collection_manager
    
    print("\n1. Testing search_vectorstore directly:", file=sys.stderr)
    try:
        docs = search_vectorstore("framework reasoning", k=5)
        print(f"   Results: {len(docs)}", file=sys.stderr)
        for i, doc in enumerate(docs):
            print(f"   {i+1}. {doc.metadata.get('path', 'unknown')}", file=sys.stderr)
    except Exception as e:
        print(f"   ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing collection_manager:", file=sys.stderr)
    try:
        manager = get_collection_manager()
        print(f"   COLLECTIONS defined: {list(manager.COLLECTIONS.keys())}", file=sys.stderr)
        
        # Test if collections exist
        for coll_name in ["frameworks", "documentation", "utilities"]:
            coll = manager.get_collection(coll_name)
            if coll:
                count = coll._collection.count() if hasattr(coll, '_collection') else "?"
                print(f"   {coll_name}: loaded, count={count}", file=sys.stderr)
            else:
                print(f"   {coll_name}: FAILED TO LOAD", file=sys.stderr)
    except Exception as e:
        print(f"   ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    print("\n3. Simulating MCP call_tool for search_documentation:", file=sys.stderr)
    try:
        # Simulate the exact code path in call_tool
        query = "framework reasoning"
        k = 5
        docs = search_vectorstore(query, k=k)
        if not docs:
            result = "No results found. Try refining your query."
        else:
            formatted = []
            for d in docs:
                meta = d.metadata or {}
                path = meta.get("path", "unknown")
                formatted.append(f"### {path}\n{d.page_content[:100]}...")
            result = "\n\n".join(formatted)
        
        print(f"   Result preview:", file=sys.stderr)
        print(result[:500], file=sys.stderr)
    except Exception as e:
        print(f"   ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_search())
