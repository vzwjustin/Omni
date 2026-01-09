import asyncio
import os
import sys

# Add current directory to path so we can import app
sys.path.append(os.getcwd())

from omni_cortex.app.core.context.query_analyzer import QueryAnalyzer
from omni_cortex.app.core.settings import get_settings

async def main():
    print("Starting debug...")
    try:
        analyzer = QueryAnalyzer()
        print(f"Analyzer initialized. Model: {analyzer.settings.routing_model}")
        
        query = "Conduct a comprehensive code quality and architecture audit"
        print(f"Analyzing query: {query}")
        
        result = await analyzer.analyze(query)
        print("Analysis successful!")
        print(result)
    except Exception as e:
        print(f"Analysis failed with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())