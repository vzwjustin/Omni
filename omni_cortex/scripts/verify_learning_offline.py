"""Verify learning flow with offline/mock embeddings."""

import asyncio
import logging

from app.collection_manager import CollectionManager, get_collection_manager
from app.nodes.common import format_code_context
from app.state import create_initial_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_learning")


# Mock Embeddings to avoid needing API keys for testing
class MockEmbeddings:
    def embed_documents(self, texts):
        # Return fake vectors of dimension 1536 (OpenAI standard)
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, _text):
        return [0.1] * 1536


# Patch CollectionManager to use mock embeddings

original_get_embedding = CollectionManager.get_embedding_function

def mock_get_embedding(self):
    return MockEmbeddings()

CollectionManager.get_embedding_function = mock_get_embedding

async def test_learning_flow():
    print("\n🚀 Testing Learning System Flow...\n")

    # 1. Add a fake learning
    print("1. Adding fake learning...")
    cm = get_collection_manager()
    success = cm.add_learning(
        query="AttributeError: 'NonType' object has no attribute 'foo'",
        answer="Check if the variable is None before accessing the attribute.",
        framework_used="active_inference",
        success_rating=1.0,
        problem_type="debugging"
    )

    if success:
        print("✅ Learning added successfully.")
    else:
        print("❌ Failed to add learning.")
        return

    # 2. Simulate Router Retrieval
    print("\n2. Testing Router Retrieval...")
    query = "I'm getting an AttributeError: 'NoneType' object has no attribute 'bar'"
    print(f"   Query: {query}")

    # Create a mock state
    state = create_initial_state(query=query)

    # Manually trigger the retrieval logic that we added to router.py
    # Since we can't easily instantiate the full router with all dependencies in this script without mocking,
    # we'll test the retrieval logic directly using the CollectionManager first,
    # then verify the integration point if possible.

    learnings = cm.search_learnings(query, k=3)
    print(f"   Found {len(learnings)} learnings.")

    if len(learnings) > 0:
        print(f"✅ Retrieved learning: {learnings[0]['problem']} -> {learnings[0]['solution'][:50]}...")
        state["episodic_memory"] = learnings
    else:
        print("❌ Failed to retrieve learning.")
        return

    # 3. Verify Context Formatting
    print("\n3. Verifying Context Formatting...")
    formatted_context = format_code_context(
        code_snippet="x = None\nx.bar()",
        file_list=["test.py"],
        ide_context=None,
        state=state
    )

    print("\n--- Formatted Context Preview ---")
    print(formatted_context)
    print("---------------------------------")

    if "Past Learnings" in formatted_context and "Check if the variable is None" in formatted_context:
        print("\n✅ SUCCESS: Context includes past learnings!")
    else:
        print("\n❌ FAILURE: Context missing learnings.")

    # 4. Test Rating Filtering
    print("\n4. Testing Rating Filtering...")
    # Add a low-quality learning
    cm.add_learning(
        query="Bad query",
        answer="Bad answer",
        framework_used="random",
        success_rating=0.1,  # Should be filtered out (default min is 0.5)
        problem_type="noise"
    )

    filtered_results = cm.search_learnings("Bad query", k=5, min_rating=0.5)
    bad_results = [r for r in filtered_results if r['solution'] == "Bad answer"]

    if len(bad_results) == 0:
        print("✅ SUCCESS: Low-rated learning was filtered out.")
    else:
        print("❌ FAILURE: Low-rated learning was returned.")

    # 5. Test Different Problem Type (Optimization)
    print("\n5. Testing Optimization Scenario...")
    cm.add_learning(
        query="Loop is too slow",
        answer="Use vectorization with numpy instead of for loop.",
        framework_used="comparative_arch",
        success_rating=0.9,
        problem_type="optimization"
    )

    opt_results = cm.search_learnings("My loop is slow", k=1)
    if opt_results and "vectorization" in opt_results[0]['solution']:
        print("✅ SUCCESS: Optimization learning retrieved.")
    else:
        print("❌ FAILURE: Optimization learning not found.")

if __name__ == "__main__":
    asyncio.run(test_learning_flow())
