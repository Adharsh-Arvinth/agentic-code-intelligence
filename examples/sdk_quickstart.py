"""
Agentic Code Intelligence - Python SDK Quickstart Example.

Demonstrates using the high-level CodeIntelligenceClient SDK to perform:
1. Natural language code search across indexed versions
2. Accessing ranked results, similarity scores, and code previews
3. AST-guarded sandboxed execution verification
"""

import sys
import os

# Add root directory to path for local execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic_code_intelligence import CodeIntelligenceClient


def main():
    print("=" * 60)
    print("Agentic Code Intelligence - Python SDK Quickstart")
    print("=" * 60)

    # 1. Initialize the SDK Client
    client = CodeIntelligenceClient(device="cpu")
    print(f"\n[SDK Status]: {client.health_check()}")

    # 2. Perform a Natural Language Code Search
    query = "Find the shortest path in a weighted graph using Dijkstra algorithm with a priority queue"
    print(f"\n[Searching Query]: '{query}'")
    results = client.search(query=query, top_k=2, version="default")

    print(f"\nRetrieved {len(results)} ranked solutions:")
    for r in results:
        print(f"\n  -> Rank {r.rank} | Doc: {r.doc_id} | Score: {r.score:.6f} | Version: {r.version}")
        preview_lines = r.code.strip().split("\n")[:4]
        preview = "\n     ".join(preview_lines)
        print(f"     Preview:\n     {preview}\n     ...")

    # 3. Test In-Memory Execution Verification
    test_input = "4\n"
    test_expected_output = "16"
    test_code = "n = int(input())\nprint(n * n)\n"
    print(f"\n[Verifying Solution via AST Sandbox]:")
    print(f"  Input: {test_input.strip()} -> Expected Output: {test_expected_output}")
    verification = client.verify_execution(
        sample_input=test_input,
        expected_output=test_expected_output,
        code_snippet=test_code
    )
    print(f"  Execution Match Result: {verification}")

    print("\n" + "=" * 60)
    print("SDK Quickstart completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
