import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from tests.test_utils import (
    SimpleIntentionAgent, SimpleRetrieverAgent,
    SimpleReasonAgent, SimpleWriterAgent, SimpleDesignerAgent
)

async def run_flow():
    # Initialize agents
    intention_agent = SimpleIntentionAgent()
    retriever_agent = SimpleRetrieverAgent()
    reason_agent = SimpleReasonAgent()
    writer_agent = SimpleWriterAgent()
    designer_agent = SimpleDesignerAgent()

    # Sample query
    query = "¿Cuál es el estado actual del mercado de inversiones?"
    context = {"query_id": "123", "original_query": query}
    
    print("\n=== Running Financial Agent Flow ===\n")
    print(f"Input Query: {query}")
    print(f"Context: {context}\n")

    # Step 1: Intention Agent
    print("=== Step 1: Intention Agent ===")
    intention_input = {"query": query, "context": context}
    intention_output = await intention_agent.process(intention_input)
    print(f"Input: {intention_input}")
    print(f"Output: {intention_output}\n")

    # Step 2: Retriever Agent
    print("=== Step 2: Retriever Agent ===")
    retriever_input = {"query": query, "context": context}
    retriever_output = await retriever_agent.process(retriever_input)
    print(f"Input: {retriever_input}")
    print(f"Output: {retriever_output}\n")

    # Step 3: Reason Agent
    print("=== Step 3: Reason Agent ===")
    reason_input = {
        "query": query,
        "intent": intention_output["intent"],
        "information": retriever_output["information"],
        "context": context
    }
    reason_output = await reason_agent.process(reason_input)
    print(f"Input: {reason_input}")
    print(f"Output: {reason_output}\n")

    # Step 4: Writer Agent
    print("=== Step 4: Writer Agent ===")
    writer_input = {
        "analysis": reason_output["analysis"],
        "context": context
    }
    writer_output = await writer_agent.process(writer_input)
    print(f"Input: {writer_input}")
    print(f"Output: {writer_output}\n")

    # Step 5: Designer Agent
    print("=== Step 5: Designer Agent ===")
    designer_input = {
        "query": query,
        "response": writer_output["response"],
        "context": context
    }
    designer_output = await designer_agent.process(designer_input)
    print(f"Input: {designer_input}")
    print(f"Output: {designer_output}\n")

if __name__ == "__main__":
    asyncio.run(run_flow())
