import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from tests.test_utils import (
    SimpleIntentionAgent, SimpleRetrieverAgent,
    SimpleReasonAgent, SimpleWriterAgent, SimpleDesignerAgent
)
from src.domain.models import UserQuery
import asyncio
from datetime import datetime, UTC
import json

async def test_agent_flow():
    print("\n=== Testing Agent Flow ===\n")
    
    # Initialize agents
    intention_agent = SimpleIntentionAgent()
    retriever_agent = SimpleRetrieverAgent()
    reason_agent = SimpleReasonAgent()
    writer_agent = SimpleWriterAgent()
    designer_agent = SimpleDesignerAgent()
    
    # Test query
    query_text = "como me afecta el precio del dolar"
    print(f"Input Query: {query_text}\n")
    
    # Create context
    context = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query_id": "test_flow_1"
    }
    
    # Step 1: Intention Agent
    print("=== Step 1: Intention Agent ===")
    intention_input = {"query": query_text, "context": context}
    print(f"Input: {json.dumps(intention_input, indent=2)}")
    intention_output = await intention_agent.process(intention_input)
    print(f"Output: {json.dumps(intention_output, indent=2)}\n")
    
    # Step 2: Retriever Agent
    print("=== Step 2: Retriever Agent ===")
    retriever_input = {
        "query": query_text,
        "context": context,
        "intent": intention_output["intent"]
    }
    print(f"Input: {json.dumps(retriever_input, indent=2)}")
    retriever_output = await retriever_agent.process(retriever_input)
    print(f"Output: {json.dumps(retriever_output, indent=2)}\n")
    
    # Step 3: Reason Agent
    print("=== Step 3: Reason Agent ===")
    reason_input = {
        "query": query_text,
        "context": context,
        "intent": intention_output["intent"],
        "information": retriever_output["information"]
    }
    print(f"Input: {json.dumps(reason_input, indent=2)}")
    reason_output = await reason_agent.process(reason_input)
    print(f"Output: {json.dumps(reason_output, indent=2)}\n")
    
    # Step 4: Writer Agent
    print("=== Step 4: Writer Agent ===")
    writer_input = {
        "query": query_text,
        "context": context,
        "intent": intention_output["intent"],
        "information": retriever_output["information"],
        "analysis": reason_output["analysis"]
    }
    print(f"Input: {json.dumps(writer_input, indent=2)}")
    writer_output = await writer_agent.process(writer_input)
    print(f"Output: {json.dumps(writer_output, indent=2)}\n")
    
    # Step 5: Designer Agent
    print("=== Step 5: Designer Agent ===")
    designer_input = {
        "query": query_text,
        "context": context,
        "intent": intention_output["intent"],
        "information": retriever_output["information"],
        "analysis": reason_output["analysis"],
        "response": writer_output["response"]
    }
    print(f"Input: {json.dumps(designer_input, indent=2)}")
    designer_output = await designer_agent.process(designer_input)
    print(f"Output: {json.dumps(designer_output, indent=2)}\n")

if __name__ == "__main__":
    asyncio.run(test_agent_flow())
