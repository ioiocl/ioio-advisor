import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import real agent implementations
from src.agents.intention_agent import Phi3IntentionAgent
from src.agents.retriever_agent import InstructorXLRetrieverAgent
from src.agents.reason_agent import Mistral7BReasonAgent
from src.agents.writer_agent import GPT4WriterAgent
from src.agents.designer_agent import StableDiffusionDesignerAgent

async def main():
    # Initialize real agents
    agents = {
        "intention": Phi3IntentionAgent(),
        "retriever": InstructorXLRetrieverAgent(),
        "reason": Mistral7BReasonAgent(),
        "writer": GPT4WriterAgent(),
        "designer": StableDiffusionDesignerAgent()
    }

    # Sample query and context (can be customized)
    query = "¿Cómo está el mercado de inversiones hoy y qué riesgos debo considerar?"
    context = {
        "query_id": "123",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_preferences": {
            "language": "es",
            "risk_profile": "moderate",
            "experience_level": "intermediate"
        }
    }

    print(f"\n=== Running Real Financial Agents Flow ===")
    print(f"Query: {query}")
    print(f"Context: {context}\n")

    # Run agents pipeline with real data
    input_data = {"query": query, "context": context}

    # Intention Agent
    intention_output = await agents["intention"].process(input_data)
    print(f"Intention Output: {intention_output}")

    # Retriever Agent
    retriever_input = {**input_data, **intention_output}
    retriever_output = await agents["retriever"].process(retriever_input)
    print(f"Retriever Output: {retriever_output}")

    # Reason Agent
    reason_input = {**retriever_input, **retriever_output}
    reason_output = await agents["reason"].process(reason_input)
    print(f"Reason Output: {reason_output}")

    # Writer Agent
    writer_input = {**reason_input, **reason_output}
    writer_output = await agents["writer"].process(writer_input)
    print(f"Writer Output: {writer_output}")

    # Designer Agent
    designer_input = {**writer_input, **writer_output}
    designer_output = await agents["designer"].process(designer_input)
    print(f"Designer Output: {designer_output}")

if __name__ == "__main__":
    asyncio.run(main())
