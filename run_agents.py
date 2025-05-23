import asyncio
from typing import Dict, Any
from datetime import datetime, UTC

# Simple agent implementations
class IntentionAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        print(f"\n[IntentionAgent] Processing query: {query}")
        return {"intent": {"main_topic": "investment", "confidence": 0.95}}

class RetrieverAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[RetrieverAgent] Retrieving market data...")
        return {
            "information": {
                "market_data": {
                    "stocks": ["AAPL", "GOOGL"],
                    "indices": {"S&P 500": "+1.2%", "NASDAQ": "+0.8%"}
                }
            }
        }

class ReasonAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[ReasonAgent] Analyzing market conditions...")
        return {
            "analysis": {
                "market_sentiment": "positive",
                "recommendation": "Consider diversifying portfolio"
            }
        }

class WriterAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        analysis = input_data.get("analysis", {})
        print(f"\n[WriterAgent] Generating response based on analysis: {analysis}")
        return {
            "response": "El mercado muestra señales positivas. Se recomienda diversificar el portafolio."
        }

class DesignerAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[DesignerAgent] Creating market visualization...")
        return {
            "visualization_url": "market_trends.png",
            "created_at": datetime.now(UTC)
        }

async def main():
    # Initialize agents
    agents = {
        "intention": IntentionAgent(),
        "retriever": RetrieverAgent(),
        "reason": ReasonAgent(),
        "writer": WriterAgent(),
        "designer": DesignerAgent()
    }

    # Sample query
    query = "¿Cómo está el mercado de inversiones hoy?"
    context = {"query_id": "123", "timestamp": datetime.now(UTC)}

    print(f"\n=== Running Financial Agents Flow ===")
    print(f"Query: {query}")
    print(f"Context: {context}\n")

    # Run each agent
    input_data = {"query": query, "context": context}
    
    # Intention Agent
    intention_output = await agents["intention"].process(input_data)
    print(f"Intention Output: {intention_output}")

    # Retriever Agent
    retriever_output = await agents["retriever"].process(input_data)
    print(f"Retriever Output: {retriever_output}")

    # Reason Agent
    reason_input = {**input_data, **intention_output, **retriever_output}
    reason_output = await agents["reason"].process(reason_input)
    print(f"Reason Output: {reason_output}")

    # Writer Agent
    writer_input = {**input_data, **reason_output}
    writer_output = await agents["writer"].process(writer_input)
    print(f"Writer Output: {writer_output}")

    # Designer Agent
    designer_input = {**input_data, **writer_output}
    designer_output = await agents["designer"].process(designer_input)
    print(f"Designer Output: {designer_output}")

if __name__ == "__main__":
    asyncio.run(main())