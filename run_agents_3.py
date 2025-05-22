import asyncio
from typing import Dict, Any
from datetime import datetime, UTC
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class IntentionAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        print(f"\n[IntentionAgent] Processing query: {query}")
        
        # Enhanced topic detection with Spanish support
        topics = ["inversión", "mercado", "bolsa", "moneda", "economía"]
        detected_topic = next((topic for topic in topics if topic in query.lower()), "general")
        
        return {
            "intent": {
                "main_topic": detected_topic,
                "confidence": 0.95,
                "query_type": "market_analysis",
                "language": "es"
            }
        }

class RetrieverAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[RetrieverAgent] Retrieving market data...")
        
        # Enhanced market data with Spanish context
        return {
            "information": {
                "market_data": {
                    "indices": {
                        "S&P 500": {"valor": "+1.2%", "tendencia": "alcista"},
                        "NASDAQ": {"valor": "+0.8%", "tendencia": "estable"}
                    },
                    "monedas": {
                        "USD/EUR": "1.08",
                        "USD/ARS": "850.5"
                    },
                    "indicadores": {
                        "inflacion_anual": "4.2%",
                        "tasa_interes": "5.25%"
                    }
                },
                "cache_status": "fresh",
                "data_timestamp": datetime.now(UTC).isoformat()
            }
        }

class ReasonAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[ReasonAgent] Analyzing market conditions...")
        
        # Enhanced analysis with risk assessment
        market_data = input_data.get("information", {}).get("market_data", {})
        
        return {
            "analysis": {
                "market_sentiment": "positivo",
                "risk_level": "moderado",
                "factors": {
                    "tendencia_mercado": "alcista",
                    "volatilidad": "baja",
                    "riesgo_economico": "moderado"
                },
                "recommendations": [
                    "Diversificar el portafolio",
                    "Mantener exposición a tecnología",
                    "Considerar cobertura en USD"
                ]
            }
        }

class WriterAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        analysis = input_data.get("analysis", {})
        print(f"\n[WriterAgent] Generating response based on analysis: {analysis}")
        
        # Enhanced response generation with educational content
        sentiment = analysis.get("market_sentiment", "neutral")
        recommendations = analysis.get("recommendations", [])
        
        response = f"El mercado muestra una tendencia {sentiment}. "
        response += "Principales recomendaciones:\n"
        response += "\n".join(f"- {rec}" for rec in recommendations)
        
        return {
            "response": response,
            "content_type": "market_analysis",
            "educational_level": "intermediate"
        }

class DesignerAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[DesignerAgent] Creating market visualization...")
        
        # Enhanced visualization with Spanish labels
        return {
            "visualization": {
                "type": "market_dashboard",
                "elements": [
                    "tendencias_mercado",
                    "indicadores_economicos",
                    "recomendaciones"
                ],
                "language": "es",
                "created_at": datetime.now(UTC)
            }
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
    query = "¿Cómo está el mercado de inversiones hoy y qué riesgos debo considerar?"
    context = {
        "query_id": "123",
        "timestamp": datetime.now(UTC),
        "user_preferences": {
            "language": "es",
            "risk_profile": "moderate",
            "experience_level": "intermediate"
        }
    }

    print(f"\n=== Running Enhanced Financial Agents Flow ===")
    print(f"Query: {query}")
    print(f"Context: {context}\n")

    # Run agents pipeline
    input_data = {"query": query, "context": context}
    
    # Sequential processing with enhanced data flow
    intention_output = await agents["intention"].process(input_data)
    print(f"Intention Output: {intention_output}")

    retriever_input = {**input_data, **intention_output}
    retriever_output = await agents["retriever"].process(retriever_input)
    print(f"Retriever Output: {retriever_output}")

    reason_input = {**retriever_input, **retriever_output}
    reason_output = await agents["reason"].process(reason_input)
    print(f"Reason Output: {reason_output}")

    writer_input = {**reason_input, **reason_output}
    writer_output = await agents["writer"].process(writer_input)
    print(f"Writer Output: {writer_output}")

    designer_input = {**writer_input, **writer_output}
    designer_output = await agents["designer"].process(designer_input)
    print(f"Designer Output: {designer_output}")

if __name__ == "__main__":
    asyncio.run(main())
