import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime, UTC
from pprint import pprint
import sys

from src.agents.intention_agent import Phi3IntentionAgent
from src.agents.retriever_agent import DataRetrieverAgent
from src.agents.reason_agent import Mistral7BReasonAgent
from src.agents.writer_agent import GPT4WriterAgent
from src.agents.designer_agent import DesignerAgent
from src.utils.logging import setup_logging
from src.utils.error_handling import handle_agent_error

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Debug configuration
DEBUG = True
logger = setup_logging(__name__)

def debug_print(label: str, data: Any) -> None:
    """Print debug information if DEBUG is True"""
    if DEBUG:
        logger.debug(f"=== {label} ===\n{json.dumps(data, indent=2, ensure_ascii=False)}\n{'=' * 40}\n")

# Educational content templates
EDUCATIONAL_CONTENT = {
    "inflation": {
        "concepts": [
            "La inflación es el aumento general de precios en una economía a lo largo del tiempo. Cuando hay inflación, cada unidad de moneda compra menos bienes y servicios."
        ],
        "examples": [
            "Por ejemplo, si una barra de pan costaba $1 el año pasado y ahora cuesta $1.05, eso representa una inflación del 5% en el precio del pan."
        ],
        "tips": [
            "Invierte en activos que típicamente aumentan con la inflación",
            "Busca aumentos salariales que al menos igualen la tasa de inflación"
        ]
    },
    "investment": {
        "concepts": [
            "El mercado de valores es como una gran tienda donde se compran y venden partes (acciones) de empresas",
            "Diversificar significa 'no poner todos los huevos en la misma canasta'",
            "El riesgo y la ganancia van de la mano: mayor posible ganancia significa mayor riesgo"
        ],
        "examples": [
            "Por ejemplo, si inviertes $100 en acciones de una empresa de helados, eres dueño de una pequeña parte de esa empresa",
            "Si diversificas $100 entre una heladería, una juguetería y una librería, reduces el riesgo de perder todo tu dinero"
        ],
        "tips": [
            "Comienza con pequeñas cantidades mientras aprendes",
            "Investiga bien antes de invertir",
            "Piensa en el largo plazo"
        ]
    },
    "inflation": {
        "concepts": [
            "La inflación es como un globo que se infla: los precios suben y el dinero vale menos",
            "Cuando hay inflación, necesitas más dinero para comprar lo mismo que antes",
            "Los sueldos deberían subir con la inflación para mantener el poder de compra"
        ],
        "examples": [
            "Si antes un pan costaba $1 y ahora cuesta $1.20, eso es inflación del 20%",
            "Si ahorras $100 bajo el colchón y hay 10% de inflación, en un año podrás comprar lo que hoy cuesta $90"
        ],
        "tips": [
            "Compara precios entre tiendas",
            "Planifica tus compras",
            "Busca inversiones que protejan tu dinero de la inflación"
        ]
    },
    "interest": {
        "concepts": [
            "El interés es el costo de pedir dinero prestado o la ganancia por prestarlo",
            "El interés compuesto es cuando ganas interés sobre el interés anterior",
            "La tasa de interés afecta a préstamos, hipotecas y ahorros"
        ],
        "examples": [
            "Si pides prestados $1000 al 10% anual, deberás pagar $100 extra en un año",
            "$100 ahorrados al 5% anual se convierten en $105 el primer año, y luego en $110.25 el segundo año"
        ],
        "tips": [
            "Compara tasas entre diferentes bancos",
            "Lee la letra pequeña",
            "Paga más del mínimo en las tarjetas de crédito"
        ]
    },
    "currency": {
        "concepts": [
            "El tipo de cambio es como el precio de una moneda en términos de otra",
            "Las monedas suben y bajan de valor según la economía de cada país",
            "Un dólar fuerte significa que necesitas más pesos para comprarlo"
        ],
        "examples": [
            "Si el dólar sube de 20 a 22 pesos, los productos importados serán más caros",
            "Si viajas a otro país, el tipo de cambio afectará cuánto puedes comprar allá"
        ],
        "tips": [
            "Compara tasas de cambio en diferentes lugares",
            "Evita cambiar dinero en aeropuertos",
            "Considera el tipo de cambio al planificar viajes o compras internacionales"
        ]
    },
    "budget": {
        "concepts": [
            "Un presupuesto es como un mapa que te ayuda a manejar tu dinero",
            "La regla 50/30/20: 50% necesidades, 30% deseos, 20% ahorro",
            "Ingresos menos gastos debe ser positivo para poder ahorrar"
        ],
        "examples": [
            "Si ganas $1000, deberías gastar máximo $500 en necesidades básicas",
            "Ahorrar $100 al mes durante un año te da $1200 más intereses"
        ],
        "tips": [
            "Anota todos tus gastos",
            "Separa dinero para emergencias",
            "Automatiza tus ahorros"
        ]
    }
}

def debug_print(title: str, data: Any) -> None:
    if DEBUG:
        print(f"\n=== DEBUG: {title} ===")
class IntentionAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        query_lower = query.lower()
        
        # Determine topic and subtopics based on query keywords
        main_topic = "general_finance"
        subtopics = []
        confidence = 0.7
        
        # Define topic patterns
        topic_patterns = {
            "investment": {
                "keywords": ["mercado", "inversion", "invertir", "acciones", "bolsa", "rendimiento"],
                "phrases": ["mercado de valores", "bolsa de valores"]
            },
            "currency": {
                "keywords": ["dolar", "moneda", "peso", "divisa", "tipo de cambio"],
                "phrases": ["sube el dolar", "baja el dolar", "sube dolar", "baja dolar"]
            },
            "interest": {
                "keywords": ["tasa", "interes", "prestamo", "credito", "hipoteca"],
                "phrases": ["tasa de interes", "tasa hipotecaria"]
            },
            "inflation": {
                "keywords": ["inflacion", "precios", "costo"],
                "phrases": ["suben los precios", "aumenta el costo", "sube precio", "aumenta precio"]
            },
            "budget": {
                "keywords": ["presupuesto", "gastos", "ahorros", "sueldo", "ingresos"],
                "phrases": ["manejo de dinero", "control de gastos"]
            }
        }
        
        # Check phrases first
        topic_found = False
        query_words = query_lower.split()
        
        # First check for phrases (they are more specific)
        for topic, patterns in topic_patterns.items():
            if any(all(word in query_lower for word in phrase.split()) for phrase in patterns["phrases"]):
                main_topic = topic
                confidence = 0.95
                topic_found = True
                break
        
        # Then check individual words if no phrase was found
        if not topic_found:
            for topic, patterns in topic_patterns.items():
                if any(keyword in query_words for keyword in patterns["keywords"]):
                    main_topic = topic
                    confidence = 0.95
                    topic_found = True
                    break
        
        # Set subtopics based on detected topic
        if topic_found:
            if main_topic == "currency":
                subtopics = ["exchange_rate", "international_trade"]
            elif main_topic == "inflation":
                subtopics = ["price_increase", "purchasing_power"]
            elif main_topic == "interest":
                subtopics = ["loans", "savings"]
            elif main_topic == "investment":
                subtopics = ["risk", "diversification"]
            elif main_topic == "budget":
                subtopics = ["savings", "expenses"]
            
        result = {
            "intent": {
                "main_topic": main_topic,
                "subtopics": subtopics,
                "confidence": confidence,
                "query_type": "educational" if "que es" in query_lower or "como funciona" in query_lower else "informative"
            }
        }
        
        return result

class RetrieverAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        debug_print("RetrieverAgent Input", input_data)
        
        # Get educational content based on topic
        intent = input_data.get("intent", {})
        topic = intent.get("main_topic", "general_finance")
        
        # Get relevant educational content
        content = EDUCATIONAL_CONTENT.get(topic, {})
        
        # Get market data (simulated)
        market_data = {
            "stocks": {
                "AAPL": {"price": "180.5", "change": "+1.2%"},
                "GOOGL": {"price": "2850.0", "change": "+0.8%"}
            },
            "indices": {
                "S&P 500": {"value": "4,850", "change": "+1.2%"},
                "NASDAQ": {"value": "15,200", "change": "+0.8%"}
            },
            "interest_rates": {
                "savings": "4.5%",
                "mortgage": "6.8%",
                "credit_card": "18.9%"
            },
            "inflation": {
                "current_rate": "3.2%",
                "food": "4.1%",
                "housing": "3.8%",
                "transportation": "2.9%"
            }
        }
        
        result = {
            "information": {
                "educational_content": content,
                "market_data": market_data
            }
        }
        
        debug_print("RetrieverAgent Output", result)
        return result

class ReasonAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        debug_print("ReasonAgent Input", input_data)
        
        intent = input_data.get("intent", {})
        info = input_data.get("information", {})
        
        topic = intent.get("main_topic", "general_finance")
        query_type = intent.get("query_type", "informative")
        
        # Get educational content and market data
        content = info.get("educational_content", {})
        market_data = info.get("market_data", {})
        
        # Analyze market conditions
        market_sentiment = "positivo"
        if market_data:
            indices = market_data.get("indices", {})
            changes = [float(val["change"].strip("%")) for val in indices.values()]
            avg_change = sum(changes) / len(changes) if changes else 0
            market_sentiment = "positivo" if avg_change > 0 else "negativo" if avg_change < 0 else "estable"
        
        # Generate analysis based on topic and query type
        analysis = {
            "topic": topic,
            "query_type": query_type,
            "market_sentiment": market_sentiment,
            "educational_aspects": {
                "concepts": content.get("concepts", []),
                "examples": content.get("examples", []),
                "tips": content.get("tips", [])
            },
            "market_data": market_data
        }
        
        # Add topic-specific recommendations
        if topic == "investment":
            analysis["recommendation"] = "Considera diversificar tu portafolio y empezar con pequeñas inversiones mientras aprendes."
        elif topic == "interest":
            analysis["recommendation"] = "Compara diferentes opciones de préstamos y lee cuidadosamente los términos y condiciones."
        elif topic == "inflation":
            analysis["recommendation"] = "Protege tus ahorros buscando inversiones que superen la tasa de inflación."
        elif topic == "currency":
            analysis["recommendation"] = "Mantén un ojo en las tendencias del tipo de cambio y planifica tus compras internacionales."
        elif topic == "budget":
            analysis["recommendation"] = "Aplica la regla 50/30/20 para organizar tus finanzas: 50% necesidades, 30% deseos, 20% ahorro."
        
        debug_print("ReasonAgent Output", {"analysis": analysis})
        return {"analysis": analysis}

class WriterAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        debug_print("WriterAgent Input", input_data)
        
        analysis = input_data.get("analysis", {})
        topic = analysis.get("topic", "general_finance")
        query_type = analysis.get("query_type", "informative")
        educational = analysis.get("educational_aspects", {})
        
        # Build response
        response = "¡Hola! 😊 Te ayudo a entender esto de manera simple:\n\n"
        
        # Add educational content if it's an educational query
        if query_type == "educational":
            # Add main concept
            concepts = educational.get("concepts", [])
            if concepts:
                response += "📚 Concepto básico:\n"
                response += f"{concepts[0]}\n\n"
            
            # Add example
            examples = educational.get("examples", [])
            if examples:
                response += "🌟 Un ejemplo sencillo:\n"
                response += f"{examples[0]}\n\n"
        
        # Add current market situation
        market_data = analysis.get("market_data", {})
        if topic == "investment":
            indices = market_data.get("indices", {})
            response += "📊 Situación actual del mercado:\n"
            for name, data in indices.items():
                response += f"- {name}: {data['value']} ({data['change']})\n"
            response += "\n"
        elif topic == "interest":
            rates = market_data.get("interest_rates", {})
            response += "💰 Tasas actuales:\n"
            for product, rate in rates.items():
                response += f"- {product.title()}: {rate}\n"
            response += "\n"
        elif topic == "inflation":
            inflation = market_data.get("inflation", {})
            response += f"📈 Inflación actual: {inflation.get('current_rate', 'N/A')}\n\n"
        
        # Add recommendation
        recommendation = analysis.get("recommendation", "")
        if recommendation:
            response += "💡 Recomendación personal:\n"
            response += f"{recommendation}\n\n"
        
        # Add practical tips
        tips = educational.get("tips", [])
        if tips:
            response += "✅ Tips prácticos:\n"
            for tip in tips[:2]:  # Show only top 2 tips
                response += f"• {tip}\n"
        
        result = {"response": response}
        debug_print("WriterAgent Output", result)
        return result

class DesignerAgent:
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        debug_print("DesignerAgent Input", input_data)
        
        # In a real implementation, this would generate actual visualizations
        # For now, we'll return a mock URL based on the topic
        visualization_type = "market_trends.png"
        if "inflation" in str(input_data):
            visualization_type = "inflation_chart.png"
        elif "interest" in str(input_data):
            visualization_type = "interest_rates.png"
        elif "budget" in str(input_data):
            visualization_type = "budget_pie.png"
        
        result = {
            "visualization_url": visualization_type,
            "created_at": datetime.now(UTC)
        }
        
        debug_print("DesignerAgent Output", result)
        return result

async def process_query(query: str, user_level: str = "beginner") -> Dict[str, Any]:
    """Process a single query through the agent pipeline"""
    try:
        # Initialize agents
        agents = {
            "intention": Phi3IntentionAgent(),
            "retriever": DataRetrieverAgent(),
            "reason": Mistral7BReasonAgent(),
            "writer": GPT4WriterAgent(),
            "designer": DesignerAgent()
        }

        # Initialize context
        context = {
            "query_id": str(hash(query)),
            "timestamp": datetime.now(UTC).isoformat(),
            "user_level": user_level,
            "language": "es",
            "user_profile": {
                "experience": user_level,
                "previous_queries": []
            }
        }

        # Initialize pipeline data
        pipeline_data = {"query": query, "context": context}
        debug_print("Initial Input", pipeline_data)

        # Run intention detection
        try:
            intention_output = await agents["intention"].process(pipeline_data)
            debug_print("Intention Output", intention_output)
            pipeline_data.update(intention_output)
        except Exception as e:
            logger.error(f"Error in intention agent: {str(e)}")
            return handle_agent_error("intention", str(e))

        # Run data retrieval
        try:
            retriever_output = await agents["retriever"].process(pipeline_data)
            debug_print("Retriever Output", retriever_output)
            pipeline_data.update(retriever_output)
        except Exception as e:
            logger.error(f"Error in retriever agent: {str(e)}")
            return handle_agent_error("retriever", str(e))

        # Run reasoning
        try:
            reason_output = await agents["reason"].process(pipeline_data)
            debug_print("Reason Output", reason_output)
            pipeline_data.update(reason_output)
        except Exception as e:
            logger.error(f"Error in reason agent: {str(e)}")
            return handle_agent_error("reason", str(e))

        # Generate response
        try:
            writer_output = await agents["writer"].process(pipeline_data)
            debug_print("Writer Output", writer_output)
            pipeline_data.update(writer_output)
        except Exception as e:
            logger.error(f"Error in writer agent: {str(e)}")
            return handle_agent_error("writer", str(e))

        # Generate visualization
        try:
            designer_output = await agents["designer"].process(pipeline_data)
            debug_print("Designer Output", designer_output)
            pipeline_data.update(designer_output)
        except Exception as e:
            logger.error(f"Error in designer agent: {str(e)}")
            # Don't fail if visualization fails, just log it
            logger.warning(f"Visualization generation failed: {str(e)}")

        return pipeline_data

    except Exception as e:
        logger.error(f"Unexpected error in agent pipeline: {str(e)}")
        return {
            "error": "Error general en el procesamiento",
            "details": str(e),
            "fallback_response": "Lo siento, ha ocurrido un error inesperado. Por favor, intenta tu consulta nuevamente."
        }

async def main():
    """Main function to run the financial assistant"""
    # Test queries
    queries = [
        "¿Qué es la inflación y cómo me afecta?",
        "¿Cómo funciona el mercado de acciones para principiantes?",
        "¿Cuál es la mejor manera de hacer un presupuesto mensual?",
        "¿Cómo me afecta la subida de las tasas de interés?",
        "¿Por qué sube el dólar y qué significa para mi bolsillo?"
    ]

    # Process each query
    for query in queries:
        print(f"\n{'='*80}\n")
        print(f"PROCESANDO CONSULTA: {query}\n")
        print(f"{'='*80}\n")

        result = await process_query(query)

        print("\nRESPUESTA FINAL:")
        print("-" * 40)

        if "error" in result:
            print(f"Error: {result['error']}")
            if "fallback_response" in result:
                print(f"\nRespuesta alternativa: {result['fallback_response']}")
        else:
            print(result["response"])
            if "visualization_url" in result:
                print(f"\nVisualización generada: {result['visualization_url']}")

        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
