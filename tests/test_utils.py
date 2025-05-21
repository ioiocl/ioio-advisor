from datetime import datetime, UTC
from typing import Dict, Any, List
from PIL import Image, ImageDraw
import io

from src.ports.agent_port import (
    IntentionAgent,
    RetrieverAgent,
    ReasonAgent,
    WriterAgent,
    DesignerAgent
)
from src.domain.models import UserQuery
from src.infrastructure.storage.local_storage import LocalStorageService

class SimpleIntentionAgent(IntentionAgent):
    async def detect_intent(self, query: str) -> Dict[str, Any]:
        print("\n[IntentionAgent] Input:")
        print(f"Query: {query}")
        
        # Determine topic based on query keywords
        query = query.lower()
        main_topic = "general_finance"  # default
        confidence = 0.9
        
        # Currency keywords
        if any(word in query for word in ["dolar", "usd", "$", "tipo de cambio", "precio del dolar", "cambio"]):
            main_topic = "currency"
            confidence = 0.95
        # Interest rate keywords
        elif any(word in query for word in ["tasa", "interes", "prestamo", "credito", "hipoteca"]):
            main_topic = "interest"
            confidence = 0.95
        # Inflation keywords
        elif any(word in query for word in ["inflacion", "precios", "sube", "aumenta", "tasa de inflacion"]):
            main_topic = "inflation"
            confidence = 0.95
        # Investment keywords
        elif any(word in query for word in ["invertir", "inversion", "donde", "conviene", "recomendacion", "acciones", "bonos", "fondos"]):
            main_topic = "investment"
            confidence = 0.95
        # Budget keywords
        elif any(word in query for word in ["presupuesto", "gastos", "ingresos", "organizar"]):
            main_topic = "budget"
            confidence = 0.95
            
        result = {
            "main_topic": main_topic,
            "confidence": confidence,
            "query": query
        }
        print("[IntentionAgent] Output:")
        print(f"Intent: {result}")
        return result
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.detect_intent(input_data["query"])

class SimpleRetrieverAgent(RetrieverAgent):
    async def retrieve_information(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[RetrieverAgent] Input:")
        print(f"Query: {query}")
        print(f"Context: {context}")
        
        # Get intent from context
        intent = context.get("intent", {}).get("intent", "financial_advice")
        
        # Return context data based on intent
        result = {
            "inflation_analysis": {
                "inflation_rate": 4.5,
                "price_changes": {"food": 6.2, "housing": 3.8, "transport": 5.1},
                "historical_data": [3.2, 3.8, 4.1, 4.5]
            },
            "savings_advice": {
                "interest_rates": {"savings": 2.5, "fixed_term": 4.8},
                "currency_rates": {"USD": 1.0, "EUR": 0.85},
                "user_savings": 5000
            },
            "investment_advice": {
                "market_data": {"stocks": ["AAPL", "GOOGL", "MSFT"], "indices": ["S&P500"]},
                "risk_levels": {"stocks": "medium", "bonds": "low", "crypto": "high"},
                "returns": {"stocks": 8.5, "bonds": 4.2, "crypto": 15.0}
            },
            "loan_analysis": {
                "interest_rates": {"personal": 12.5, "mortgage": 6.8},
                "user_credit_score": 720,
                "debt_capacity": 15000
            },
            "budget_planning": {
                "income": 3000,
                "expenses": {"housing": 900, "food": 400, "transport": 200},
                "savings_potential": 500
            }
        }.get(intent, {"market_data": {"stocks": ["AAPL", "GOOGL"]}, "user_portfolio": {"balance": 10000}})
        
        print("[RetrieverAgent] Output:")
        print(f"Retrieved Data: {result}")
        return result
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Get required data from input
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        intent = input_data.get("intent", {})
        
        # Update context with intent
        context["intent"] = intent
        
        # Retrieve information
        result = await self.retrieve_information(query, context)
        
        return result

class SimpleReasonAgent(ReasonAgent):
    async def analyze(self, query: str, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        print("[ReasonAgent] Input:")
        print(f"Query: {query}")
        print(f"Context: {context}")
        
        # Get topic from intent
        topic = intent.get("main_topic", "general_finance")
        
        # Generate query_id if not present
        query_id = context.get("query_id", str(hash(f"{query}_{datetime.now(UTC).isoformat()}")))
        
        # Set random seed for consistent but varied responses
        import random
        seed = hash(query_id) % 100000
        random.seed(seed)
        
        # Topic-specific key factors
        key_factors = {
            "inflation": ["precio", "aumento", "ajuste", "costo", "inflacion"],
            "investment": ["acciones", "bonos", "fondos", "riesgo", "cartera", "balance"],
            "currency": ["dolar", "usd", "$", "tipo de cambio", "precio"],
            "interest": ["tasa", "interes", "anual", "mensual", "prestamo"],
            "budget": ["gasto", "ingreso", "necesidad", "ahorro"],
            "market": ["bolsa", "mercado", "acciones", "indice", "tendencia"],
            "general_finance": ["precio", "aumento", "ajuste"]
        }
        
        # Create analysis dictionary
        analysis = {
            "query_id": query_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "topic": topic,
            "key_factors": key_factors.get(topic, key_factors["general_finance"])
        }
        
        # Analyze query for specific topic detection
        query_lower = query.lower()
        if "bolsa" in query_lower or "mercado" in query_lower:
            analysis["topic"] = "market"
        elif "dolar" in query_lower or "tipo de cambio" in query_lower or "$" in query:
            analysis["topic"] = "currency"
        elif "tasa" in query_lower or "interes" in query_lower:
            analysis["topic"] = "interest"
        elif "inflacion" in query_lower or "precios" in query_lower:
            analysis["topic"] = "inflation"
        elif "inversion" in query_lower or "invertir" in query_lower or "acciones" in query_lower:
            analysis["topic"] = "investment"
        elif "presupuesto" in query_lower or "gastos" in query_lower:
            analysis["topic"] = "budget"
            
        # Add topic-specific analysis
        if topic == "inflation":
            analysis.update({
                "price_trends": ["alza", "estable", "baja"],
                "impact_areas": ["consumo", "ahorro", "inversiones"],
                "key_indicators": ["ipc", "canasta_basica", "salarios"]
            })
        elif topic == "investment":
            analysis.update({
                "risk_levels": ["bajo", "moderado", "alto"],
                "asset_types": ["acciones", "bonos", "fondos_mutuos"],
                "market_conditions": ["alcista", "bajista", "estable"]
            })
        elif topic == "currency":
            analysis.update({
                "exchange_rates": ["compra", "venta", "promedio"],
                "market_factors": ["oferta", "demanda", "politica_monetaria"],
                "impact_sectors": ["importaciones", "exportaciones", "turismo"]
            })
        
        # Add live market data based on topic
        live_data = {}
        
        if topic == "stock":
            live_data["stocks"] = {
                "AAPL": {
                    "price": "175.50",
                    "change_percent": "+1.25%"
                },
                "GOOGL": {
                    "price": "2850.75",
                    "change_percent": "+0.85%"
                },
                "MSFT": {
                    "price": "335.25",
                    "change_percent": "+0.95%"
                }
            }
            analysis["live_data"] = live_data
        
        elif topic == "currency":
            live_data["forex"] = {
                "rates": {
                    "USD/EUR": "0.92",
                    "USD/MXN": "17.50",
                    "EUR/GBP": "0.86"
                },
                "last_updated": datetime.now(UTC).isoformat()
            }
            analysis["live_data"] = live_data
        
        elif topic == "interest":
            live_data["interest_rates"] = {
                "mortgage": 6.75,
                "personal": 12.50,
                "auto": 7.25,
                "savings": 4.50,
                "last_updated": datetime.now(UTC).isoformat()
            }
            analysis["live_data"] = live_data
        
        elif topic == "inflation":
            live_data["inflation"] = {
                "current_rate": 4.2,
                "categories": {
                    "alimentos": 5.8,
                    "vivienda": 3.5,
                    "transporte": 4.0,
                    "salud": 2.8
                },
                "trend": "estable",
                "forecast": 4.0,
                "last_updated": datetime.now(UTC).isoformat()
            }
            analysis["live_data"] = live_data
        
        elif topic == "investment":
            live_data["investments"] = {
                "market_indices": {
                    "SP500": {
                        "value": "4,780.25",
                        "change": "+0.8%"
                    },
                    "NASDAQ": {
                        "value": "15,120.75",
                        "change": "+1.2%"
                    },
                    "DOW": {
                        "value": "35,950.50",
                        "change": "+0.5%"
                    }
                },
                "sector_performance": {
                    "tecnologia": {
                        "trend": "alcista",
                        "return_ytd": 15.5
                    },
                    "finanzas": {
                        "trend": "estable",
                        "return_ytd": 8.2
                    },
                    "salud": {
                        "trend": "moderado",
                        "return_ytd": 6.5
                    }
                },
                "last_updated": datetime.now(UTC).isoformat()
            }
        
        elif topic == "budget":
            live_data["budget"] = {
                "income_allocation": {
                    "necesidades": {
                        "percentage": 50,
                        "categories": ["vivienda", "alimentacion", "servicios"]
                    },
                    "deseos": {
                        "percentage": 30,
                        "categories": ["entretenimiento", "ropa", "viajes"]
                    },
                    "ahorro": {
                        "percentage": 20,
                        "categories": ["emergencias", "retiro", "inversiones"]
                    }
                },
                "expense_tracking": {
                    "actual_vs_planned": {
                        "necesidades": {
                            "planned": 5000,
                            "actual": 4800
                        },
                        "deseos": {
                            "planned": 3000,
                            "actual": 3200
                        },
                        "ahorro": {
                            "planned": 2000,
                            "actual": 1800
                        }
                    },
                    "last_updated": datetime.now(UTC).isoformat()
                }
            }
            analysis["live_data"] = live_data
        
        # Add recommendations based on topic and analysis
        if topic == "investment":
            analysis["recommendation"] = "Diversificar cartera considerando el perfil de riesgo y objetivos"
        elif topic == "currency":
            analysis["recommendation"] = "Monitorear tipos de cambio y considerar coberturas"
        elif topic == "interest":
            analysis["recommendation"] = "Comparar tasas entre instituciones y evaluar refinanciamiento"
        elif topic == "inflation":
            analysis["recommendation"] = "Ajustar presupuesto y buscar inversiones que superen la inflación"
        elif topic == "budget":
            analysis["recommendation"] = "Optimizar gastos y aumentar el porcentaje de ahorro"
        elif topic == "market":
            analysis["recommendation"] = "Mantener estrategia a largo plazo y evitar decisiones emocionales"
        else:
            analysis["recommendation"] = "Consultar con un asesor financiero para un análisis personalizado"
        
        return analysis       
        # Add live data if available for the topic
        if topic in live_data:
            analysis.update(live_data[topic])
            
        if topic == "inflation":
            analysis.update({
                "key_factors": ["precio", "aumento", "ajuste"] + analysis.get("key_factors", [])
            })
        elif topic == "investment":
            analysis.update({
                "risk_level": "moderate",
                "market_conditions": "stable",
                "investment_horizon": "medium-term",
                "portfolio_balance": {"renta_fija": 40, "renta_variable": 60},
                "key_factors": ["riesgo", "cartera", "balance", "diversificacion"]
            })
        elif topic == "savings":
            analysis.update({
                "recommended_options": ["cuenta_ahorro", "deposito_plazo", "fondo_inversion"],
                "risk_profile": "conservador",
                "diversification": {"moneda_local": 70, "moneda_extranjera": 30},
                "key_factors": ["moneda", "inversion", "diversificacion", "riesgo"]
            })
        elif topic == "loan":
            analysis.update({
                "current_rates": {"personal": 12.5, "hipotecario": 7.8, "vehiculo": 9.5},
                "requirements": ["ingreso_estable", "historial_crediticio", "garantias"],
                "payment_terms": [12, 24, 36, 48, 60],
                "key_factors": ["tasa", "cuota", "condicion", "plazo"]
            })
        elif topic == "budget":
            analysis.update({
                "income_allocation": {"necesidades": 50, "deseos": 30, "ahorro": 20}
            })
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and analyze."""
        # Get required data
        query = input_data.get("query", "")
        intent = input_data.get("intent", {})
        context = input_data.get("context", {})
        
        # Analyze query
        # Analyze the query
        result = await self.analyze(query, intent, context)
        
        # Ensure query_id is passed through
        result["query_id"] = context["query_id"]
        return result

class SimpleIntentionAgent(IntentionAgent):
    async def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect intent from query."""
        # Simulate intent detection
        intent = {
            "main_topic": "currency",
            "confidence": 0.95,
            "query": query
        }
        return intent
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and detect intent."""
        print("[IntentionAgent] Input:")
        print(f"Query: {input_data['query']}")
        
        # Detect intent
        intent = await self.detect_intent(input_data['query'])
        print(f"[IntentionAgent] Output:")
        print(f"Intent: {intent}")
        
        return {"intent": intent}

class SimpleRetrieverAgent(RetrieverAgent):
    async def retrieve_information(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve information based on query and context."""
        return {
            "market_data": {
                "stocks": ["AAPL", "GOOGL"]
            },
            "user_portfolio": {
                "balance": 10000
            }
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and retrieve information."""
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        
        # Retrieve information
        result = await self.retrieve_information(query, context)
        
        return {"information": result}

    async def compose_response(self, analysis: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """Compose a response based on analysis and context."""
        # Get required data
        topic = analysis.get("topic", "general")
        query_id = user_context.get("query_id", "default")
        original_query = user_context.get("original_query", "")
        live_data = analysis.get("live_data", {})

        if topic == "investment":
            market_data = live_data.get("investments", {})
            indices = market_data.get("market_indices", {})
            sectors = market_data.get("sector_performance", {})
            
            response = f"Estado actual del mercado:\n\n"
            response += "Índices principales:\n"
            for index, data in indices.items():
                response += f"- {index}: {data['value']} ({data['change']})\n"
            
            response += "\nDesempeño por sector:\n"
            for sector, data in sectors.items():
                response += f"- {sector.title()}: {data['trend'].title()} (YTD: {data['return_ytd']}%)\n"
        random.seed(int(query_id) % 100000 if query_id else None)
        
        # Use pre-generated text if available
        if "text" in analysis and analysis["text"]:
            return analysis["text"]
        
        # Get key factors for the topic
        key_factors = analysis.get("key_factors", [])
        
        # Generate response based on topic
        responses = {
            "inflation": [
                "La inflación actual está en niveles moderados. Los precios muestran una tendencia estable.",
                "Se observa un ligero aumento en la inflación. Es importante monitorear los precios."
            ],
            "investment": [
                "El mercado ofrece diversas oportunidades de inversión. Considera tu perfil de riesgo.",
                "Es recomendable diversificar tu portafolio entre diferentes tipos de activos."
            ],
            "interest": [
                "Las tasas de interés se mantienen en niveles competitivos.",
                "Es un buen momento para revisar las condiciones de créditos y préstamos."
            ],
            "currency": [
                "El tipo de cambio muestra estabilidad en el mercado actual.",
                "Las divisas principales mantienen una tendencia estable."
            ],
            "budget": [
                "Es importante mantener un presupuesto balanceado entre gastos e ingresos.",
                "Considera la regla 50/30/20 para organizar tus finanzas personales."
            ],
            "market": [
                "El mercado muestra señales positivas para los inversionistas.",
                "Se recomienda mantener una estrategia diversificada en el mercado actual."
            ],
            "general_finance": [
                "Es importante mantener una estrategia financiera balanceada.",
                "Consulta con un asesor financiero para recomendaciones personalizadas."
            ]
        }
        
        # Get responses for the topic
        topic_responses = responses.get(topic, responses["general_finance"])
        
        # Choose a random response using query_id as seed
        import random
        random.seed(int(query_id) % 100000 if query_id else None)
        response = random.choice(topic_responses)

        # Add recommendation if available
        recommendation = analysis.get("recommendation", "")
        if recommendation:
            response += f"\n\nRecomendación: {recommendation}"

        # Add query context
        if original_query:
            response = f"En respuesta a su consulta sobre {original_query}:\n\n{response}"

        return response
        """Write a response based on the reasoning and context."""
        # Get query_id from context
        query_id = context.get("query_id", str(hash(str(reasoning))))
        
        # Ensure unique responses by including query_id in the input
        reasoning["query_id"] = query_id
        
        response = "Here's a simple response about " + reasoning.get("topic", "finance")
        return response

class SimpleWriterAgent(WriterAgent):
    def __init__(self):
        self._used_responses = set()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and generate a response."""
        analysis = input_data.get("analysis", {})
        context = input_data.get("context", {})
        response = await self.compose_response(analysis, context)
        return {"response": response}

    async def compose_response(self, analysis: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """Compose a response based on analysis and context."""
        topic = analysis.get("topic", "general")
        sentiment = analysis.get("market_sentiment", "neutral")
        recommendation = analysis.get("recommendation", "")

        # Generate response based on topic and sentiment
        if topic == "investment":
            response = f"El mercado muestra señales {sentiment}es. "
        elif topic == "currency":
            response = "Los tipos de cambio se mantienen estables. "
        elif topic == "interest":
            response = "Las tasas de interés están en niveles competitivos. "
        else:
            response = "La situación financiera general es estable. "

        # Add recommendation if available
        if recommendation:
            response += f"\n\nRecomendación: {recommendation}"

        return response

class SimpleDesignerAgent(DesignerAgent):
    def __init__(self):
        self.storage = LocalStorageService()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and generate visualizations."""
        print("[SimpleDesignerAgent] Starting process...")
        
        # Get query_id from context
        query_id = input_data["context"].get("query_id", str(hash(input_data["response"])))
        print(f"[SimpleDesignerAgent] Query ID: {query_id}")
        
        try:
            visualization = await self.generate_visualization(
                input_data["context"],
                input_data["response"]
            )
            print(f"[SimpleDesignerAgent] Generated visualization: {visualization}")
            
            # Add query_id to visualization metadata
            if "metadata" not in visualization:
                visualization["metadata"] = {}
            visualization["metadata"]["query_id"] = query_id
            
            # Ensure unique visualization URLs
            if "visualization_url" in visualization:
                visualization["visualization_url"] = f"{visualization['visualization_url']}?id={query_id}"
            if "image_url" in visualization:
                visualization["image_url"] = f"{visualization['image_url']}?id={query_id}"
            
            print(f"[SimpleDesignerAgent] Final visualization data: {visualization}")
            return {"visualization": visualization}
            
        except Exception as e:
            import traceback
            print(f"[SimpleDesignerAgent] Error: {str(e)}")
            print(f"[SimpleDesignerAgent] Traceback: {traceback.format_exc()}")
            raise

    async def generate_visualization(self, context: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Generate both a data visualization and topic illustration."""
        print(f"[SimpleDesignerAgent] Generating visualization for text: {text[:100]}...")
        
        try:
            # Create a simple illustration
            img = Image.new('RGB', (400, 300), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((50, 150), text[:50], fill='black')
            
            # Save illustration with high quality
            image_output = io.BytesIO()
            img.save(image_output, format='PNG', quality=100)
            image_data = image_output.getvalue()
            
            # Save file to storage and get URL
            image_url = await self.storage.save_image(image_data, extension='png')
            
            # Return result with URLs and metadata
            result = {
                "visualization_url": None,
                "image_url": image_url,
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "version": "1.0"
                }
            }
            print(f"[SimpleDesignerAgent] Returning result: {result}")
            return result
            
        except Exception as e:
            import traceback
            print(f"[SimpleDesignerAgent] Error in generate_visualization: {str(e)}")
            print(f"[SimpleDesignerAgent] Traceback: {traceback.format_exc()}")
            raise
        print("[SimpleDesignerAgent] Starting process...")
        
        # Get query_id from context
        query_id = input_data["context"].get("query_id", str(hash(input_data["response"])))
        print(f"[SimpleDesignerAgent] Query ID: {query_id}")
        
        try:
            visualization = await self.generate_visualization(
                input_data["context"],
                input_data["response"]
            )
            print(f"[SimpleDesignerAgent] Generated visualization: {visualization}")
            
            # Add query_id to visualization metadata
            if "metadata" not in visualization:
                visualization["metadata"] = {}
            visualization["metadata"]["query_id"] = query_id
            
            # Ensure unique visualization URLs
            if "visualization_url" in visualization:
                visualization["visualization_url"] = f"{visualization['visualization_url']}?id={query_id}"
            if "image_url" in visualization:
                visualization["image_url"] = f"{visualization['image_url']}?id={query_id}"
            
            print(f"[SimpleDesignerAgent] Final visualization data: {visualization}")
            return visualization
            
        except Exception as e:
            import traceback
            print(f"[SimpleDesignerAgent] Error: {str(e)}")
            print(f"[SimpleDesignerAgent] Traceback: {traceback.format_exc()}")
            raise
