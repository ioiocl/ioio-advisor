from typing import Dict, Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import aiohttp
from datetime import datetime, timedelta
import json
import asyncio
from functools import lru_cache

from ..ports.agent_port import RetrieverAgent

class InstructorXLRetrieverAgent(RetrieverAgent):
    """Implementation of the retriever agent using Instructor XL with enhanced data retrieval."""
    
    def __init__(self):
        self.model_name = "hkunlp/instructor-xl"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # API endpoints for financial data
        self.api_endpoints = {
            "exchange_rate": "https://api.exchangerate-api.com/v4/latest/USD",
            "economic_indicators": "https://api.worldbank.org/v2/country/CL/indicator/",
            "news": "https://newsapi.org/v2/everything",
            "market_data": "https://api.marketdata.com/v1/",
            "stock_prices": "https://api.stockmarket.com/v2/quotes"
        }
        
        # Cache settings
        self.cache_ttl = {
            "exchange_rate": timedelta(hours=1),
            "economic_indicators": timedelta(days=1),
            "news": timedelta(minutes=30),
            "market_data": timedelta(minutes=15),
            "stock_prices": timedelta(minutes=5)
        }
        
        # Educational content templates
        self.educational_templates = {
            "currency": {
                "concepts": [
                    "tipo de cambio",
                    "mercado de divisas",
                    "oferta y demanda",
                    "inflación y moneda"
                ],
                "examples": [
                    "Cuando el dólar sube, los productos importados se vuelven más caros",
                    "Si el peso se devalúa, nuestros productos son más competitivos en el exterior"
                ],
                "tips": [
                    "Diversifica tus ahorros en diferentes monedas",
                    "Mantente informado sobre las tendencias del mercado cambiario"
                ]
            },
            "investment": {
                "concepts": ["riesgo y retorno", "diversificación", "horizonte temporal"],
                "examples": ["Invertir en acciones a largo plazo", "Crear un portafolio diversificado"],
                "tips": ["Investiga antes de invertir", "No pongas todos los huevos en la misma canasta"]
            }
        }
        
        # Initialize cache
        self._data_cache = {}
        self._last_update = {}
        
        # Market sentiment indicators
        self.sentiment_indicators = {
            "positive": ["crecimiento", "subida", "ganancia", "optimista", "mejora"],
            "negative": ["caída", "pérdida", "riesgo", "pesimista", "deterioro"],
            "neutral": ["estable", "mantiene", "equilibrio", "constante"]
        }
    
    async def retrieve_information(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve relevant financial information based on query and context."""
        intent = context.get("intent", {})
        main_topic = intent.get("main_topic", "general_finance")
        query_type = intent.get("query_type", "get_information")
        subtopics = intent.get("subtopics", [])
        user_level = context.get("user_level", "beginner")
        
        try:
            # Collect data from different sources with caching
            data = await self._gather_financial_data(main_topic, subtopics)
            
            # Add educational content based on user level and topic
            educational_content = self._get_educational_content(
                main_topic,
                subtopics,
                user_level
            )
            
            # Analyze market sentiment
            sentiment = self._analyze_market_sentiment(data)
            
            # Generate retrieval instruction with enhanced context
            instruction = self._generate_retrieval_instruction(
                query=query,
                topic=main_topic,
                query_type=query_type,
                user_level=user_level,
                sentiment=sentiment
            )
            
            # Process data with Instructor XL
            inputs = self.tokenizer(
                instruction,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
            
            processed_info = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Structure the response with enhanced information
            return {
                "information": {
                    "market_data": data.get("market_data", {}),
                    "educational_content": educational_content
                },
                "analysis": self._parse_processed_info(processed_info),
                "market_sentiment": sentiment,
                "metadata": {
                    "sources": list(data.keys()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "cache_status": self._get_cache_status(),
                    "data_freshness": self._calculate_data_freshness()
                }
            }
            
        except Exception as e:
            # Log the error and return a graceful fallback response
            print(f"Error in retrieve_information: {str(e)}")
            return {
                "information": {
                    "market_data": self._get_fallback_data(),
                    "educational_content": self._get_basic_educational_content(main_topic)
                },
                "analysis": {
                    "error": "Lo siento, hubo un problema al obtener los datos más recientes. "
                            "Te muestro información básica mientras resolvemos el inconveniente."
                },
                "market_sentiment": "neutral",
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "using_fallback": True
                }
            }
    
    def _generate_retrieval_instruction(
        self,
        query: str,
        topic: str,
        query_type: str,
        user_level: str,
        sentiment: str
    ) -> str:
        """Generate enhanced instruction for the model based on query context."""
        # Adjust language complexity based on user level
        complexity = "simple y práctica" if user_level == "beginner" else "detallada y técnica"
        
        # Adjust focus based on query type
        focus_map = {
            "get_information": "explicar conceptos y datos actuales",
            "get_recommendation": "proporcionar recomendaciones específicas",
            "understand_reason": "analizar causas y efectos",
            "get_analysis": "realizar un análisis profundo"
        }
        focus = focus_map.get(query_type, "explicar conceptos y datos actuales")
        
        return f"""Analiza la siguiente consulta financiera y proporciona una respuesta {complexity}:

Consulta: {query}
Tema: {topic}
Tipo: {query_type}
Sentimiento del mercado: {sentiment}

Instrucciones:
1. {focus}
2. Identifica métricas financieras clave y sus valores actuales
3. Extrae tendencias y patrones relevantes
4. Explica las implicaciones para el usuario

Formato de respuesta:
- Datos Clave
- Tendencias
- Implicaciones
- Recomendaciones
- Contexto Adicional

Respuesta:"""
    
    async def _gather_financial_data(self, topic: str, subtopics: List[str]) -> Dict[str, Any]:
        """Gather financial data from various sources with caching."""
        data = {}
        topics_to_fetch = {topic} | set(subtopics)
        
        async def fetch_with_cache(endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
            """Fetch data with caching support."""
            cache_key = f"{endpoint}:{json.dumps(params) if params else ''}"
            
            # Check cache first
            if cache_key in self._data_cache:
                last_update = self._last_update.get(cache_key)
                if last_update and datetime.utcnow() - last_update < self.cache_ttl.get(endpoint, timedelta(minutes=15)):
                    return self._data_cache[cache_key]
            
            try:
                async with aiohttp.ClientSession() as session:
                    url = self.api_endpoints[endpoint]
                    if params:
                        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Update cache
                            self._data_cache[cache_key] = data
                            self._last_update[cache_key] = datetime.utcnow()
                            return data
            except Exception as e:
                print(f"Error fetching {endpoint}: {str(e)}")
                return None
        
        # Gather all required data concurrently
        tasks = []
        
        # Exchange rates for currency topics
        if "currency" in topics_to_fetch:
            tasks.append(fetch_with_cache("exchange_rate"))
        
        # Market data for all queries
        tasks.append(fetch_with_cache("market_data", {"type": "summary"}))
        
        # Economic indicators based on topics
        indicators = {
            "inflation": "FP.CPI.TOTL.ZG",
            "gdp": "NY.GDP.MKTP.KD.ZG",
            "unemployment": "SL.UEM.TOTL.ZS"
        }
        for topic in topics_to_fetch:
            if topic in indicators:
                tasks.append(fetch_with_cache(
                    "economic_indicators",
                    {"indicator": indicators[topic]}
                ))
        
        # Stock prices for investment topics
        if "investment" in topics_to_fetch:
            tasks.append(fetch_with_cache("stock_prices", {"market": "main"}))
        
        # News in Spanish
        news_topics = ",".join(topics_to_fetch)
        tasks.append(fetch_with_cache(
            "news",
            {"q": news_topics, "language": "es", "sortBy": "publishedAt"}
        ))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, dict):
                data.update(result)
        
        return data
    
    def _get_educational_content(
        self,
        main_topic: str,
        subtopics: List[str],
        user_level: str
    ) -> Dict[str, Any]:
        """Get educational content based on topic and user level."""
        content = {}
        topics_to_cover = {main_topic} | set(subtopics)
        
        for topic in topics_to_cover:
            if topic in self.educational_templates:
                template = self.educational_templates[topic]
                
                # Adjust content based on user level
                if user_level == "beginner":
                    # For beginners, provide basic concepts and simple examples
                    content[topic] = {
                        "concepts": template["concepts"][:2],  # First 2 basic concepts
                        "examples": template["examples"][:1],  # 1 simple example
                        "tips": template["tips"][:2]  # 2 basic tips
                    }
                else:
                    # For advanced users, provide all content
                    content[topic] = template
        
        return content
    
    def _analyze_market_sentiment(self, data: Dict[str, Any]) -> str:
        """Analyze market sentiment from financial data."""
        sentiment_score = 0
        total_indicators = 0
        
        # Analyze market data
        market_data = data.get("market_data", {})
        
        # Check stock market indices
        for index, values in market_data.get("indices", {}).items():
            change = values.get("change", "0%")
            change_value = float(change.strip("%") or 0)
            sentiment_score += 1 if change_value > 0 else -1 if change_value < 0 else 0
            total_indicators += 1
        
        # Check interest rates
        rates = market_data.get("interest_rates", {})
        savings_rate = float(rates.get("savings", "0%").strip("%") or 0)
        sentiment_score += 1 if savings_rate > 4 else -1 if savings_rate < 2 else 0
        total_indicators += 1
        
        # Check inflation
        inflation = market_data.get("inflation", {})
        current_rate = float(inflation.get("current_rate", "0%").strip("%") or 0)
        sentiment_score += -1 if current_rate > 5 else 1 if current_rate < 3 else 0
        total_indicators += 1
        
        # Calculate final sentiment
        if total_indicators == 0:
            return "neutral"
        
        avg_sentiment = sentiment_score / total_indicators
        if avg_sentiment > 0.3:
            return "positivo"
        elif avg_sentiment < -0.3:
            return "negativo"
        return "neutral"
    
    def _get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status."""
        status = {}
        for endpoint in self.api_endpoints:
            cache_hits = sum(1 for key in self._data_cache if key.startswith(endpoint))
            status[endpoint] = {
                "cached_items": cache_hits,
                "last_update": self._last_update.get(endpoint, None)
            }
        return status
    
    def _calculate_data_freshness(self) -> Dict[str, str]:
        """Calculate the freshness of cached data."""
        freshness = {}
        now = datetime.utcnow()
        
        for endpoint, last_update in self._last_update.items():
            if last_update:
                age = now - last_update
                if age < timedelta(minutes=5):
                    freshness[endpoint] = "muy reciente"
                elif age < timedelta(hours=1):
                    freshness[endpoint] = "reciente"
                elif age < timedelta(days=1):
                    freshness[endpoint] = "hoy"
                else:
                    freshness[endpoint] = "desactualizado"
        
        return freshness
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Get basic fallback data when API calls fail."""
        return {
            "indices": {
                "S&P 500": {"value": "4,850", "change": "+1.2%"},
                "NASDAQ": {"value": "15,200", "change": "+0.8%"}
            },
            "inflation": {
                "current_rate": "3.2%",
                "food": "4.1%",
                "housing": "3.8%",
                "transportation": "2.9%"
            },
            "interest_rates": {
                "savings": "4.5%",
                "mortgage": "6.8%",
                "credit_card": "18.9%"
            }
        }
    
    def _get_basic_educational_content(self, topic: str) -> Dict[str, Any]:
        """Get basic educational content when full content is unavailable."""
        return {
            "concepts": ["conceptos básicos de finanzas"],
            "examples": ["ejemplo simple de gestión financiera"],
            "tips": ["mantente informado sobre el mercado"]
        }
    
    def _parse_processed_info(self, info: str) -> Dict[str, Any]:
        """Parse the model's output into structured information."""
        sections = ["Datos Clave", "Tendencias", "Implicaciones", "Recomendaciones", "Contexto Adicional"]
        result = {}
        
        current_section = None
        current_content = []
        
        for line in info.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line in sections:
                if current_section:
                    result[current_section.lower().replace(" ", "_")] = current_content
                current_section = line
                current_content = []
            elif current_section:
                current_content.append(line)
        
        if current_section:
            result[current_section.lower().replace(" ", "_")] = current_content
        
        return result
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the retriever agent."""
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        return await self.retrieve_information(query, context)
