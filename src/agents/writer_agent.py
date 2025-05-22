from typing import Dict, Any, List, Optional
import os
import json
import aiohttp
from datetime import datetime
from openai import AsyncOpenAI
from ..ports.agent_port import WriterAgent
from enum import Enum

class UserLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class ContentStyle(Enum):
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    EDUCATIONAL = "educational"

class GPT4WriterAgent(WriterAgent):
    """Implementation of the writer agent using GPT-4."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-turbo-preview"
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "demo")
        self.exchangerate_key = os.getenv("EXCHANGERATE_API_KEY", "demo")
        
        # Risk level thresholds
        self.risk_thresholds = {
            "low": 0.3,
            "moderate": 0.6,
            "high": 0.8
        }
        
        # Templates for different response styles
        self.templates = {
            "currency_impact": """
Explica de manera {style} cómo el dólar afecta la economía personal:

{key_points}

Implicaciones principales:
{implications}

Recomendaciones personalizadas para nivel {level}:
{recommendations}

Resumen de riesgo:
{risk_summary}

Contexto de mercado:
{market_context}
""",
            "inflation": """
Análisis {style} de la situación inflacionaria:

{key_points}

Impacto en tu economía (Nivel {level}):
{implications}

Acciones recomendadas:
{recommendations}

Indicadores clave:
{indicators}

Tendencias importantes:
{trends}
""",
            "default": """
Análisis financiero {style}:

{key_points}

Implicaciones para tu nivel ({level}):
{implications}

Recomendaciones personalizadas:
{recommendations}

Contexto de mercado:
{market_context}

Próximos pasos sugeridos:
{next_steps}
"""
        }
    
    async def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock data from Alpha Vantage."""
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alpha_vantage_key}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    
    async def get_forex_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Get real-time forex data from ExchangeRate API."""
        url = f"https://v6.exchangerate-api.com/v6/{self.exchangerate_key}/pair/{from_currency}/{to_currency}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    
    async def get_interest_rates(self) -> Dict[str, Any]:
        """Get current interest rates from a reliable source."""
        # For demo purposes, using mock data
        return {
            "mortgage": 7.25,
            "personal": 12.5,
            "auto": 9.75
        }
    
    async def get_inflation_data(self) -> Dict[str, Any]:
        """Get current inflation data."""
        # For demo purposes, using mock data
        return {
            "rate": 4.2,
            "categories": {
                "alimentos": 5.1,
                "vivienda": 3.8,
                "transporte": 4.5
            }
        }
    
    async def enrich_with_live_data(self, analysis: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich the analysis with real-time financial data."""
        intent = user_context.get("intent", {}).get("main_topic", "")
        
        if "stock" in intent or "market" in intent:
            tech_stocks = ["AAPL", "GOOGL", "MSFT"]
            stock_data = {}
            for symbol in tech_stocks:
                stock_data[symbol] = await self.get_stock_data(symbol)
            analysis["live_data"] = {"stocks": stock_data}
            
        elif "currency" in intent or "dollar" in intent:
            forex_data = await self.get_forex_rate("USD", "EUR")
            analysis["live_data"] = {"forex": forex_data}
            
        elif "interest" in intent:
            rates = await self.get_interest_rates()
            analysis["live_data"] = {"interest_rates": rates}
            
        elif "inflation" in intent:
            inflation = await self.get_inflation_data()
            analysis["live_data"] = {"inflation": inflation}
            
        return analysis

    def _determine_user_level(self, user_context: Dict[str, Any]) -> UserLevel:
        """Determine user's financial knowledge level."""
        # Extract relevant factors
        interaction_history = user_context.get("interaction_history", [])
        query_complexity = user_context.get("query_complexity", "low")
        explicit_level = user_context.get("user_level")
        
        if explicit_level:
            return UserLevel(explicit_level)
        
        # Default to beginner if no history
        if not interaction_history:
            return UserLevel.BEGINNER
            
        # Analyze query complexity and history
        if query_complexity == "high" and len(interaction_history) > 5:
            return UserLevel.ADVANCED
        elif query_complexity == "medium" or len(interaction_history) > 2:
            return UserLevel.INTERMEDIATE
        
        return UserLevel.BEGINNER
    
    def _determine_content_style(self, user_context: Dict[str, Any]) -> ContentStyle:
        """Determine appropriate content style based on context."""
        intent = user_context.get("intent", {})
        topic = intent.get("main_topic", "")
        
        if "learn" in topic or "explain" in topic:
            return ContentStyle.EDUCATIONAL
        elif "analysis" in topic or "professional" in topic:
            return ContentStyle.FORMAL
        
        return ContentStyle.CONVERSATIONAL
    
    def _generate_risk_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a risk summary based on analysis data."""
        risk_score = analysis.get("risk_score", 0.5)
        risk_factors = analysis.get("risk_factors", [])
        
        if risk_score <= self.risk_thresholds["low"]:
            risk_level = "bajo"
        elif risk_score <= self.risk_thresholds["moderate"]:
            risk_level = "moderado"
        else:
            risk_level = "alto"
            
        summary = f"Nivel de riesgo: {risk_level}\n"
        if risk_factors:
            summary += "Factores principales:\n"
            summary += "\n".join(f"• {factor}" for factor in risk_factors[:3])
        
        return summary
    
    async def compose_response(
        self,
        analysis: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """Compose a clear and concise response using GPT-4."""
        
        # Enrich analysis with live data
        analysis = await self.enrich_with_live_data(analysis, user_context)
        
        try:
            # Determine user level and content style
            user_level = self._determine_user_level(user_context)
            content_style = self._determine_content_style(user_context)
            
            # Prepare the content for the prompt
            key_findings = analysis.get("key_findings", [])
            implications = analysis.get("short_term_implications", [])
            recommendations = analysis.get("recommended_actions", [])
            live_data = analysis.get("live_data", {})
            topic = user_context.get("intent", {}).get("main_topic", "default")
            
            # Generate additional context
            risk_summary = self._generate_risk_summary(analysis)
            market_context = self._generate_market_context(live_data)
            
            # Filter and personalize recommendations based on user level
            recommendations = self._personalize_recommendations(recommendations, user_level)
        except Exception as e:
            return self._generate_fallback_response(analysis, user_context, f"Error al preparar el contenido: {str(e)}")
        
        try:
            # Add live data to key findings
            if live_data:
                if "stocks" in live_data:
                    for symbol, data in live_data["stocks"].items():
                        if "Global Quote" in data:
                            price = data["Global Quote"].get("05. price", "N/A")
                            change = data["Global Quote"].get("10. change percent", "N/A")
                            key_findings.append(f"{symbol}: ${price} ({change})")
                
                if "forex" in live_data:
                    rate = live_data["forex"].get("conversion_rate", "N/A")
                    key_findings.append(f"USD/EUR: {rate}")
                
                if "interest_rates" in live_data:
                    rates = live_data["interest_rates"]
                    key_findings.append(f"Tasas actuales - Hipoteca: {rates['mortgage']}%, Personal: {rates['personal']}%, Auto: {rates['auto']}%")
                
                if "inflation" in live_data:
                    inflation = live_data["inflation"]
                    key_findings.append(f"Inflación general: {inflation['rate']}%")
                    for category, rate in inflation["categories"].items():
                        key_findings.append(f"Inflación {category}: {rate}%")
        except Exception as e:
            return self._generate_fallback_response(analysis, user_context, f"Error al procesar datos en vivo: {str(e)}")
        
        try:
            # Select the appropriate template
            template = self.templates.get(topic, self.templates["default"])
            
            # Format the template with the analysis content
            formatted_content = template.format(
                style=content_style.value,
                level=user_level.value,
                key_points=self._format_list(key_findings),
                implications=self._format_list(implications),
                recommendations=self._format_list(recommendations),
                risk_summary=risk_summary,
                market_context=market_context,
                indicators=self._format_indicators(live_data),
                trends=self._generate_trends(analysis),
                next_steps=self._suggest_next_steps(user_level)
            )
        except Exception as e:
            return self._generate_fallback_response(analysis, user_context, f"Error al formatear la respuesta: {str(e)}")
        
        try:
            # Generate the response using GPT-4
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Eres un experto financiero que explica conceptos 
                        complejos de manera simple y clara. Tu objetivo es que cualquier 
                        persona pueda entender las implicaciones financieras en su vida 
                        diaria. Usa un tono amigable y cercano, pero mantén la 
                        precisión técnica."""
                    },
                    {
                        "role": "user",
                        "content": f"""Basado en el siguiente análisis, genera una 
                        respuesta clara y concisa que explique las implicaciones 
                        financieras al usuario común:

{formatted_content}

Asegúrate de:
1. Usar lenguaje simple y directo
2. Dar ejemplos concretos cuando sea posible
3. Mantener un tono tranquilizador pero honesto
4. Incluir acciones específicas que el usuario puede tomar"""
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return self._generate_fallback_response(analysis, user_context, f"Error al generar respuesta con GPT-4: {str(e)}")
    
    def _format_list(self, items: list) -> str:
        """Format a list of items as a bulleted string."""
        return "\n".join(f"• {item}" for item in items)
    
    def _format_indicators(self, live_data: Dict[str, Any]) -> str:
        """Format economic indicators into a readable string."""
        indicators = []
        
        if "inflation" in live_data:
            inflation = live_data["inflation"]
            indicators.append(f"Inflación general: {inflation['rate']}%")
            
        if "interest_rates" in live_data:
            rates = live_data["interest_rates"]
            indicators.append(f"Tasa hipotecaria: {rates['mortgage']}%")
            
        if "forex" in live_data:
            rate = live_data["forex"].get("conversion_rate", "N/A")
            indicators.append(f"Tipo de cambio USD/EUR: {rate}")
            
        return "\n".join(indicators)
    
    def _generate_market_context(self, live_data: Dict[str, Any]) -> str:
        """Generate market context from live data."""
        context = []
        
        if "stocks" in live_data:
            performance = []
            for symbol, data in live_data["stocks"].items():
                if "Global Quote" in data:
                    change = data["Global Quote"].get("10. change percent", "N/A")
                    performance.append(f"{symbol}: {change}")
            if performance:
                context.append("Rendimiento del mercado:")
                context.extend(performance)
        
        return "\n".join(context)
    
    def _generate_trends(self, analysis: Dict[str, Any]) -> str:
        """Generate trend analysis from the data."""
        trends = analysis.get("trends", [])
        if not trends:
            return "No hay tendencias significativas identificadas"
        return "\n".join(f"• {trend}" for trend in trends)
    
    def _personalize_recommendations(self, recommendations: List[str], level: UserLevel) -> List[str]:
        """Personalize recommendations based on user level."""
        if level == UserLevel.BEGINNER:
            return [rec for rec in recommendations if "técnico" not in rec.lower()]
        elif level == UserLevel.INTERMEDIATE:
            return recommendations
        else:
            return [rec for rec in recommendations if "básico" not in rec.lower()]
    
    def _suggest_next_steps(self, level: UserLevel) -> str:
        """Suggest next steps based on user level."""
        if level == UserLevel.BEGINNER:
            return "1. Revisa los conceptos básicos mencionados\n2. Considera consultar con un asesor financiero\n3. Empieza con acciones pequeñas y seguras"
        elif level == UserLevel.INTERMEDIATE:
            return "1. Profundiza en el análisis técnico\n2. Diversifica tu portafolio\n3. Establece objetivos financieros específicos"
        else:
            return "1. Optimiza tu estrategia de inversión\n2. Considera instrumentos financieros avanzados\n3. Revisa tu plan de gestión de riesgos"
    
    def _generate_fallback_response(self, analysis: Dict[str, Any], user_context: Dict[str, Any], error: str) -> str:
        """Generate a fallback response when the main response generation fails."""
        try:
            key_findings = analysis.get("key_findings", [])
            implications = analysis.get("short_term_implications", [])
            
            response = "Lo siento, hubo un problema al generar la respuesta detallada. Aquí hay un resumen básico:\n\n"
            
            if key_findings:
                response += "Puntos principales:\n" + self._format_list(key_findings[:3]) + "\n\n"
            
            if implications:
                response += "Implicaciones:\n" + self._format_list(implications[:2]) + "\n\n"
            
            response += "Por favor, intenta tu consulta nuevamente o contacta soporte si el problema persiste."
            
            return response
        except:
            return "Lo siento, hubo un problema al procesar tu consulta. Por favor, intenta nuevamente."
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the writer agent."""
        try:
            analysis = input_data.get("analysis", {})
            user_context = input_data.get("context", {})
            response = await self.compose_response(analysis, user_context)
            return {"response": response}
        except Exception as e:
            error_response = self._generate_fallback_response({}, {}, f"Error en el procesamiento: {str(e)}")
            return {"response": error_response}
