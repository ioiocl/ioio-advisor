from typing import Dict, Any
import os
import json
import aiohttp
from datetime import datetime
from openai import AsyncOpenAI
from ..ports.agent_port import WriterAgent

class GPT4WriterAgent(WriterAgent):
    """Implementation of the writer agent using GPT-4."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-turbo-preview"
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "demo")
        self.exchangerate_key = os.getenv("EXCHANGERATE_API_KEY", "demo")
        
        # Templates for different response styles
        self.templates = {
            "currency_impact": """
Explica de manera simple y clara cómo el dólar afecta la economía personal:

{key_points}

Implicaciones principales:
{implications}

Recomendaciones:
{recommendations}
""",
            "inflation": """
Análisis simple de la situación inflacionaria:

{key_points}

¿Cómo te afecta?
{implications}

¿Qué puedes hacer?
{recommendations}
""",
            "default": """
Análisis financiero:

{key_points}

Implicaciones:
{implications}

Recomendaciones:
{recommendations}
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

    async def compose_response(
        self,
        analysis: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """Compose a clear and concise response using GPT-4."""
        
        # Enrich analysis with live data
        analysis = await self.enrich_with_live_data(analysis, user_context)
        
        # Prepare the content for the prompt
        key_findings = analysis.get("key_findings", [])
        implications = analysis.get("short_term_implications", [])
        recommendations = analysis.get("recommended_actions", [])
        live_data = analysis.get("live_data", {})
        topic = user_context.get("intent", {}).get("main_topic", "default")
        
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
        
        # Select the appropriate template
        template = self.templates.get(topic, self.templates["default"])
        
        # Format the template with the analysis content
        formatted_content = template.format(
            key_points=self._format_list(key_findings),
            implications=self._format_list(implications),
            recommendations=self._format_list(recommendations)
        )
        
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
    
    def _format_list(self, items: list) -> str:
        """Format a list of items as a bulleted string."""
        return "\n".join(f"• {item}" for item in items)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the writer agent."""
        analysis = input_data.get("analysis", {})
        user_context = input_data.get("context", {})
        response = await self.compose_response(analysis, user_context)
        return {"response": response}
