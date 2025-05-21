from typing import Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import aiohttp
from datetime import datetime, timedelta

from ..ports.agent_port import RetrieverAgent

class InstructorXLRetrieverAgent(RetrieverAgent):
    """Implementation of the retriever agent using Instructor XL."""
    
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
            "news": "https://newsapi.org/v2/everything"
        }
    
    async def retrieve_information(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve relevant financial information based on query and context."""
        intent = context.get("intent", {})
        main_topic = intent.get("main_topic", "general_finance")
        
        # Collect data from different sources
        data = await self._gather_financial_data(main_topic)
        
        # Generate retrieval instruction
        instruction = self._generate_retrieval_instruction(query, main_topic)
        
        # Process data with Instructor XL
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True
            )
        
        processed_info = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Structure the response
        return {
            "raw_data": data,
            "processed_info": self._parse_processed_info(processed_info),
            "sources": list(data.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_retrieval_instruction(self, query: str, topic: str) -> str:
        """Generate instruction for the model based on query and topic."""
        return f"""Given the following financial data and user query, extract and organize relevant information:

Query: {query}
Topic: {topic}

Instructions:
1. Identify key financial metrics and their current values
2. Extract relevant trends and patterns
3. Find correlations between different data points
4. Organize information by relevance to the query

Format the response as a structured analysis with sections for:
- Key Metrics
- Trends
- Implications
- Additional Context

Response:"""
    
    async def _gather_financial_data(self, topic: str) -> Dict[str, Any]:
        """Gather financial data from various sources based on the topic."""
        data = {}
        
        async with aiohttp.ClientSession() as session:
            # Get exchange rates
            if topic in ["currency_impact", "general_finance"]:
                async with session.get(self.api_endpoints["exchange_rate"]) as response:
                    if response.status == 200:
                        data["exchange_rates"] = await response.json()
            
            # Get economic indicators
            indicators = {
                "inflation": "FP.CPI.TOTL.ZG",
                "gdp": "NY.GDP.MKTP.KD.ZG"
            }
            if topic in indicators:
                indicator = indicators[topic]
                async with session.get(
                    f"{self.api_endpoints['economic_indicators']}{indicator}"
                ) as response:
                    if response.status == 200:
                        data["economic_indicators"] = await response.json()
            
            # Get relevant news
            async with session.get(
                f"{self.api_endpoints['news']}?q={topic}&language=es&sortBy=publishedAt"
            ) as response:
                if response.status == 200:
                    data["news"] = await response.json()
        
        return data
    
    def _parse_processed_info(self, info: str) -> Dict[str, Any]:
        """Parse the model's output into structured information."""
        sections = ["Key Metrics", "Trends", "Implications", "Additional Context"]
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
