from typing import Dict, Any
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..ports.agent_port import IntentionAgent

class Phi3IntentionAgent(IntentionAgent):
    """Implementation of the intention detection agent using Phi-3 Mini."""
    
    def __init__(self):
        """Initialize the agent with the Phi-3 Mini model."""
        self.model_name = os.getenv('PHI3_MODEL_PATH', 'microsoft/phi-3')
        self.hf_token = os.getenv('HUGGINGFACE_API_KEY')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Pre-defined financial topics and their keywords
        self.topics = {
            "market": ["acciones", "bolsa", "mercado", "tecnologia", "tech"],
            "currency_impact": ["dolar", "tipo de cambio", "moneda", "divisa", "usd", "precio del dolar"],
            "inflation": ["inflacion", "precios", "costo de vida", "ipc", "tasa de inflacion"],
            "investments": ["inversion", "invertir", "ahorros", "portfolio", "cartera", "fondos"],
            "budget": ["presupuesto", "gastos", "finanzas personales", "ahorrar", "organizar"],
            "stock": ["acciones", "tecnologia", "tech", "comprar acciones", "vender acciones"]
        }
    
    async def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect the user's intention from their query."""
        
        # Prepare prompt for Phi-3
        prompt = f"""Analyze the following financial query and identify:
1. Main topic
2. Subtopics
3. User's intention

Query: "{query}"

Format your response as JSON with the following structure:
{{
    "main_topic": "topic_name",
    "subtopics": ["subtopic1", "subtopic2"],
    "intention": "brief_description",
    "confidence": 0.95
}}

Response:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the JSON part (this is a simplified version)
        try:
            import json
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
        except:
            # Fallback to keyword-based detection
            main_topic = self._detect_main_topic(query)
            subtopics = self._detect_subtopics(query)
            
            # Map query types to intentions
            if "como" in query.lower() or "cual" in query.lower():
                intention = "get_information"
            elif "donde" in query.lower() or "que" in query.lower():
                intention = "get_recommendation"
            elif "por que" in query.lower():
                intention = "understand_reason"
            else:
                intention = "get_analysis"
            
            result = {
                "main_topic": main_topic,
                "subtopics": subtopics,
                "intention": intention,
                "confidence": 0.9
            }
        
        return result
    
    def _detect_main_topic(self, query: str) -> str:
        """Detect the main topic based on keywords."""
        query = query.lower()
        
        # Check for specific patterns first
        query = query.lower()
        
        if any(term in query for term in ["acciones", "bolsa", "mercado"]) and \
           any(term in query for term in ["tecnologia", "tech"]):
            return "stock"
            
        if "dolar" in query or "usd" in query:
            return "currency_impact"
            
        if "ahorros" in query or "invertir" in query:
            return "investments"
            
        if "inflacion" in query or "ipc" in query or "inflación" in query:
            return "inflation"
            
        # Fall back to general keyword matching
        max_matches = 0
        best_topic = "general_finance"
        
        for topic, keywords in self.topics.items():
            matches = sum(1 for keyword in keywords if keyword in query)
            if matches > max_matches:
                max_matches = matches
                best_topic = topic
                
        return best_topic
    
    def _detect_subtopics(self, query: str) -> list[str]:
        """Detect relevant subtopics based on query content."""
        query = query.lower()
        subtopics = []
        for topic, keywords in self.topics.items():
            if any(keyword in query for keyword in keywords):
                subtopics.append(topic)
        return subtopics[:3]  # Return top 3 most relevant subtopics
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the intention agent."""
        query = input_data.get("query", "")
        return await self.detect_intent(query)
