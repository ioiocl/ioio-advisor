from typing import Dict, Any
import os
import torch
import unicodedata
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
        
        # Pre-defined financial topics with keywords and phrases (normalized text)
        self.topic_patterns = {
            'currency': {
                'keywords': ['dolar', 'euro', 'peso', 'moneda', 'cambio', 'divisa', 'tipo de cambio', 'usd', 'eur', 'bolsillo', 'cotizacion', 'devaluacion', 'apreciacion'],
                'phrases': ['sube el dolar', 'baja el dolar', 'precio del dolar', 'cotizacion del dolar', 'valor del peso', 'tipo de cambio', 'mercado cambiario', 'para mi bolsillo', 'significa para mi bolsillo']
            },
            'investment': {
                'keywords': ['inversion', 'invertir', 'acciones', 'bolsa', 'mercado', 'fondo', 'portfolio', 'cartera', 'rendimiento'],
                'phrases': ['donde invertir', 'como invertir', 'mercado de valores', 'rendimiento de inversion', 'retorno de inversion']
            },
            'interest': {
                'keywords': ['interes', 'tasa', 'prestamo', 'hipoteca', 'credito', 'financiamiento', 'deuda'],
                'phrases': ['tasa de interes', 'prestamo bancario', 'credito hipotecario', 'tasa preferencial', 'costo financiero']
            },
            'inflation': {
                'keywords': ['inflacion', 'precios', 'costo', 'ipc', 'carestia', 'poder adquisitivo', 'devaluacion'],
                'phrases': ['subida de precios', 'aumento de costos', 'perdida de valor', 'poder de compra', 'costo de vida']
            },
            'budget': {
                'keywords': ['presupuesto', 'gasto', 'ahorro', 'ingreso', 'sueldo', 'salario', 'finanzas personales'],
                'phrases': ['como ahorrar', 'control de gastos', 'manejo de dinero', 'finanzas personales', 'economia domestica']
            }
        }
        
        # Normalize all keywords and phrases
        for patterns in self.topic_patterns.values():
            patterns["keywords"] = [self._normalize_text(k) for k in patterns["keywords"]]
            patterns["phrases"] = [self._normalize_text(p) for p in patterns["phrases"]]
    
    async def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect the intent of a user query."""
        normalized_query = self._normalize_text(query)
        main_topic = self._detect_main_topic(normalized_query)
        query_type = self._detect_query_type(normalized_query)
        subtopics = self._detect_subtopics(query)
        
        # Prepare model input with more context
        model_input = f"Query: {query}\nNormalized: {normalized_query}\nTopic: {main_topic}\nType: {query_type}\nSubtopics: {', '.join(subtopics)}"
        
        # Get model prediction
        prediction = await self.model.predict(model_input)
        # Validate and enhance topic detection using model output
        model_topic = prediction.get('topic', main_topic)
        if model_topic != 'unknown' and model_topic in self.topic_patterns:
            main_topic = model_topic
        
        # Calculate confidence based on multiple factors
        topic_confidence = 1.0 if len(subtopics) > 0 else 0.7
        model_confidence = prediction.get('confidence', 0.7)
        confidence = (topic_confidence + model_confidence) / 2
        
        # Determine final query type
        model_type = prediction.get('query_type', query_type)
        if model_type in ['get_information', 'get_recommendation', 'understand_reason', 'get_analysis']:
            query_type = model_type
        
        return {
            "main_topic": main_topic,
            "query_type": query_type,
            "subtopics": subtopics,
            "confidence": round(confidence, 2)
        }
    
    def _detect_query_type(self, normalized_query: str) -> str:
        """Detect the type of query based on patterns."""
        # Question patterns (already normalized, so no need for accents)
        if "como" in normalized_query or "cual" in normalized_query:
            return "get_information"
        elif "donde" in normalized_query or "que" in normalized_query:
            return "get_recommendation"
        elif "por que" in normalized_query or "significa" in normalized_query:
            return "understand_reason"
        elif "analiza" in normalized_query or "explica" in normalized_query:
            return "get_analysis"
        else:
            # Default to get_information for questions we don't recognize
            return "get_information"
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing accents, punctuation, and converting to lowercase."""
        # Convert to lowercase first
        text = text.lower()
        # Remove question marks and other punctuation
        text = text.replace('¿', '').replace('?', '').replace('.', '').replace(',', '')
        # Normalize Unicode characters (NFD splits characters and their combining marks)
        text = unicodedata.normalize('NFD', text)
        # Remove diacritical marks
        text = ''.join(c for c in text if not unicodedata.combining(c))
        return text

    def _detect_main_topic(self, query: str) -> str:
        """Detect the main topic of the query."""
        # Normalize both the query and patterns
        normalized_query = self._normalize_text(query)
        
        # Check each topic's keywords and phrases
        for topic, patterns in self.topic_patterns.items():
            # Normalize and check keywords
            normalized_keywords = [self._normalize_text(k) for k in patterns['keywords']]
            if any(keyword in normalized_query for keyword in normalized_keywords):
                return topic
            
            # Normalize and check phrases
            normalized_phrases = [self._normalize_text(p) for p in patterns['phrases']]
            if any(phrase in normalized_query for phrase in normalized_phrases):
                return topic
        
        return 'general_finance'
    
    def _detect_subtopics(self, query: str) -> list[str]:
        """Detect relevant subtopics based on query content."""
        subtopics = set()  # Use a set to avoid duplicates
        normalized_query = self._normalize_text(query)
        
        # Check each topic's keywords and phrases
        for topic, patterns in self.topic_patterns.items():
            # Normalize and check keywords
            normalized_keywords = [self._normalize_text(k) for k in patterns['keywords']]
            if any(keyword in normalized_query for keyword in normalized_keywords):
                subtopics.add(topic)
                continue  # If we found a keyword match, no need to check phrases
            
            # Normalize and check phrases
            normalized_phrases = [self._normalize_text(p) for p in patterns['phrases']]
            if any(phrase in normalized_query for phrase in normalized_phrases):
                subtopics.add(topic)
        
        return list(subtopics)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the intention agent."""
        query = input_data.get("query", "")
        return await self.detect_intent(query)
