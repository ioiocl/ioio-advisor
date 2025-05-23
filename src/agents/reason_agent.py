from typing import Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime  # Ensure datetime is imported

from ..ports.agent_port import ReasonAgent

class Mistral7BReasonAgent(ReasonAgent):
    """Implementation of the reasoning agent using Mistral 7B with enhanced financial analysis."""
    
    def __init__(self):
        self.model_name = "facebook/opt-350m"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Financial analysis thresholds
        self.thresholds = {
            "inflation": {
                "high": 5.0,  # Above 5% is concerning
                "moderate": 3.0,  # 2-5% is moderate
                "low": 2.0  # Below 2% is stable
            },
            "interest_rates": {
                "high": 6.0,  # Above 6% is high
                "moderate": 4.0,  # 4-6% is moderate
                "low": 2.0  # Below 2% is low
            },
            "market_change": {
                "significant": 2.0,  # >2% daily change is significant
                "moderate": 1.0,  # 1-2% is moderate
                "minor": 0.5  # <0.5% is minor
            }
        }
        
        # Risk assessment weights
        self.risk_weights = {
            "market_volatility": 0.3,
            "economic_indicators": 0.3,
            "global_factors": 0.2,
            "sector_specific": 0.2
        }
    
    async def analyze(
        self,
        query: str,
        retrieved_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze financial information using enhanced reasoning."""
        try:
            # Extract market data and educational content
            market_data = retrieved_info.get("information", {}).get("market_data", {})
            educational_content = retrieved_info.get("information", {}).get("educational_content", {})
            sentiment = retrieved_info.get("market_sentiment", "neutral")
            
            # Analyze risk factors
            risk_analysis = self._analyze_risk_factors(market_data)
            
            # Analyze market trends
            trend_analysis = self._analyze_market_trends(market_data)
            
            # Generate personalized insights
            insights = self._generate_insights(market_data, educational_content)
            
            # Prepare the analysis prompt with enhanced context
            prompt = self._generate_analysis_prompt(
                query=query,
                market_data=market_data,
                risk_analysis=risk_analysis,
                trend_analysis=trend_analysis,
                insights=insights,
                sentiment=sentiment
            )
            
            # Generate analysis using Mistral 7B
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
            
            analysis_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse and structure the analysis
            analysis = self._parse_analysis(analysis_text)
            
            # Add metadata
            analysis["metadata"] = {
                "risk_score": risk_analysis["overall_risk"],
                "confidence_factors": self._calculate_confidence_factors(market_data),
                "data_quality": self._assess_data_quality(market_data),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error in financial analysis: {str(e)}")
            return {
                "error": "Lo siento, hubo un problema al analizar la información financiera.",
                "fallback_analysis": self._generate_fallback_analysis(query),
                "metadata": {
                    "error": str(e),
                    "using_fallback": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    def _generate_analysis_prompt(
        self,
        query: str,
        market_data: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        insights: Dict[str, Any],
        sentiment: str
    ) -> str:
        """Generate an enhanced prompt for financial analysis."""
        risk_level = risk_analysis.get("overall_risk", "moderado")
        key_trends = "\n".join(f"- {trend}" for trend in trend_analysis.get("key_trends", []))
        key_insights = "\n".join(f"- {insight}" for insight in insights.get("key_points", []))
        
        return f"""Analiza las implicaciones financieras de la siguiente consulta usando razonamiento estructurado:

Consulta: {query}

Contexto del Mercado:
- Sentimiento general: {sentiment}
- Nivel de riesgo: {risk_level}

Tendencias Principales:
{key_trends}

Hallazgos Clave:
{key_insights}

Análisis paso a paso:

1) Primero, analicemos los factores principales:
- Indicadores económicos actuales
- Tendencias del mercado
- Factores de riesgo

2) Examinemos las relaciones entre estos factores:
- Correlaciones importantes
- Impactos mutuos
- Factores externos

3) Consideremos las implicaciones:
- Impacto a corto plazo
- Perspectivas a mediano plazo
- Riesgos y oportunidades

4) Finalmente, formulemos conclusiones y recomendaciones:
- Conclusiones principales
- Acciones recomendadas
- Consideraciones importantes

Por favor, estructura la respuesta con:
1. Hallazgos clave
2. Implicaciones a corto plazo
3. Perspectiva a mediano plazo
4. Acciones recomendadas
5. Nivel de confianza

Respuesta:"""
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as a bulleted string."""
        return "\n".join(f"- {item}" for item in items)
    
    def _analyze_risk_factors(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk factors in market data."""
        risk_scores = {
            "market_volatility": self._assess_market_volatility(market_data),
            # Use fallback value if economic_health is missing or raises an error
            "economic_health": None,
            "global_exposure": self._assess_global_exposure(market_data),
            "sector_risks": self._assess_sector_risks(market_data)
        }
        try:
            risk_scores["economic_health"] = self._assess_economic_health(market_data)
            if risk_scores["economic_health"] is None:
                raise ValueError("economic_health returned None")
        except Exception as e:
            print(f"Warning: Could not assess economic_health: {e}")
            risk_scores["economic_health"] = 0.5  # Default fallback risk
        
        # Calculate overall risk score
        overall_risk = sum(
            score * self.risk_weights[factor]
            for factor, score in risk_scores.items()
        )
        
        # Convert to risk level in Spanish
        risk_level = (
            "alto" if overall_risk > 0.7
            else "moderado" if overall_risk > 0.4
            else "bajo"
        )
        
        return {
            "overall_risk": risk_level,
            "risk_scores": risk_scores,
            "risk_factors": self._identify_key_risk_factors(risk_scores)
        }
    
    def _analyze_market_trends(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends from the data."""
        trends = []
        
        # Analyze market indices
        indices = market_data.get("indices", {})
        for index, values in indices.items():
            change = float(values.get("change", "0%").strip("%") or 0)
            if abs(change) > self.thresholds["market_change"]["significant"]:
                trend = (
                    f"Movimiento significativo en {index}: {'subida' if change > 0 else 'bajada'} "
                    f"del {abs(change):.1f}%"
                )
                trends.append(trend)
        
        # Analyze interest rates
        rates = market_data.get("interest_rates", {})
        for rate_type, value in rates.items():
            rate = float(value.strip("%") or 0)
            if rate > self.thresholds["interest_rates"]["high"]:
                trends.append(f"Tasas de {rate_type} elevadas: {rate:.1f}%")
        
        # Analyze inflation
        inflation = market_data.get("inflation", {})
        current_rate = float(inflation.get("current_rate", "0%").strip("%") or 0)
        if current_rate > self.thresholds["inflation"]["high"]:
            trends.append(f"Inflación alta: {current_rate:.1f}%")
        
        return {
            "key_trends": trends,
            "trend_strength": self._calculate_trend_strength(market_data),
            "trend_duration": self._estimate_trend_duration(market_data)
        }
    
    def _generate_insights(self, market_data: Dict[str, Any], educational_content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized insights from market data and educational content."""
        insights = []
        
        # Market insights
        market_state = self._analyze_market_state(market_data)
        insights.extend(market_state["insights"])
        
        # Educational insights
        for topic, content in educational_content.items():
            for concept in content.get("concepts", []):
                if self._is_relevant_to_market_state(concept, market_state):
                    insights.append(concept)
        
        return {
            "key_points": insights[:5],  # Top 5 most relevant insights
            "market_context": market_state,
            "educational_relevance": self._calculate_educational_relevance(insights)
        }
    
    def _calculate_confidence_factors(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence factors for different aspects of the analysis."""
        return {
            "data_completeness": self._assess_data_completeness(market_data),
            "trend_clarity": self._assess_trend_clarity(market_data),
            "risk_certainty": self._assess_risk_certainty(market_data)
        }
    
    def _assess_data_quality(self, market_data: Dict[str, Any]) -> str:
        """Assess the quality of market data."""
        quality_score = self._calculate_data_quality_score(market_data)
        
        if quality_score > 0.8:
            return "alta"
        elif quality_score > 0.5:
            return "media"
        return "baja"
    
    def _generate_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Generate a basic fallback analysis when detailed analysis fails."""
        return {
            "hallazgos_clave": [
                "Datos limitados disponibles",
                "Se requiere más información para un análisis detallado"
            ],
            "recomendaciones": [
                "Consultar fuentes adicionales de información",
                "Monitorear el mercado para obtener datos más completos"
            ],
            "nivel_confianza": "bajo"
        }
    
    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the analysis text into structured data."""
        sections = {
            "hallazgos_clave": [],
            "implicaciones_corto_plazo": [],
            "perspectiva_mediano_plazo": [],
            "acciones_recomendadas": [],
            "nivel_confianza": 0.0,
            "cadena_razonamiento": []
        }
        
        current_section = None
        
        for line in analysis_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers in Spanish
            if "Hallazgos clave" in line:
                current_section = "hallazgos_clave"
            elif "Implicaciones a corto plazo" in line:
                current_section = "implicaciones_corto_plazo"
            elif "Perspectiva a mediano plazo" in line:
                current_section = "perspectiva_mediano_plazo"
            elif "Acciones recomendadas" in line:
                current_section = "acciones_recomendadas"
            elif "Nivel de confianza" in line:
                try:
                    confidence = float(line.split(":")[-1].strip().rstrip("%")) / 100
                    sections["nivel_confianza"] = confidence
                except ValueError:
                    sections["nivel_confianza"] = 0.8
            elif line.startswith(("1)", "2)", "3)", "4)")):
                sections["cadena_razonamiento"].append(line)
            elif current_section and current_section != "nivel_confianza":
                if line.startswith("- "):
                    sections[current_section].append(line[2:])
                else:
                    sections[current_section].append(line)
        
        return sections
    
    def _assess_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Assess market volatility from indices and rates."""
        volatility_score = 0.0
        count = 0
        
        # Check index changes
        for values in market_data.get("indices", {}).values():
            change = abs(float(values.get("change", "0%").strip("%") or 0))
            if change > self.thresholds["market_change"]["significant"]:
                volatility_score += 1.0
            elif change > self.thresholds["market_change"]["moderate"]:
                volatility_score += 0.5
            count += 1
        
        return volatility_score / max(count, 1)
    
    def _assess_economic_health(self, market_data: Dict[str, Any]) -> float:
        """Assess overall economic health from indicators."""
        health_score = 0.0
        indicators = 0
        
        # Check inflation
        inflation = float(market_data.get("inflation", {}).get("current_rate", "0%").strip("%") or 0)
        if inflation < self.thresholds["inflation"]["low"]:
            health_score += 1.0
        elif inflation < self.thresholds["inflation"]["moderate"]:
            health_score += 0.5
        indicators += 1
        
        # Check interest rates
        rates = market_data.get("interest_rates", {})
        if rates:
            avg_rate = sum(float(rate.strip("%") or 0) for rate in rates.values()) / len(rates)
            if avg_rate < self.thresholds["interest_rates"]["moderate"]:
                health_score += 1.0
            elif avg_rate < self.thresholds["interest_rates"]["high"]:
                health_score += 0.5
            indicators += 1
        
        return health_score / max(indicators, 1)
    
    def _assess_global_exposure(self, market_data: Dict[str, Any]) -> float:
        """Assess exposure to global market factors."""
        # Simplified implementation - could be enhanced with more global factors
        return 0.5  # Moderate global exposure by default
    
    def _assess_sector_risks(self, market_data: Dict[str, Any]) -> float:
        """Assess sector-specific risks."""
        # Simplified implementation - could be enhanced with sector analysis
        return 0.4  # Moderate sector risk by default
    
    def _identify_key_risk_factors(self, risk_scores: Dict[str, float]) -> List[str]:
        """Identify key risk factors from risk scores."""
        factors = []
        for factor, score in risk_scores.items():
            if score > 0.7:
                factors.append(f"Riesgo alto: {factor}")
            elif score > 0.4:
                factors.append(f"Riesgo moderado: {factor}")
        return factors
    
    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> str:
        """Calculate the strength of market trends."""
        strength_score = 0.0
        count = 0
        
        for values in market_data.get("indices", {}).values():
            change = abs(float(values.get("change", "0%").strip("%") or 0))
            if change > self.thresholds["market_change"]["significant"]:
                strength_score += 1.0
            elif change > self.thresholds["market_change"]["moderate"]:
                strength_score += 0.5
            count += 1
        
        avg_strength = strength_score / max(count, 1)
        return "fuerte" if avg_strength > 0.7 else "moderada" if avg_strength > 0.3 else "débil"
    
    def _estimate_trend_duration(self, market_data: Dict[str, Any]) -> str:
        """Estimate the duration of current trends."""
        # Simplified implementation - could be enhanced with historical data
        return "corto plazo"  # Default to short term
    
    def _analyze_market_state(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current market state."""
        insights = []
        
        # Analyze market indices
        indices = market_data.get("indices", {})
        if indices:
            avg_change = sum(
                float(values.get("change", "0%").strip("%") or 0)
                for values in indices.values()
            ) / len(indices)
            
            if avg_change > 1.0:
                insights.append("Mercado alcista con tendencia positiva")
            elif avg_change < -1.0:
                insights.append("Mercado bajista con precaución")
        
        # Analyze rates and inflation
        rates = market_data.get("interest_rates", {})
        inflation = market_data.get("inflation", {})
        
        if rates and inflation:
            inflation_rate = float(inflation.get("current_rate", "0%").strip("%") or 0)
            savings_rate = float(rates.get("savings", "0%").strip("%") or 0)
            
            if savings_rate > inflation_rate:
                insights.append("Tasas de ahorro favorables vs inflación")
            else:
                insights.append("Inflación supera rendimientos de ahorro")
        
        return {
            "insights": insights,
            "market_direction": "positivo" if avg_change > 0 else "negativo",
            "stability": "estable" if abs(avg_change) < 1.0 else "volátil"
        }
    
    def _is_relevant_to_market_state(self, concept: str, market_state: Dict[str, Any]) -> bool:
        """Determine if an educational concept is relevant to current market state."""
        # Simplified implementation - could be enhanced with NLP
        return True  # Default to including all concepts
    
    def _calculate_educational_relevance(self, insights: List[str]) -> float:
        """Calculate the relevance score of educational content."""
        # Simplified implementation
        return 0.8  # High relevance by default
    
    def _assess_data_completeness(self, market_data: Dict[str, Any]) -> float:
        """Assess the completeness of market data."""
        required_fields = ["indices", "interest_rates", "inflation"]
        available = sum(1 for field in required_fields if field in market_data)
        return available / len(required_fields)
    
    def _assess_trend_clarity(self, market_data: Dict[str, Any]) -> float:
        """Assess how clear the market trends are."""
        clarity_score = 0.0
        count = 0
        
        for values in market_data.get("indices", {}).values():
            change = abs(float(values.get("change", "0%").strip("%") or 0))
            if change > self.thresholds["market_change"]["significant"]:
                clarity_score += 1.0
            elif change > self.thresholds["market_change"]["moderate"]:
                clarity_score += 0.5
            count += 1
        
        return clarity_score / max(count, 1)
    
    def _assess_risk_certainty(self, market_data: Dict[str, Any]) -> float:
        """Assess the certainty of risk assessment."""
        # Simplified implementation
        return 0.7  # Moderately high certainty by default
    
    def _calculate_data_quality_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        scores = [
            self._assess_data_completeness(market_data),
            self._assess_trend_clarity(market_data),
            self._assess_risk_certainty(market_data)
        ]
        return sum(scores) / len(scores)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the reason agent."""
        try:
            query = input_data.get("query", "")
            retrieved_info = input_data.get("retrieved_info", {})
            return await self.analyze(query, retrieved_info)
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return {
                "error": "Lo siento, hubo un problema al procesar la consulta.",
                "details": str(e)
            }
