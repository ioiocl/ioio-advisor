import pytest
import asyncio
from unittest.mock import Mock, patch
from src.agents.reason_agent import Mistral7BReasonAgent

@pytest.fixture
def reason_agent():
    agent = Mistral7BReasonAgent()
    return agent

@pytest.mark.asyncio
async def test_market_volatility_assessment(reason_agent):
    """Test market volatility assessment"""
    market_data = {
        "indices": {
            "MERVAL": {"value": "950000", "change": "+1.2%"},
            "S&P500": {"value": "4850", "change": "+0.8%"}
        },
        "volatility": {
            "daily": "15.2%",
            "weekly": "12.8%"
        }
    }
    
    volatility = reason_agent._assess_market_volatility(market_data)
    assert isinstance(volatility, dict)
    assert "level" in volatility
    assert "score" in volatility
    assert "factors" in volatility
    assert isinstance(volatility["score"], float)
    assert 0 <= volatility["score"] <= 100

@pytest.mark.asyncio
async def test_economic_health_indicators(reason_agent):
    """Test economic health indicators analysis"""
    economic_data = {
        "gdp_growth": "+2.5%",
        "unemployment": "7.2%",
        "inflation": {
            "monthly": "4.5%",
            "annual": "52.3%"
        },
        "interest_rates": {
            "policy_rate": "118%",
            "market_rate": "125%"
        }
    }
    
    health_indicators = reason_agent._analyze_economic_health(economic_data)
    assert isinstance(health_indicators, dict)
    assert "overall_health" in health_indicators
    assert "risk_factors" in health_indicators
    assert "confidence" in health_indicators
    assert isinstance(health_indicators["confidence"], float)

@pytest.mark.asyncio
async def test_risk_factor_analysis(reason_agent):
    """Test risk factor analysis system"""
    data = {
        "market_volatility": {"score": 75.5, "level": "high"},
        "economic_indicators": {"health": "moderate", "trend": "declining"},
        "global_factors": {"impact": "significant", "trend": "stable"},
        "sector_risks": {"level": "moderate", "factors": ["inflation", "rates"]}
    }
    
    risk_analysis = reason_agent._analyze_risk_factors(data)
    assert isinstance(risk_analysis, dict)
    assert "total_risk_score" in risk_analysis
    assert "risk_breakdown" in risk_analysis
    assert len(risk_analysis["risk_breakdown"]) >= 4
    assert 0 <= risk_analysis["total_risk_score"] <= 100

@pytest.mark.asyncio
async def test_trend_strength_calculation(reason_agent):
    """Test trend strength calculation"""
    market_data = {
        "price_data": [100, 105, 108, 112, 115],
        "volume_data": [1000, 1200, 1100, 1300, 1400],
        "momentum_indicators": {
            "rsi": 65,
            "macd": "positive"
        }
    }
    
    trend_analysis = reason_agent._calculate_trend_strength(market_data)
    assert isinstance(trend_analysis, dict)
    assert "strength" in trend_analysis
    assert "confidence" in trend_analysis
    assert "direction" in trend_analysis
    assert isinstance(trend_analysis["strength"], float)
    assert 0 <= trend_analysis["strength"] <= 100

@pytest.mark.asyncio
async def test_market_sentiment_integration(reason_agent):
    """Test market sentiment integration"""
    sentiment_data = {
        "news_sentiment": "positive",
        "technical_indicators": "bullish",
        "market_momentum": "strong",
        "investor_confidence": "moderate"
    }
    
    sentiment = reason_agent._analyze_market_sentiment(sentiment_data)
    assert isinstance(sentiment, dict)
    assert "overall_sentiment" in sentiment
    assert "confidence_score" in sentiment
    assert "contributing_factors" in sentiment
    assert isinstance(sentiment["confidence_score"], float)

@pytest.mark.asyncio
async def test_data_quality_assessment(reason_agent):
    """Test data quality assessment"""
    data = {
        "market_data": {"completeness": 0.85, "timestamp": "2025-05-22T14:15:00"},
        "economic_data": {"completeness": 0.92, "timestamp": "2025-05-22T14:00:00"},
        "sentiment_data": {"completeness": 0.78, "timestamp": "2025-05-22T14:10:00"}
    }
    
    quality_assessment = reason_agent._assess_data_quality(data)
    assert isinstance(quality_assessment, dict)
    assert "overall_quality" in quality_assessment
    assert "data_freshness" in quality_assessment
    assert "completeness_score" in quality_assessment
    assert isinstance(quality_assessment["completeness_score"], float)

@pytest.mark.asyncio
async def test_spanish_language_output(reason_agent):
    """Test Spanish language output generation"""
    analysis_data = {
        "risk_level": "high",
        "market_trend": "upward",
        "key_factors": ["inflation", "interest_rates"]
    }
    
    description = reason_agent._generate_spanish_description(analysis_data)
    assert isinstance(description, str)
    assert any(word in description.lower() for word in ["riesgo", "tendencia", "inflación"])
    assert "%" in description or "por ciento" in description.lower()

@pytest.mark.asyncio
async def test_error_handling_and_fallback(reason_agent):
    """Test error handling and fallback analysis"""
    # Test with incomplete data
    incomplete_data = {"partial": "data"}
    result = await reason_agent.analyze("test query", incomplete_data)
    assert isinstance(result, dict)
    assert "error" in result or "fallback_analysis" in result
    
    # Test with invalid data
    invalid_data = None
    result = await reason_agent.analyze("test query", invalid_data)
    assert isinstance(result, dict)
    assert "error" in result
    assert isinstance(result.get("error"), str)

@pytest.mark.asyncio
async def test_confidence_scoring(reason_agent):
    """Test confidence scoring system"""
    analysis_data = {
        "market_data": {"completeness": 0.9},
        "economic_data": {"completeness": 0.85},
        "sentiment_data": {"completeness": 0.75}
    }
    
    confidence = reason_agent._calculate_confidence_score(analysis_data)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1
    assert "confidence_score" in reason_agent._get_metadata(analysis_data)

@pytest.mark.asyncio
async def test_full_analysis_pipeline(reason_agent):
    """Test the complete analysis pipeline"""
    query = "¿Cómo afecta la inflación al mercado de acciones?"
    data = {
        "information": {
            "market_data": {
                "indices": {
                    "MERVAL": {"value": "950000", "change": "+1.2%"}
                },
                "interest_rates": {
                    "policy_rate": "118%"
                },
                "inflation": {
                    "monthly": "4.5%",
                    "annual": "52.3%"
                }
            },
            "educational_content": {
                "inflation": {
                    "concepts": ["La inflación afecta el valor real de las acciones"],
                    "examples": ["Cuando la inflación sube, las empresas pueden ajustar precios"]
                }
            }
        },
        "market_sentiment": "neutral"
    }
    
    result = await reason_agent.analyze(query, data)
    
    # Verify response structure
    assert isinstance(result, dict)
    assert "hallazgos_clave" in result
    assert "implicaciones_corto_plazo" in result
    assert "perspectiva_mediano_plazo" in result
    assert "acciones_recomendadas" in result
    assert "metadata" in result
    
    # Verify content
    assert len(result["hallazgos_clave"]) > 0
    assert len(result["implicaciones_corto_plazo"]) > 0
    assert len(result["acciones_recomendadas"]) > 0
    assert isinstance(result["metadata"]["confidence"], float)
    
    # Verify Spanish language
    response_text = str(result)
    assert any(word in response_text.lower() for word in [
        "inflación", "mercado", "acciones", "riesgo", "análisis"
    ])
