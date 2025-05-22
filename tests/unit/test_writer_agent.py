import pytest
import asyncio
from unittest.mock import Mock, patch
from src.agents.writer_agent import GPT4WriterAgent

@pytest.fixture
def writer_agent():
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        agent = GPT4WriterAgent()
        return agent

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling and fallback responses"""
    agent = GPT4WriterAgent()
    
    # Test with empty analysis
    result = await agent.compose_response({}, {})
    assert isinstance(result, str)
    assert "Lo siento" in result or "Error" in result
    
    # Test with invalid data
    result = await agent.compose_response({"invalid": "data"}, {})
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_user_level_determination(writer_agent):
    """Test user financial knowledge level determination"""
    # Test beginner level detection
    beginner_context = {
        "user_profile": {
            "experience": "novice",
            "previous_queries": ["¿Qué es un plazo fijo?", "¿Cómo funciona la inflación?"]
        }
    }
    level = writer_agent._determine_user_level(beginner_context)
    assert str(level).lower() == "beginner"
    
    # Test advanced level detection
    advanced_context = {
        "user_profile": {
            "experience": "expert",
            "previous_queries": ["Análisis técnico del MERVAL", "Estrategia de cobertura con opciones"]
        }
    }
    level = writer_agent._determine_user_level(advanced_context)
    assert str(level).lower() == "advanced"

@pytest.mark.asyncio
async def test_content_style_determination(writer_agent):
    """Test content style determination based on context"""
    # Test technical style
    technical_context = {
        "intent": {
            "main_topic": "technical_analysis",
            "complexity": "high"
        }
    }
    style = writer_agent._determine_content_style(technical_context)
    assert "technical" in str(style).lower()
    
    # Test educational style
    educational_context = {
        "intent": {
            "main_topic": "basic_concepts",
            "complexity": "low"
        }
    }
    style = writer_agent._determine_content_style(educational_context)
    assert "educational" in str(style).lower()

@pytest.mark.asyncio
async def test_risk_summary_generation(writer_agent):
    """Test risk summary generation"""
    analysis = {
        "risk_factors": [
            {"factor": "volatilidad", "level": "high"},
            {"factor": "liquidez", "level": "medium"}
        ],
        "market_conditions": {
            "sentiment": "negative",
            "volatility": "high"
        }
    }
    
    risk_summary = writer_agent._generate_risk_summary(analysis)
    assert isinstance(risk_summary, str)
    assert "riesgo" in risk_summary.lower()
    assert "volatilidad" in risk_summary.lower()

@pytest.mark.asyncio
async def test_market_context_generation(writer_agent):
    """Test market context generation from live data"""
    live_data = {
        "stocks": {
            "AAPL": {"Global Quote": {"05. price": "180.5", "10. change percent": "+2.5%"}},
            "GOOGL": {"Global Quote": {"05. price": "140.2", "10. change percent": "-1.2%"}}
        },
        "forex": {
            "conversion_rate": "0.85"
        },
        "interest_rates": {
            "mortgage": "5.2",
            "personal": "12.5",
            "auto": "7.8"
        }
    }
    
    context = writer_agent._generate_market_context(live_data)
    assert isinstance(context, str)
    assert any(stock in context for stock in ["AAPL", "GOOGL"])
    assert "tasa" in context.lower()

@pytest.mark.asyncio
async def test_recommendation_personalization(writer_agent):
    """Test recommendation personalization based on user level"""
    recommendations = [
        "Invertir en acciones tecnológicas",
        "Diversificar la cartera",
        "Considerar instrumentos derivados",
        "Mantener un fondo de emergencia"
    ]
    
    # Test for beginner level
    beginner_recs = writer_agent._personalize_recommendations(recommendations, "beginner")
    assert len(beginner_recs) <= len(recommendations)
    assert "fondo de emergencia" in ' '.join(beginner_recs).lower()
    
    # Test for advanced level
    advanced_recs = writer_agent._personalize_recommendations(recommendations, "advanced")
    assert "derivados" in ' '.join(advanced_recs).lower()

@pytest.mark.asyncio
async def test_educational_content_integration(writer_agent):
    """Test educational content integration"""
    analysis = {
        "key_findings": ["El mercado muestra tendencia alcista"],
        "educational_content": {
            "concepts": ["tendencia alcista", "soporte y resistencia"],
            "examples": ["Cuando el precio supera resistencias previas"],
            "resources": ["Guía básica de análisis técnico"]
        }
    }
    user_context = {"user_profile": {"experience": "beginner"}}
    
    response = await writer_agent.compose_response(analysis, user_context)
    assert isinstance(response, str)
    assert any(concept in response.lower() for concept in ["tendencia", "soporte"])
    assert "ejemplo" in response.lower()

@pytest.mark.asyncio
async def test_full_response_generation(writer_agent):
    """Test complete response generation pipeline"""
    analysis = {
        "key_findings": [
            "El índice Merval subió un 2.5%",
            "La inflación mensual fue del 3.8%"
        ],
        "short_term_implications": [
            "Mayor presión sobre el tipo de cambio",
            "Posible ajuste en tasas de interés"
        ],
        "recommended_actions": [
            "Diversificar inversiones",
            "Considerar cobertura cambiaria"
        ],
        "live_data": {
            "stocks": {
                "GGAL": {"Global Quote": {"05. price": "280.5", "10. change percent": "+1.8%"}}
            },
            "forex": {"conversion_rate": "850.50"}
        }
    }
    
    user_context = {
        "intent": {
            "main_topic": "market_analysis",
            "complexity": "medium"
        },
        "user_profile": {
            "experience": "intermediate",
            "previous_queries": ["Análisis del mercado local"]
        }
    }
    
    response = await writer_agent.compose_response(analysis, user_context)
    
    # Verify response structure and content
    assert isinstance(response, str)
    assert len(response) > 0
    assert "Merval" in response
    assert "inflación" in response.lower()
    assert "recomend" in response.lower()
    assert "%" in response

@pytest.mark.asyncio
async def test_process_method(writer_agent):
    """Test the main process method"""
    input_data = {
        "analysis": {
            "key_findings": ["Test finding"],
            "short_term_implications": ["Test implication"],
            "recommended_actions": ["Test action"]
        },
        "context": {
            "intent": {"main_topic": "test"},
            "user_profile": {"experience": "intermediate"}
        }
    }
    
    result = await writer_agent.process(input_data)
    assert isinstance(result, dict)
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
