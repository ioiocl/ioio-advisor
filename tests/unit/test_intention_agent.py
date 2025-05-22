import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from src.agents.intention_agent import Phi3IntentionAgent

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    return tokenizer

@pytest.fixture
def mock_model():
    model = Mock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

@pytest.fixture
def intention_agent():
    """Create a test instance of Phi3IntentionAgent."""
    agent = Phi3IntentionAgent()
    
    # Mock tokenizer
    mock_tokenizer_output = MagicMock()
    mock_tokenizer_output.to = MagicMock(return_value=mock_tokenizer_output)
    agent.tokenizer = MagicMock(return_value=mock_tokenizer_output)
    agent.tokenizer.decode = MagicMock(return_value='{"main_topic": "currency", "subtopics": ["economy"], "intention": "understand_impact", "confidence": 0.95}')
    
    # Mock model
    mock_model = MagicMock()
    mock_model.device = 'cuda'
    mock_model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    agent.model = mock_model
    
    return agent

@pytest.mark.asyncio
async def test_currency_topic_detection(intention_agent):
    """Test enhanced currency topic detection"""
    # Test USD detection
    usd_query = "¿Cuál es el precio del usd blue?"
    result = await intention_agent.detect_intent(usd_query)
    assert result["main_topic"] == "currency"
    assert "usd" in result["subtopics"]
    
    # Test EUR detection
    eur_query = "¿Cómo está el euro hoy?"
    result = await intention_agent.detect_intent(eur_query)
    assert result["main_topic"] == "currency"
    assert "eur" in result["subtopics"]
    
    # Test exchange rate phrases
    exchange_query = "¿Cuál es el tipo de cambio actual?"
    result = await intention_agent.detect_intent(exchange_query)
    assert result["main_topic"] == "currency"
    assert "exchange_rate" in result["subtopics"]

@pytest.mark.asyncio
async def test_query_type_detection(intention_agent):
    """Test enhanced query type detection"""
    # Test 'understand_reason'
    reason_query = "¿Qué significa la devaluación del peso?"
    result = await intention_agent.detect_intent(reason_query)
    assert result["intention"] == "understand_reason"
    
    # Test 'get_analysis'
    analysis_query = "Analiza el impacto de la inflación"
    result = await intention_agent.detect_intent(analysis_query)
    assert result["intention"] == "get_analysis"
    
    # Test explanation request
    explain_query = "Explica cómo funciona el mercado de divisas"
    result = await intention_agent.detect_intent(explain_query)
    assert result["intention"] == "get_analysis"

@pytest.mark.asyncio
async def test_confidence_calculation(intention_agent):
    """Test improved confidence calculation"""
    # High confidence case
    high_conf_query = "¿Cuál es el precio del dólar blue?"
    result = await intention_agent.detect_intent(high_conf_query)
    assert result["confidence"] > 0.8
    assert "confidence_factors" in result
    
    # Lower confidence case
    low_conf_query = "¿Qué está pasando con la economía?"
    result = await intention_agent.detect_intent(low_conf_query)
    assert "confidence" in result
    assert "confidence_factors" in result

@pytest.mark.asyncio
async def test_topic_validation(intention_agent):
    """Test enhanced topic validation"""
    # Test with clear financial topic
    financial_query = "¿Cómo invertir en el mercado de valores?"
    result = await intention_agent.detect_intent(financial_query)
    assert result["topic_validation"]["is_financial"] is True
    assert result["topic_validation"]["confidence"] > 0.8
    
    # Test with ambiguous topic
    ambiguous_query = "¿Qué pasa con los precios?"
    result = await intention_agent.detect_intent(ambiguous_query)
    assert "topic_validation" in result
    assert "confidence" in result["topic_validation"]

@pytest.mark.asyncio
async def test_subtopic_handling(intention_agent):
    """Test improved subtopic handling"""
    # Test multiple subtopics
    complex_query = "¿Cómo afecta la inflación al dólar y las acciones?"
    result = await intention_agent.detect_intent(complex_query)
    assert len(result["subtopics"]) >= 2
    assert "inflation" in result["subtopics"]
    assert "currency" in result["subtopics"] or "usd" in result["subtopics"]
    assert "stocks" in result["subtopics"]

@pytest.mark.asyncio
async def test_text_normalization(intention_agent):
    """Test normalized text handling"""
    # Test with accented characters
    accented_query = "¿Cuál es la cotización del dólar?"
    result = await intention_agent.detect_intent(accented_query)
    assert result["main_topic"] == "currency"
    
    # Test with mixed case
    mixed_case_query = "CóMo EsTá eL DoLaR"
    result = await intention_agent.detect_intent(mixed_case_query)
    assert result["main_topic"] == "currency"
    
    # Test with special characters
    special_chars_query = "$$$¿¿¿dólar???$$$"
    result = await intention_agent.detect_intent(special_chars_query)
    assert result["main_topic"] == "currency"

@pytest.mark.asyncio
async def test_full_intent_detection_pipeline(intention_agent):
    """Test the complete intent detection pipeline"""
    test_queries = [
        {
            "query": "¿A cuánto está el dólar blue hoy?",
            "expected": {
                "main_topic": "currency",
                "subtopics": ["usd", "informal_market"],
                "intention": "get_price"
            }
        },
        {
            "query": "¿Por qué sube la inflación?",
            "expected": {
                "main_topic": "inflation",
                "intention": "understand_reason"
            }
        },
        {
            "query": "Analiza el impacto de las tasas en el mercado",
            "expected": {
                "main_topic": "interest_rates",
                "subtopics": ["market_impact"],
                "intention": "get_analysis"
            }
        }
    ]
    
    for test_case in test_queries:
        result = await intention_agent.detect_intent(test_case["query"])
        
        # Verify main topic
        assert result["main_topic"] == test_case["expected"]["main_topic"]
        
        # Verify intention
        assert result["intention"] == test_case["expected"]["intention"]
        
        # Verify subtopics if specified
        if "subtopics" in test_case["expected"]:
            assert all(topic in result["subtopics"] 
                      for topic in test_case["expected"]["subtopics"])
        
        # Verify required fields
        assert "confidence" in result
        assert "topic_validation" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
