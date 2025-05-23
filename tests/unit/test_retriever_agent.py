import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.agents.retriever_agent import InstructorXLRetrieverAgent

@pytest.fixture
def retriever_agent():
    with patch.dict('os.environ', {
        'ALPHA_VANTAGE_KEY': 'test_key',
        'EXCHANGERATE_API_KEY': 'test_key'
    }):
        agent = InstructorXLRetrieverAgent()
        return agent

@pytest.mark.asyncio
async def test_cache_system(retriever_agent):
    """Test the data caching system"""
    # Test cache key generation
    cache_key = retriever_agent._generate_cache_key("stock_data", {"symbol": "AAPL"})
    assert isinstance(cache_key, str)
    assert len(cache_key) > 0
    
    # Test cache validity
    now = datetime.now()
    old_time = now - timedelta(hours=3)
    
    # Should be invalid for stock data (TTL = 1 hour)
    assert not retriever_agent._is_cache_valid("stock_data", old_time)
    
    # Should be valid for educational content (TTL = 24 hours)
    assert retriever_agent._is_cache_valid("educational_content", old_time)

@pytest.mark.asyncio
async def test_concurrent_data_fetching(retriever_agent):
    """Test concurrent data fetching with asyncio"""
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    # Mock the API calls
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_get.return_value.__aenter__.return_value.json = Mock(
            return_value={"Global Quote": {"05. price": "180.5"}}
        )
        
        results = await retriever_agent._fetch_concurrent_stock_data(symbols)
        
        assert isinstance(results, dict)
        assert len(results) == len(symbols)
        assert all(symbol in results for symbol in symbols)

@pytest.mark.asyncio
async def test_fallback_data(retriever_agent):
    """Test fallback data handling"""
    # Test with failed API call
    with patch('aiohttp.ClientSession.get', side_effect=Exception("API Error")):
        result = await retriever_agent._get_market_data_with_fallback("AAPL")
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "fallback_data" in result
        assert result["fallback_data"] is not None

@pytest.mark.asyncio
async def test_educational_content(retriever_agent):
    """Test educational content retrieval and adaptation"""
    # Test for beginner level
    beginner_content = retriever_agent._get_educational_content(
        main_topic="stocks",
        subtopics=[],
        user_level="beginner"
    )
    assert isinstance(beginner_content, dict)
    assert "stocks" in beginner_content
    assert "concepts" in beginner_content["stocks"]
    assert "examples" in beginner_content["stocks"]
    assert len(beginner_content["stocks"]["concepts"]) > 0
    
    # Test for advanced level
    advanced_content = retriever_agent._get_educational_content(
        main_topic="stocks",
        subtopics=[],
        user_level="advanced"
    )
    assert isinstance(advanced_content, dict)
    assert "stocks" in advanced_content
    assert "concepts" in advanced_content["stocks"]
    assert advanced_content["stocks"] != beginner_content["stocks"]

@pytest.mark.asyncio
async def test_market_sentiment_analysis(retriever_agent):
    """Test market sentiment analysis"""
    market_data = {
        "indices": {
            "MERVAL": {"change": "+1.2%"},
            "S&P500": {"change": "+0.8%"}
        },
        "interest_rates": {"policy_rate": "118%"},
        "inflation": {"monthly": "4.5%"}
    }
    
    sentiment = retriever_agent._analyze_market_sentiment(market_data)
    assert isinstance(sentiment, dict)
    assert "score" in sentiment
    assert "factors" in sentiment
    assert isinstance(sentiment["score"], float)
    assert -1 <= sentiment["score"] <= 1

@pytest.mark.asyncio
async def test_data_freshness_monitoring(retriever_agent):
    """Test data freshness monitoring"""
    data = {
        "price": "100.50",
        "timestamp": datetime.now().isoformat()
    }
    
    freshness = retriever_agent._check_data_freshness(data)
    assert isinstance(freshness, dict)
    assert "is_fresh" in freshness
    assert "age_seconds" in freshness
    assert isinstance(freshness["is_fresh"], bool)

@pytest.mark.asyncio
async def test_error_reporting(retriever_agent):
    """Test error reporting system"""
    error_info = retriever_agent._create_error_report(
        error_type="API_ERROR",
        message="Failed to fetch data",
        context={"symbol": "AAPL"}
    )
    
    assert isinstance(error_info, dict)
    assert "error_type" in error_info
    assert "message" in error_info
    assert "timestamp" in error_info
    assert "context" in error_info

@pytest.mark.asyncio
async def test_response_structure(retriever_agent):
    """Test response structure and organization"""
    query = "análisis del dólar"
    
    with patch.object(retriever_agent, '_fetch_market_data') as mock_fetch:
        mock_fetch.return_value = {
            "forex": {"USD/ARS": "850.50"},
            "interest_rates": {"policy_rate": "118%"}
        }
        
        response = await retriever_agent.retrieve(query)
        
        assert isinstance(response, dict)
        assert "information" in response
        assert "market_data" in response["information"]
        assert "educational_content" in response["information"]
        assert "metadata" in response

@pytest.mark.asyncio
async def test_spanish_news_integration(retriever_agent):
    """Test Spanish news integration"""
    with patch.object(retriever_agent, '_fetch_spanish_news') as mock_news:
        mock_news.return_value = [
            {
                "title": "Análisis del mercado argentino",
                "summary": "Las principales acciones...",
                "url": "http://example.com/news/1"
            }
        ]
        
        news = await retriever_agent._get_relevant_news("mercado")
        assert isinstance(news, list)
        assert len(news) > 0
        assert all(isinstance(item, dict) for item in news)
        assert all("title" in item for item in news)

@pytest.mark.asyncio
async def test_full_retrieval_pipeline(retriever_agent):
    """Test the complete data retrieval pipeline"""
    query = "¿Cómo está el dólar hoy?"
    
    # Mock all external API calls
    with patch.multiple(retriever_agent,
                       _fetch_market_data=Mock(return_value={
                           "forex": {"USD/ARS": "850.50"},
                           "interest_rates": {"policy_rate": "118%"}
                       }),
                       _fetch_spanish_news=Mock(return_value=[
                           {"title": "Análisis del dólar", "summary": "El dólar continúa..."}
                       ])):
        
        result = await retriever_agent.retrieve(query)
        
        # Verify response structure
        assert isinstance(result, dict)
        assert "information" in result
        assert "market_data" in result["information"]
        assert "educational_content" in result["information"]
        assert "metadata" in result
        
        # Verify market data
        market_data = result["information"]["market_data"]
        assert "forex" in market_data
        assert isinstance(market_data["forex"], dict)
        
        # Verify educational content
        edu_content = result["information"]["educational_content"]
        assert isinstance(edu_content, dict)
        assert any(key in edu_content for key in ["forex", "currency", "dollar"])
        
        # Verify metadata
        assert "cache_status" in result["metadata"]
        assert "data_freshness" in result["metadata"]
        assert "retrieval_timestamp" in result["metadata"]
