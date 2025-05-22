import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.agents.designer_agent import (
    StableDiffusionDesignerAgent,
    ChartType,
    VisualizationStyle,
    ChartConfig
)

@pytest.fixture
def designer_agent():
    with patch.dict('os.environ', {
        'STABILITY_API_KEY': 'test_key',
        'OPENAI_API_KEY': 'test_key'
    }):
        agent = StableDiffusionDesignerAgent()
        return agent

@pytest.mark.asyncio
async def test_chart_config_initialization(designer_agent):
    """Test the initialization of chart configurations"""
    # Test stock chart config
    stock_config = designer_agent.chart_templates["stock"]
    assert isinstance(stock_config, ChartConfig)
    assert stock_config.type == ChartType.CANDLESTICK
    assert "bollinger_bands" in stock_config.elements
    assert "RSI" in stock_config.indicators
    assert stock_config.interactive is True

    # Test currency chart config
    currency_config = designer_agent.chart_templates["currency"]
    assert currency_config.type == ChartType.LINE
    assert "volatility_bands" in currency_config.elements
    assert "momentum" in currency_config.indicators

@pytest.mark.asyncio
async def test_cache_system(designer_agent):
    """Test the visualization caching system"""
    # Mock context and text
    context = {"intent": {"main_topic": "stock"}}
    text = "Test visualization text"
    
    # Generate cache key
    cache_key = designer_agent._get_cache_key(context, text)
    
    # Test cache key generation
    assert isinstance(cache_key, str)
    assert len(cache_key) > 0
    
    # Test cache validity
    designer_agent._cache_timestamps[cache_key] = datetime.now()
    assert designer_agent._is_cache_valid(cache_key) is True
    
    # Test cache expiration
    designer_agent._cache_timestamps[cache_key] = datetime.now() - timedelta(hours=2)
    assert designer_agent._is_cache_valid(cache_key) is False

@pytest.mark.asyncio
async def test_chart_style_caching(designer_agent):
    """Test the chart style caching mechanism"""
    # Get style multiple times for the same topic
    style1 = designer_agent._get_chart_style("stock", "intermediate")
    style2 = designer_agent._get_chart_style("stock", "intermediate")
    
    # Should be the same object due to LRU cache
    assert style1 is style2
    
    # Test different topics
    stock_style = designer_agent._get_chart_style("stock", "advanced")
    currency_style = designer_agent._get_chart_style("currency", "advanced")
    assert stock_style != currency_style

@pytest.mark.asyncio
async def test_error_handling(designer_agent):
    """Test error handling in the process method"""
    # Test with invalid input
    result = await designer_agent.process({"invalid": "data"})
    assert result["visualization"] is None
    assert "error" in result
    assert "message" in result["error"]
    assert "timestamp" in result["error"]

@pytest.mark.asyncio
async def test_data_extraction(designer_agent):
    """Test data point extraction from text"""
    text = """El índice subió un 2.5% alcanzando 1500 puntos.
              La inflación fue del 3.8% en el último mes.
              El dólar cerró a 850.50 pesos."""
    
    data_points = designer_agent._extract_data_points(text)
    
    assert len(data_points["values"]) >= 2  # Should find 1500 and 850.50
    assert len(data_points["percentages"]) >= 2  # Should find 2.5% and 3.8%
    assert len(data_points["trends"]) >= 1  # Should find "subió"

@pytest.mark.asyncio
async def test_topic_detection(designer_agent):
    """Test financial topic detection"""
    # Test stock detection
    stock_text = "AAPL cerró con una subida del 2%"
    assert designer_agent._detect_topic_from_text(stock_text) == "stock"
    
    # Test currency detection
    currency_text = "El dólar se fortaleció frente al euro"
    assert designer_agent._detect_topic_from_text(currency_text) == "currency"
    
    # Test inflation detection
    inflation_text = "La inflación mensual alcanzó el 5%"
    assert designer_agent._detect_topic_from_text(inflation_text) == "inflation"

@pytest.mark.asyncio
async def test_chart_data_generation(designer_agent):
    """Test chart data generation"""
    topic = "investment"
    data_points = {
        "values": [{"value": 100, "context": "initial"}, {"value": 150, "context": "final"}],
        "percentages": [{"value": 5.2, "context": "return"}],
        "trends": [{"direction": "up", "context": "growth"}],
        "categories": ["Stocks", "Bonds"],
        "time_periods": ["2024", "2025"]
    }
    
    chart_data = designer_agent._generate_chart_data(topic, data_points)
    
    assert isinstance(chart_data, dict)
    assert "type" in chart_data
    assert "data" in chart_data or "series" in chart_data
    assert "colors" in chart_data
    assert "indicators" in chart_data

@pytest.mark.asyncio
async def test_style_metadata(designer_agent):
    """Test style metadata generation"""
    topics = ["stock", "currency", "interest", "inflation", "investment"]
    
    for topic in topics:
        style = designer_agent._get_style_metadata(topic)
        assert "color_scheme" in style
        assert "layout" in style
        assert "design_elements" in style

@pytest.mark.asyncio
async def test_dalle_prompt_generation(designer_agent):
    """Test DALL-E prompt generation"""
    topic = "investment"
    data_points = {
        "values": [{"value": 100, "context": "investment"}],
        "trends": [{"direction": "up", "context": "growth"}]
    }
    chart_data = {
        "indicators": {"trend": ["up", "up"]}
    }
    
    prompt = designer_agent._generate_dalle_prompt(topic, data_points, chart_data)
    
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "investment" in prompt.lower()
    assert "growth" in prompt.lower() or "upward" in prompt.lower()

@pytest.mark.asyncio
async def test_visualization_generation_integration(designer_agent):
    """Test the complete visualization generation pipeline"""
    with patch.object(designer_agent, '_call_stability_api') as mock_api:
        # Mock API response
        mock_api.return_value = b"test_image_data"
        
        context = {"intent": {"main_topic": "stock"}}
        text = "AAPL subió un 2% alcanzando $200 por acción"
        
        result = await designer_agent.generate_visualization(context, text)
        
        assert isinstance(result, dict)
        assert "chart_url" in result
        assert "image_url" in result
        assert "metadata" in result
        
        metadata = result["metadata"]
        assert metadata["topic"] == "stock"
        assert "chart" in metadata
        assert "dalle_image" in metadata
