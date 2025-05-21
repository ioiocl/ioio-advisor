import pytest
import httpx
import json
from datetime import datetime
from typing import List, Dict, Any

@pytest.mark.asyncio
async def test_live_market_data():
    """Test that responses include current market data."""
    query = "Como estan las acciones de tecnologia hoy?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response includes today's date
        today = datetime.now().strftime("%Y")
        assert today in data["text"], "Response should include current year"
        
        # Verify response includes tech company tickers
        text = data["text"].lower()
        assert any(ticker in text for ticker in ["aapl", "googl", "msft"]), \
            "Response should mention major tech companies"
        
        # Verify response includes numerical data
        assert any(char.isdigit() for char in text), \
            "Response should include numerical values"

@pytest.mark.asyncio
async def test_live_currency_rates():
    """Test that responses include current currency exchange rates."""
    query = "Cual es el precio del dolar hoy?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        text = data["text"].lower()
        
        # Verify response includes currency terms
        assert any(currency in text for currency in ["dolar", "usd", "$"]), \
            "Response should mention USD"
        
        # Verify response includes exchange rate data
        assert any(char.isdigit() for char in text), \
            "Response should include numerical exchange rate"
        
        # Verify response is current
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in data["created_at"], \
            "Response should be from today"

@pytest.mark.asyncio
async def test_live_interest_rates():
    """Test that responses include current interest rates."""
    query = "Cuales son las tasas de interes actuales para prestamos?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        text = data["text"].lower()
        
        # Verify response includes interest rate terms
        assert any(term in text for term in ["tasa", "interes", "anual", "mensual"]), \
            "Response should mention interest rates"
        
        # Verify response includes percentage values
        assert "%" in text, "Response should include percentage values"
        
        # Verify different types of loans are mentioned
        assert any(loan_type in text for loan_type in ["hipoteca", "personal", "auto"]), \
            "Response should mention different loan types"

@pytest.mark.asyncio
async def test_live_inflation_data():
    """Test that responses include current inflation data."""
    query = "Cual es la tasa de inflacion actual?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        text = data["text"].lower()
        
        # Verify response includes inflation terms
        assert any(term in text for term in ["inflacion", "inflación"]), \
            "Response should mention inflation"
        
        # Verify response includes percentage
        assert "%" in text, "Response should include percentage values"
        
        # Verify response includes trend
        assert any(term in text for term in ["tendencia", "alza", "baja"]), \
            "Response should describe inflation trend"
        
        # Verify response includes categories
        assert any(category in text for category in ["alimentos", "vivienda", "transporte"]), \
            "Response should break down inflation by category"

@pytest.mark.asyncio
async def test_live_investment_recommendations():
    """Test that responses include current investment recommendations."""
    query = "Donde me conviene invertir ahora mismo?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        text = data["text"].lower()
        
        # Verify response includes investment vehicles
        assert any(vehicle in text for vehicle in ["acciones", "bonos", "fondos"]), \
            "Response should mention investment vehicles"
        
        # Verify response includes risk levels
        assert any(risk in text for risk in ["riesgo", "volatilidad", "seguro"]), \
            "Response should discuss risk levels"
        
        # Verify response includes return expectations
        assert any(term in text for term in ["rendimiento", "retorno", "ganancia"]), \
            "Response should mention expected returns"
        
        # Verify response includes timeframes
        assert any(term in text for term in ["corto plazo", "largo plazo", "mediano plazo"]), \
            "Response should discuss investment timeframes"

@pytest.mark.asyncio
async def test_real_time_response_latency():
    """Test that responses are generated within acceptable timeframes."""
    queries = [
        "Como esta la bolsa hoy?",
        "Cual es el precio del dolar?",
        "Cuanto esta la inflacion?",
        "Que acciones conviene comprar?",
        "Cuales son las tasas de prestamos?"
    ]
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        for query in queries:
            start_time = datetime.now()
            
            response = await client.post(
                "/query",
                json={"query": query}
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Verify response time is under 5 seconds
            assert duration < 5, f"Query '{query}' took too long ({duration:.2f}s)"
            
            # Verify response is valid
            assert response.status_code == 200
            data = response.json()
            assert len(data["text"]) >= 50, "Response should be sufficiently detailed"
            assert data["visualization_url"], "Response should include visualization"
