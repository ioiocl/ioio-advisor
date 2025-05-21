import pytest
import httpx
from datetime import datetime
import json
import unicodedata

def normalize_text(text):
    """Normalize text by removing accents and converting to lowercase."""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').lower()

@pytest.mark.asyncio
async def test_stock_market_visualization():
    """Test that stock market queries generate appropriate visualizations."""
    query = "Como van las acciones de Apple y Microsoft hoy?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify image was generated
        assert "visualization_url" in data
        assert data["visualization_url"] is not None
        assert data["visualization_url"].startswith("/static/images")
        
        # Verify response includes stock-related content
        text = data["text"].lower()
        assert any(term in text for term in ["aapl", "msft", "acciones"])

@pytest.mark.asyncio
async def test_currency_visualization():
    """Test that currency queries generate appropriate visualizations."""
    query = "Cual es el tipo de cambio del dólar al euro?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify image was generated
        assert "visualization_url" in data
        assert data["visualization_url"] is not None
        assert data["visualization_url"].startswith("/static/images")
        
        # Verify response includes currency-related content
        text = data["text"].lower()
        assert any(term in text for term in ["dólar", "euro", "cambio"])

@pytest.mark.asyncio
async def test_inflation_visualization():
    """Test that inflation queries generate appropriate visualizations."""
    query = "Como ha evolucionado la inflacion este anio?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify image was generated
        assert "visualization_url" in data
        assert data["visualization_url"] is not None
        assert data["visualization_url"].startswith("/static/images")
        
        # Verify response includes inflation-related content
        text = normalize_text(data["text"])
        assert any(term in text for term in ["inflacion", "tasa", "categorias"])

@pytest.mark.asyncio
async def test_investment_visualization():
    """Test that investment queries generate appropriate visualizations."""
    query = "Cuales son las mejores opciones de inversion ahora?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify image was generated
        assert "visualization_url" in data
        assert data["visualization_url"] is not None
        assert data["visualization_url"].startswith("/static/images")
        
        # Verify response includes investment-related content
        text = normalize_text(data["text"])
        assert any(term in text for term in ["inversion", "riesgo", "rendimiento", "plazo"])

@pytest.mark.asyncio
async def test_interest_rate_visualization():
    """Test that interest rate queries generate appropriate visualizations."""
    query = "Cuales son las tasas de interes para hipotecas?"
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify image was generated
        assert "visualization_url" in data
        assert data["visualization_url"] is not None
        assert data["visualization_url"].startswith("/static/images")
        
        # Verify response includes interest rate-related content
        text = data["text"].lower()
        assert any(term in text for term in ["tasa", "interés", "hipoteca", "préstamo"])

@pytest.mark.asyncio
async def test_visualization_consistency():
    """Test that each topic generates a consistent visualization."""
    queries = [
        ("Como van las acciones de Google?", ["acciones", "bolsa", "mercado", "google"]),
        ("Cual es el precio del euro?", ["dolar", "euro", "cambio"]),
        ("Cual es la inflacion actual?", ["inflacion", "precios", "aumento"]),
        ("Donde invertir mi dinero?", ["inversion", "riesgo", "rendimiento"]),
        ("Tasas de prestamos personales?", ["tasa", "interes", "prestamo"])
    ]
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        for query, expected_terms in queries:
            response = await client.post(
                "/query",
                json={"query": query}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify visualization was generated
            assert "visualization_url" in data
            assert data["visualization_url"] is not None
            assert data["visualization_url"].startswith("/static/images")
            
            # Verify response includes topic-specific terms
            text = normalize_text(data["text"])
            normalized_terms = [normalize_text(term) for term in expected_terms]
            assert any(term in text for term in normalized_terms), \
                f"Response should include terms related to the query: {query}"
