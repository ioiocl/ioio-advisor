import asyncio
import json
import aiohttp
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

async def test_query_endpoint():
    """Test the /query endpoint with various inputs."""
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8000/query"
        
        # Test cases
        test_cases = [
            {
                "name": "Basic query",
                "payload": {
                    "query": "¿Cuál es la tasa de interés actual?",
                    "context": {}
                }
            },
            {
                "name": "Investment query",
                "payload": {
                    "query": "¿Dónde debo invertir mi dinero?",
                    "context": {
                        "risk_profile": "moderate"
                    }
                }
            },
            {
                "name": "Inflation query",
                "payload": {
                    "query": "¿Cómo está la inflación?",
                    "context": {
                        "timeframe": "current"
                    }
                }
            }
        ]
        
        print("\nStarting API tests...")
        for test in test_cases:
            print(f"\nTesting: {test['name']}")
            print(f"Request payload: {json.dumps(test['payload'], indent=2)}")
            
            try:
                async with session.post(url, json=test["payload"]) as response:
                    status = response.status
                    response_text = await response.text()
                    
                    print(f"Status code: {status}")
                    print(f"Response: {response_text}")
                    
                    assert status == 200, f"Expected status 200, got {status}"
                    response_data = json.loads(response_text)
                    assert "text" in response_data, "Response missing 'text' field"
                    
                    print("[PASS] Test passed")
                    
            except Exception as e:
                print(f"[FAIL] Test failed: {str(e)}")
                print(f"[DEBUG] Server response: {response_text if 'response_text' in locals() else 'No response'}")

if __name__ == "__main__":
    asyncio.run(test_query_endpoint())
