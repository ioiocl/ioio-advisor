import pytest
import asyncio
from datetime import datetime
from src.agents.reason_agent import Mistral7BReasonAgent

@pytest.mark.asyncio
async def test_reason_agent_analysis():
    # Initialize the agent
    agent = Mistral7BReasonAgent()
    
    # Sample test cases with different scenarios
    test_cases = [
        {
            "name": "Currency Impact Query",
            "query": "¿Cómo afecta la inflación al tipo de cambio?",
            "retrieved_info": {
                "information": {
                    "market_data": {
                        "indices": {
                            "USD/ARS": {"value": "850.50", "change": "+2.5%"},
                            "EUR/ARS": {"value": "920.30", "change": "+1.8%"}
                        },
                        "interest_rates": {
                            "savings": "8.5%",
                            "mortgage": "12.3%"
                        },
                        "inflation": {
                            "current_rate": "5.2%",
                            "food": "6.1%",
                            "housing": "4.8%"
                        }
                    },
                    "educational_content": {
                        "currency": {
                            "concepts": [
                                "La inflación afecta directamente al valor de la moneda",
                                "El tipo de cambio refleja la fortaleza económica"
                            ],
                            "examples": [
                                "Cuando la inflación sube, la moneda local tiende a debilitarse"
                            ]
                        }
                    }
                },
                "market_sentiment": "negativo"
            }
        },
        {
            "name": "Investment Strategy Query",
            "query": "¿Qué inversiones son recomendables con la situación actual del mercado?",
            "retrieved_info": {
                "information": {
                    "market_data": {
                        "indices": {
                            "MERVAL": {"value": "950000", "change": "+1.2%"},
                            "S&P500": {"value": "4850", "change": "+0.8%"}
                        },
                        "interest_rates": {
                            "savings": "4.5%",
                            "bonds": "6.2%"
                        },
                        "inflation": {
                            "current_rate": "3.8%"
                        }
                    },
                    "educational_content": {
                        "investment": {
                            "concepts": [
                                "Diversificación reduce el riesgo",
                                "Inversiones defensivas en mercados volátiles"
                            ],
                            "tips": [
                                "Considerar plazos fijos UVA para proteger contra inflación"
                            ]
                        }
                    }
                },
                "market_sentiment": "neutral"
            }
        }
    ]
    
    # Run tests
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        
        try:
            # Analyze the query
            result = await agent.analyze(
                test_case["query"],
                test_case["retrieved_info"]
            )
            
            # Print analysis results
            print("\nAnálisis:")
            print("=========")
            
            if "error" in result:
                print(f"Error: {result['error']}")
                if "details" in result:
                    print(f"Details: {result['details']}")
                continue
            
            # Print key findings
            print("\nHallazgos clave:")
            for finding in result.get("hallazgos_clave", []):
                print(f"- {finding}")
            
            # Print implications
            print("\nImplicaciones a corto plazo:")
            for imp in result.get("implicaciones_corto_plazo", []):
                print(f"- {imp}")
            
            # Print recommendations
            print("\nAcciones recomendadas:")
            for action in result.get("acciones_recomendadas", []):
                print(f"- {action}")
            
            # Print metadata
            meta = result.get("metadata", {})
            print("\nMetadata:")
            print(f"- Nivel de riesgo: {meta.get('risk_score', 'N/A')}")
            print(f"- Calidad de datos: {meta.get('data_quality', 'N/A')}")
            
        except Exception as e:
            print(f"Error executing test case: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_reason_agent_analysis())
