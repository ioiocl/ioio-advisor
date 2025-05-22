import asyncio
import json
from datetime import datetime, UTC
from typing import Dict, Any
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set required environment variables
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_API_KEY', '')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
os.environ['STABILITY_API_KEY'] = os.getenv('STABILITY_API_KEY', '')

# Set model paths
os.environ['PHI3_MODEL_PATH'] = os.getenv('PHI3_MODEL_PATH', 'microsoft/phi-3')
os.environ['INSTRUCTOR_MODEL_PATH'] = os.getenv('INSTRUCTOR_MODEL_PATH', 'hkunlp/instructor-xl')
os.environ['MISTRAL_MODEL_PATH'] = os.getenv('MISTRAL_MODEL_PATH', 'mistralai/Mistral-7B-v0.1')

from src.agents.intention_agent import Phi3IntentionAgent
from src.agents.retriever_agent import InstructorXLRetrieverAgent
from src.agents.reason_agent import Mistral7BReasonAgent
from src.agents.writer_agent import GPT4WriterAgent
from src.agents.designer_agent import StableDiffusionDesignerAgent

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

def print_step(step: str, data: Dict[str, Any]) -> None:
    """Print a step's input/output in a clear format"""
    print(f"\n{'='*80}")
    print(f"STEP: {step}")
    print(f"{'='*80}")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"{'='*80}\n")

async def run_financial_assistant(query: str, user_level: str = "beginner") -> None:
    """Run the financial assistant pipeline with clear step outputs"""
    
    print(f"\n\nPROCESANDO CONSULTA: {query}\n")
    print("="*80)

    try:
        # Initialize agents
        intention_agent = Phi3IntentionAgent()
        retriever_agent = InstructorXLRetrieverAgent()
        reason_agent = Mistral7BReasonAgent()
        writer_agent = GPT4WriterAgent()
        designer_agent = StableDiffusionDesignerAgent()

        # Initialize context
        context = {
            "query_id": str(hash(query)),
            "timestamp": datetime.now(UTC).isoformat(),
            "user_level": user_level,
            "language": "es",
            "user_profile": {
                "experience": user_level,
                "previous_queries": []
            }
        }

        # Step 1: Intention Detection
        print_step("1. INTENTION DETECTION - Input", {"query": query, "context": context})
        intention_output = await intention_agent.process({"query": query, "context": context})
        print_step("1. INTENTION DETECTION - Output", intention_output)

        # Step 2: Data Retrieval
        retriever_input = {
            "query": query,
            "context": context,
            **intention_output
        }
        print_step("2. DATA RETRIEVAL - Input", retriever_input)
        retriever_output = await retriever_agent.process(retriever_input)
        print_step("2. DATA RETRIEVAL - Output", retriever_output)

        # Step 3: Analysis and Reasoning
        reason_input = {
            "query": query,
            "context": context,
            **intention_output,
            **retriever_output
        }
        print_step("3. ANALYSIS AND REASONING - Input", reason_input)
        reason_output = await reason_agent.process(reason_input)
        print_step("3. ANALYSIS AND REASONING - Output", reason_output)

        # Step 4: Response Generation
        writer_input = {
            "query": query,
            "context": context,
            **reason_output
        }
        print_step("4. RESPONSE GENERATION - Input", writer_input)
        writer_output = await writer_agent.process(writer_input)
        print_step("4. RESPONSE GENERATION - Output", writer_output)

        # Step 5: Visualization
        designer_input = {
            "query": query,
            "context": context,
            **writer_output
        }
        print_step("5. VISUALIZATION - Input", designer_input)
        designer_output = await designer_agent.process(designer_input)
        print_step("5. VISUALIZATION - Output", designer_output)

        # Final Response
        print("\nRESPUESTA FINAL:")
        print("-" * 40)
        print(writer_output["response"])
        if "visualization_url" in designer_output:
            print(f"\nVisualización: {designer_output['visualization_url']}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Respuesta alternativa: Lo siento, ha ocurrido un error. Por favor, intenta tu consulta nuevamente.")

async def main():
    """Main function to run test queries"""
    test_queries = [
        "¿Qué es la inflación y cómo me afecta?",
        "¿Cómo funciona el mercado de acciones para principiantes?",
        "¿Por qué sube el dólar y qué significa para mi bolsillo?"
    ]

    for query in test_queries:
        await run_financial_assistant(query)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
