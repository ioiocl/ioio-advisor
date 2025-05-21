# IOIO Financial AI Agent System

A sophisticated AI agent system that provides clear and simple financial information to users using multiple specialized AI models and Hexagonal Architecture.

## Requirements

- Python 3.13.2 or higher
- Poetry for dependency management
- API keys for:
  - OpenAI (GPT-4)
  - Stability AI (Stable Diffusion)
  - HuggingFace (for Phi-3 and Mistral-7B)
  - Alpha Vantage (optional, for market data)
  - ExchangeRate API (optional, for currency data)

## Architecture

The system follows Hexagonal Architecture (DDD) principles and consists of 5 main agents:

1. **Intention Agent** (Phi-3 Mini)
   - Processes and understands user queries
   - Detects user intentions and financial topics

2. **Retriever Agent** (Instructor XL)
   - Retrieves relevant information from various sources
   - Handles API calls and document processing

3. **Reason Agent** (Mistral 7B)
   - Analyzes retrieved information
   - Uses Chain of Thought (CoT) for reasoning

4. **Writer Agent** (GPT-4)
   - Generates clear and concise responses
   - Adapts content for user understanding

5. **Designer Agent** (Stable Diffusion)
   - Creates relevant visualizations
   - Generates explanatory illustrations

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Hack7
```

2. Install Poetry:
```bash
pip install poetry
```

3. Install dependencies:
```bash
poetry install
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```env
OPENAI_API_KEY=your_openai_key
STABILITY_API_KEY=your_stability_key
HUGGINGFACE_API_KEY=your_huggingface_key
ALPHA_VANTAGE_KEY=your_alphavantage_key
EXCHANGERATE_API_KEY=your_exchangerate_key
```

5. Run the application:
```bash
poetry run uvicorn src.infrastructure.api.main:app --reload
```

The API will be available at `http://localhost:8000`

## Testing

1. Run all tests:
```bash
poetry run pytest
```

2. Run specific test categories:
```bash
# Unit tests
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# System tests
poetry run pytest tests/system/

# Debug flow test
poetry run pytest tests/integration/test_agent_flow_debug.py -s
```

## Example Queries

1. **Inflation Impact Query**
```json
POST /query
{
    "query": "como me afecta la inflacion en chile",
    "context": {
        "user_location": "Chile",
        "language": "es"
    }
}
```

2. **Currency Exchange Query**
```json
POST /query
{
    "query": "cual es el precio del dolar hoy",
    "context": {
        "user_location": "Chile",
        "language": "es"
    }
}
```

3. **Investment Advice Query**
```json
POST /query
{
    "query": "como puedo invertir mis ahorros",
    "context": {
        "user_location": "Chile",
        "language": "es",
        "portfolio_value": 5000000
    }
}
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
Hack7/
├── src/
│   ├── application/     # Use cases and application services
│   ├── domain/         # Domain models and business logic
│   ├── infrastructure/ # External services implementation
│   ├── ports/         # Input/Output ports (interfaces)
│   ├── adapters/      # Adapters for external services
│   ├── agents/        # AI agent implementations
│   └── config/        # Configuration files
├── tests/
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── system/       # End-to-end system tests
└── pyproject.toml    # Project dependencies and configuration
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## License

MIT
