#!/bin/bash
# Usage: ./test_intention_agent.sh <SERVICE_URL>
# Example: ./test_intention_agent.sh http://localhost:8000

SERVICE_URL=${1:-http://localhost:8000}

# Sample JSON payload for a financial query in Spanish
cat <<EOF > payload.json
{
  "query": "¿Cuál es la tendencia del dólar en Argentina?"
}
EOF

echo "Testing Intention Agent at $SERVICE_URL/process ..."
curl -X POST "$SERVICE_URL/process" \
  -H "Content-Type: application/json" \
  -d @payload.json

echo -e "\n\nTest complete."
