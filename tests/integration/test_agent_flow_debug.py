import pytest
from datetime import datetime, UTC
from src.domain.models import UserQuery, Response
from src.application.agent_coordinator import AgentCoordinator
from tests.test_utils import (
    SimpleIntentionAgent, SimpleRetrieverAgent, SimpleReasonAgent,
    SimpleWriterAgent, SimpleDesignerAgent
)

@pytest.fixture
def agents():
    """Create instances of all agents."""
    return {
        "intention": SimpleIntentionAgent(),
        "retriever": SimpleRetrieverAgent(),
        "reason": SimpleReasonAgent(),
        "writer": SimpleWriterAgent(),
        "designer": SimpleDesignerAgent()
    }

@pytest.fixture
def coordinator(agents):
    """Create AgentCoordinator with test agents."""
    return AgentCoordinator(
        intention_agent=agents["intention"],
        retriever_agent=agents["retriever"],
        reason_agent=agents["reason"],
        writer_agent=agents["writer"],
        designer_agent=agents["designer"]
    )

@pytest.mark.asyncio
async def test_show_agent_flow(coordinator):
    """Test showing the complete flow through all agents with console output."""
    
    print("\n=== Starting Agent Flow Test ===\n")
    
    # Test data
    query = UserQuery.create(
        "como me afecta la inflacion en chile",
        context={
            "user_location": "Chile",
            "timestamp": datetime.now(UTC).isoformat(),
            "language": "es"
        }
    )
    
    print(f"Input Query: {query.query_text}")
    print(f"Context: {query.context}\n")
    
    # Execute the flow
    response = await coordinator.process_query(query)
    
    print("\n=== Flow Complete ===")
    print("\nFinal Response:")
    print(f"Text: {response.text}")
    print(f"Visualization URL: {response.visualization_url}")
    print(f"Created At: {response.created_at}")
    
    # Basic assertions to ensure response is valid
    assert isinstance(response, Response)
    assert len(response.text) > 0
    assert response.created_at is not None
