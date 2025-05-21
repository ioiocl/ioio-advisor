from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class AgentPort(ABC):
    """Base interface for all AI agents in the system."""
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results."""
        pass

class IntentionAgent(AgentPort):
    """Interface for the intention detection agent."""
    
    @abstractmethod
    async def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect user intention from the query."""
        pass

class RetrieverAgent(AgentPort):
    """Interface for the information retrieval agent."""
    
    @abstractmethod
    async def retrieve_information(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant information based on query and context."""
        pass

class ReasonAgent(AgentPort):
    """Interface for the reasoning agent."""
    
    @abstractmethod
    async def analyze(self, query: str, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze information and generate reasoning."""
        pass

class WriterAgent(AgentPort):
    """Interface for the response writing agent."""
    
    @abstractmethod
    async def compose_response(self, analysis: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """Compose a clear and concise response."""
        pass

class DesignerAgent(AgentPort):
    """Interface for the visualization generation agent."""
    
    @abstractmethod
    async def generate_visualization(self, context: Dict[str, Any], text: str) -> bytes:
        """Generate a visual representation of the information."""
        pass
