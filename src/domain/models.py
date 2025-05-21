from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime, UTC

@dataclass
class UserQuery:
    """Represents a user's financial information query."""
    query_text: str
    timestamp: datetime
    context: Dict[str, Any]
    
    @classmethod
    def create(cls, query_text: str, context: Optional[Dict[str, Any]] = None) -> 'UserQuery':
        return cls(
            query_text=query_text,
            timestamp=datetime.now(UTC),
            context=context or {}
        )

@dataclass
class QueryIntent:
    """Represents the detected intention from a user query."""
    main_topic: str
    subtopics: list[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class FinancialContext:
    """Represents retrieved financial context and data."""
    sources: list[str]
    data_points: Dict[str, Any]
    timestamp: datetime
    relevance_score: float

@dataclass
class Analysis:
    """Represents the reasoned analysis of financial information."""
    key_points: list[str]
    implications: Dict[str, str]
    confidence_level: float
    reasoning_chain: list[str]

from pydantic import BaseModel

class Response(BaseModel):
    """Represents the final response to be delivered to the user."""
    text: str
    visualization: Dict[str, Any]
    query_id: str
    created_at: datetime

    @classmethod
    def create(cls, content: str, visualization_data: Optional[Dict[str, Any]] = None, query_id: Optional[str] = None) -> 'Response':
        """Create a new response with the given content and optional visualization data."""
        vis_data = visualization_data or {}
        return cls(
            text=content,
            visualization=vis_data,
            query_id=query_id or str(hash(content)),
            created_at=datetime.now(UTC)
        )
