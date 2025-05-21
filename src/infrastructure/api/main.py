from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, UTC
from typing import Optional, Dict, Any

from ...domain.models import UserQuery, Response
from ...application.agent_coordinator import AgentCoordinator
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from tests.test_utils import (
    SimpleIntentionAgent, SimpleRetrieverAgent,
    SimpleReasonAgent, SimpleWriterAgent, SimpleDesignerAgent
)

app = FastAPI(
    title="Financial AI Agent API",
    description="API for processing financial queries using multiple AI agents",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    
    @property
    def clean_query(self) -> str:
        """Get the cleaned query text."""
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        return self.query.strip()

class QueryResponse(BaseModel):
    text: str
    visualization_url: Optional[str] = None
    image_url: Optional[str] = None
    created_at: datetime = datetime.now(UTC)

# Initialize test agent instances
intention_agent = SimpleIntentionAgent()
retriever_agent = SimpleRetrieverAgent()
reason_agent = SimpleReasonAgent()
writer_agent = SimpleWriterAgent()
designer_agent = SimpleDesignerAgent()

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a financial query and return the response."""
    try:
        # Initialize context if not provided
        context = request.context or {}
        
        # Add timestamp to context
        context["timestamp"] = datetime.now(UTC).isoformat()
        
        # Create coordinator with agent implementations
        coordinator = AgentCoordinator(
            intention_agent=intention_agent,
            retriever_agent=retriever_agent,
            reason_agent=reason_agent,
            writer_agent=writer_agent,
            designer_agent=designer_agent
        )
        
        # Create query domain object
        query = UserQuery.create(
            query_text=request.clean_query,
            context=context
        )
        
        # Process query through the agent workflow
        response = await coordinator.process_query(query)
        
        # Extract visualization data
        visualization = response.visualization or {}
        
        # Create response
        return QueryResponse(
            text=response.text,
            visualization_url=visualization.get('visualization_url', ''),
            image_url=visualization.get('image_url', ''),
            created_at=datetime.now(UTC)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
