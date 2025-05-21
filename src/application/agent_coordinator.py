from typing import Dict, Any, Optional, Annotated
from pydantic import BaseModel
from langgraph.graph import Graph, StateGraph
from typing import Optional
from datetime import datetime, UTC
from ..domain.models import UserQuery, Response
from ..ports.agent_port import (
    IntentionAgent,
    RetrieverAgent,
    ReasonAgent,
    WriterAgent,
    DesignerAgent
)

# Define state schema
class State(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    intent: Optional[Dict[str, Any]] = None
    context_data: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    visualization: Optional[Dict[str, Any]] = None

class AgentCoordinator:
    """Coordinates the flow of information between different AI agents using LangGraph."""
    
    def __init__(
        self,
        intention_agent: IntentionAgent,
        retriever_agent: RetrieverAgent,
        reason_agent: ReasonAgent,
        writer_agent: WriterAgent,
        designer_agent: DesignerAgent
    ):
        self.intention_agent = intention_agent
        self.retriever_agent = retriever_agent
        self.reason_agent = reason_agent
        self.writer_agent = writer_agent
        self.designer_agent = designer_agent
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph workflow for processing queries."""
        
        # Define the graph
        workflow = StateGraph(state_schema=Dict[str, Any])
        
        # Add nodes with their actions
        workflow.add_node("process_intent", self._process_intent)
        workflow.add_node("retrieve_info", self._retrieve_info)
        workflow.add_node("analyze_info", self._analyze_info)
        workflow.add_node("compose_response", self._compose_response)
        workflow.add_node("generate_visual", self._generate_visual)
        workflow.add_node("finalize", self._finalize)

        # Define the flow
        workflow.add_edge("process_intent", "retrieve_info")
        workflow.add_edge("retrieve_info", "analyze_info")
        workflow.add_edge("analyze_info", "compose_response")
        workflow.add_edge("compose_response", "generate_visual")
        workflow.add_edge("generate_visual", "finalize")

        # Set entry point
        workflow.set_entry_point("process_intent")

        return workflow.compile()

    async def process_query(self, query: UserQuery) -> Response:
        """Process a user query through the agent workflow."""
        
        # Initialize or get context
        context = query.context or {}
        
        # Generate unique query_id based on query text and timestamp
        timestamp = datetime.now(UTC).isoformat()
        query_id = str(hash(f"{query.query_text}_{timestamp}_{id(query)}"))
        
        # Update context with essential information
        context.update({
            "query_id": query_id,
            "timestamp": timestamp,
            "original_query": query.query_text
        })
        
        # Create initial state with all required fields
        initial_state = {
            "query": query.query_text,
            "context": context,
            "intent": None,
            "analysis": None,
            "response": None,
            "visualization": None
        }

        print(f"[AgentCoordinator] Initial state: {initial_state}")
        try:
            # Process through agent workflow
            final_state = await self.graph.ainvoke(initial_state)
            print(f"[AgentCoordinator] Final state: {final_state}")
            
            # Extract visualization data
            vis_data = final_state.get("visualization", {})
            print(f"[AgentCoordinator] Visualization data: {vis_data}")
            
            # Create response
            response = Response.create(
                content=final_state["response"],
                visualization_data=vis_data,
                query_id=final_state["context"]["query_id"]
            )
            print(f"[AgentCoordinator] Created response: {response}")
            
            return response
            
        except Exception as e:
            print(f"[AgentCoordinator] Error in process_query: {str(e)}")
            print(f"[AgentCoordinator] Error details: {type(e).__name__}")
            import traceback
            print(f"[AgentCoordinator] Traceback: {traceback.format_exc()}")
            raise

    async def _process_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process user intent using the intention agent."""
        result = await self.intention_agent.process({
            "query": state["query"],
            "context": state["context"]
        })
        state["intent"] = result
        return state

    async def _retrieve_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant information using the retriever agent."""
        result = await self.retriever_agent.process({
            "query": state["query"],
            "context": state["context"],
            "intent": state["intent"]
        })
        # Update context with retrieved information
        if "context" not in state:
            state["context"] = {}
        state["context"].update(result)
        return state

    async def _analyze_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze information using the reason agent."""
        print(f"State in _analyze_info: {state}")
        result = await self.reason_agent.process({
            "query": state["query"],
            "context": state["context"],
            "intent": state["intent"]
        })
        state["analysis"] = result
        return state

    async def _compose_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compose response using the writer agent."""
        result = await self.writer_agent.process({
            "query": state["query"],
            "context": state["context"],
            "analysis": state["analysis"]
        })
        state["response"] = result["response"]
        return state

    async def _generate_visual(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization using the designer agent."""
        result = await self.designer_agent.process({
            "context": state["context"],
            "response": state["response"]
        })
        if isinstance(result, dict) and "visualization" in result:
            state["visualization"] = result["visualization"]
        else:
            state["visualization"] = result
        return state

    async def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the state and prepare the response."""
        return state
