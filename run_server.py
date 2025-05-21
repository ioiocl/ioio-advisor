from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, UTC
from typing import Optional, Dict, Any
from run_agents import IntentionAgent, RetrieverAgent, ReasonAgent, WriterAgent, DesignerAgent

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
agents = {
    "intention": IntentionAgent(),
    "retriever": RetrieverAgent(),
    "reason": ReasonAgent(),
    "writer": WriterAgent(),
    "designer": DesignerAgent()
}

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    text: str
    visualization_url: Optional[str] = None
    created_at: datetime = datetime.now(UTC)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Process query through agent pipeline
        input_data = {"query": request.query, "context": request.context or {}}
        
        # Step 1: Intention Agent
        intention_output = await agents["intention"].process(input_data)
        
        # Step 2: Retriever Agent
        retriever_output = await agents["retriever"].process(input_data)
        
        # Step 3: Reason Agent
        reason_input = {**input_data, **intention_output, **retriever_output}
        reason_output = await agents["reason"].process(reason_input)
        
        # Step 4: Writer Agent
        writer_input = {**input_data, **reason_output}
        writer_output = await agents["writer"].process(writer_input)
        
        # Step 5: Designer Agent
        designer_input = {**input_data, **writer_output}
        designer_output = await agents["designer"].process(designer_input)
        
        return QueryResponse(
            text=writer_output["response"],
            visualization_url=designer_output.get("visualization_url"),
            created_at=datetime.now(UTC)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now(UTC)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
