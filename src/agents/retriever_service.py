from fastapi import FastAPI, Request
from retriever_agent import InstructorXLRetrieverAgent
import uvicorn

app = FastAPI()
agent = InstructorXLRetrieverAgent()

@app.post("/process")
async def process(request: Request):
    data = await request.json()
    return await agent.process(data)

if __name__ == "__main__":
    uvicorn.run("retriever_service:app", host="0.0.0.0", port=8000, reload=True)
