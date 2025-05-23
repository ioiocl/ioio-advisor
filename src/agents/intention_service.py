from fastapi import FastAPI, Request
from intention_agent import Phi3IntentionAgent
import uvicorn

app = FastAPI()
agent = Phi3IntentionAgent()

@app.post("/process")
async def process(request: Request):
    data = await request.json()
    return await agent.process(data)

if __name__ == "__main__":
    uvicorn.run("intention_service:app", host="0.0.0.0", port=8000, reload=True)
