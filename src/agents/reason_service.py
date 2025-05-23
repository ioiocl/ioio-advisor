from fastapi import FastAPI, Request
from reason_agent import Mistral7BReasonAgent
import uvicorn

app = FastAPI()
agent = Mistral7BReasonAgent()

@app.post("/process")
async def process(request: Request):
    data = await request.json()
    return await agent.process(data)

if __name__ == "__main__":
    uvicorn.run("reason_service:app", host="0.0.0.0", port=8000, reload=True)
