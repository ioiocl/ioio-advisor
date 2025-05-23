from fastapi import FastAPI, Request
from writer_agent import GPT4WriterAgent
import uvicorn

app = FastAPI()
agent = GPT4WriterAgent()

@app.post("/process")
async def process(request: Request):
    data = await request.json()
    return await agent.process(data)

if __name__ == "__main__":
    uvicorn.run("writer_service:app", host="0.0.0.0", port=8000, reload=True)
