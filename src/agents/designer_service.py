from fastapi import FastAPI, Request
from designer_agent import StableDiffusionDesignerAgent
import uvicorn

app = FastAPI()
agent = StableDiffusionDesignerAgent()

@app.post("/process")
async def process(request: Request):
    data = await request.json()
    return await agent.generate_visualization(data, data.get("text", ""))

if __name__ == "__main__":
    uvicorn.run("designer_service:app", host="0.0.0.0", port=8000, reload=True)
