import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.models import OrchestrationResponse, QueryRequest, QueryResponse
from graph.graph_builder import GraphBuilder
from graph.orchestrator_graph_builder import OrchestratorGraphBuilder
from llm.llm import OpenAILLM

load_dotenv()

DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="LLM Orchestration API", debug=DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    llm = OpenAILLM().get_llm()
    response = await asyncio.to_thread(GraphBuilder(llm).run, req.question)

    return QueryResponse(
        answer=response["result"],
        route=response["route"],
        route_reason=response["route_reason"],
    )


@app.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    response = await asyncio.to_thread(OrchestratorGraphBuilder().run, req.question)
    workers_used = [r["worker"] for r in response["worker_results"]]

    return OrchestrationResponse(
        answer=response["answer"],
        workers_used=workers_used,
        plan=response["plan"],
        worker_results=response["worker_results"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )
