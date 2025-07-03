import os
import json
import base64
import time
import logging
from logging.config import dictConfig
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from azure.storage.queue.aio import QueueClient
from azure.identity.aio import DefaultAzureCredential

from graph import create_agent_graph

# --- Logging & App Setup ---
load_dotenv()
LOGGING_CONFIG = {
    "version": 1, "disable_existing_loggers": False,
    "formatters": {"default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
    "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "default", "level": "INFO"}},
    "loggers": {"": {"handlers": ["default"], "level": "INFO", "propagate": False}},
}
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("orchestrator")
app = FastAPI(title="LangGraph MCP Orchestrator")
app.mount("/static", StaticFiles(directory="web"), name="static")

# --- Agent & HITL Configuration ---
SYSTEM_PROMPT = """
You are a specialized AI assistant for processing business and cloud infrastructure queries.
Your primary directives are:
1.  Strictly Adhere to Provided Data: You must ONLY use the information returned from the tools to formulate your answer.
2.  Do Not Speculate: If the tool output does not contain the answer, you must state that the information is unavailable.
3.  Be Concise and Factual: Present data and answers clearly and directly. Do not offer opinions, predictions, or advice.
4.  Acknowledge Errors: If a tool returns an error, report the error clearly to the user.
"""
QUEUE_NAME = "review-queue"
STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up. Compiling agent graph...")
    app.state.agent = await create_agent_graph()
    logger.info("Application startup complete.")

# --- Middleware & Routes ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request received: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Request completed: {request.method} {request.url.path} - Status {response.status_code} in {process_time:.2f}ms")
    return response

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("web/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/prompt")
async def handle_prompt(request: Request):
    data = await request.json()
    user_prompt = data.get('prompt')
    if not user_prompt: return JSONResponse(status_code=400, content={"error": "Prompt not provided."})

    try:
        agent = request.app.state.agent
        inputs = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]}
        
        logger.info(f"Invoking agent for prompt: '{user_prompt[:50]}...'")
        final_state = await agent.ainvoke(inputs)
        final_answer = final_state["messages"][-1].content
        logger.info(f"Agent generated final answer. For local testing, answer is: {final_answer}")
        

        # For local testing, bypass HITL and return the answer directly.
        if STORAGE_ACCOUNT_URL:
            logger.info("Production environment detected. Sending to HITL review queue.")
            try:
                credential = DefaultAzureCredential()
                async with QueueClient(account_url=STORAGE_ACCOUNT_URL, queue_name=QUEUE_NAME, credential=credential) as queue_client:
                    message_content = {"original_prompt": user_prompt, "proposed_answer": final_answer, "status": "pending_review"}
                    message_b64 = base64.b64encode(json.dumps(message_content).encode('utf-8')).decode('utf-8')
                    await queue_client.send_message(message_b64)
                return JSONResponse(content={"message": "Your request has been received and sent for human review."})
            except Exception as e:
                logger.error(f"Failed to send message to review queue: {e}")
                raise HTTPException(status_code=500, detail="Could not process the request for review.")
        else:
            logger.warning("Bypassing HITL queue for local development. Returning answer directly.")
            return JSONResponse(content={"answer": final_answer})
    except Exception as e:
        logger.error(f"An error occurred during agent invocation: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred while processing your request."}
        )