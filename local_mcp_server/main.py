import os
import re
import pyodbc
import time
import json
import base64
import logging
from logging.config import dictConfig
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError
from collections import namedtuple
from typing import Optional, Union, Any, Dict, List
from decimal import Decimal
from datetime import datetime, date

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda

# --- Logging & App Setup ---
dictConfig({
    "version": 1, "disable_existing_loggers": False,
    "formatters": {"default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
    "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "default", "level": "INFO"}},
    "loggers": {"": {"handlers": ["default"], "level": "INFO", "propagate": False}},
})
logger = logging.getLogger("local_mcp_server")
load_dotenv()
app = FastAPI(title="Local AdventureWorks MCP Server")
security = HTTPBearer()
EXPECTED_TOKEN = os.getenv("MCP_SERVER_AUTH_TOKEN")


# --- Caching for Table Names ---
TableCache = namedtuple("TableCache", ["tables_str", "timestamp"])
table_cache = TableCache(None, 0)
TABLE_CACHE_TTL_SECONDS = 3600

# --- Dynamic JSON Encoder ---
def make_serializable(obj: Any) -> Any:
    """
    Recursively walks a data structure and converts non-serializable
    types to JSON-friendly formats before serialization.
    """
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    return obj

# --- Database & Helper Functions ---
def get_db_connection():
    conn_str = f"DRIVER={os.getenv('DB_DRIVER')};SERVER={os.getenv('DB_SERVER')};DATABASE={os.getenv('DB_NAME')};UID={os.getenv('DB_USER')};PWD={os.getenv('DB_PASSWORD')};"
    return pyodbc.connect(conn_str)

def get_all_table_names() -> str:
    global table_cache
    if table_cache.tables_str and (time.time() - table_cache.timestamp) < TABLE_CACHE_TTL_SECONDS: return table_cache.tables_str
    logger.info("Fetching fresh table list from database.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA');"
        cursor.execute(sql)
        tables = [f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}" for row in cursor.fetchall()]
        cursor.close(); conn.close()
        tables_str = ", ".join(tables)
        table_cache = TableCache(tables_str, time.time())
        return tables_str
    except Exception as e:
        logger.error(f"Failed to fetch table names: {e}"); return ""

def get_focused_schema(table_names_str: str) -> str:
    table_names = [t.strip() for t in table_names_str.split(',') if t.strip()]
    if not table_names: return "No tables selected."
    conn = get_db_connection()
    cursor = conn.cursor()
    schema_parts = []
    for i, full_table_name in enumerate(table_names, 1):
        try: table_schema, table_name = full_table_name.split('.')
        except ValueError: continue
        column_info = [f"{row.COLUMN_NAME} ({row.DATA_TYPE})" for row in cursor.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{table_schema}' AND TABLE_NAME = '{table_name}' ORDER BY ORDINAL_POSITION;").fetchall()]
        schema_parts.append(f"{i}. {full_table_name} ({', '.join(column_info)})")
    cursor.close(); conn.close()
    return "Tables:\n" + "\n".join(schema_parts)

def sanitize_sql(sql_query: str):
    if not sql_query or not re.match(r"^\s*SELECT", sql_query, re.IGNORECASE): raise ValueError("Query must be a SELECT statement.")
    blocked_keywords = ['DROP', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'EXEC']
    if any(keyword in sql_query.upper() for keyword in blocked_keywords): raise ValueError("Query contains a blocked keyword.")
    return sql_query

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN: raise HTTPException(status_code=403, detail="Invalid token")
    return credentials

# --- Application Startup Event ---
@app.on_event("startup")
def startup_event():
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0)
    sql_llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    table_selection_template = "You are a database expert. Based on the user's question, which of the following tables are most relevant? Respond with a comma-separated list of table names ONLY.\n\nUser Question: \"{question}\"\n\nAvailable Tables: {tables}"
    table_selection_chain = (ChatPromptTemplate.from_template(table_selection_template)| llm | StrOutputParser())
    sql_generation_template = """
    You are an expert SQL Server AI assistant. Your primary task is to write a single, valid SQL query to answer the user's question, based on the provided schema.

    **CRITICAL INSTRUCTIONS:**
    1.  Your response MUST contain only the raw SQL query, and nothing else.
    2.  You MUST select not only the columns that directly answer the question, but also any other relevant columns that provide context (such as names, descriptions, or prices).
    3.  For example, if asked for the "most expensive product ID", you must also select the product's name and its price to justify the answer.

    Schema:
    {schema}

    User Question: "{question}"
    """
    sql_generation_chain = (ChatPromptTemplate.from_template(sql_generation_template)| sql_llm| StrOutputParser())
    app.state.sql_chain = (RunnablePassthrough.assign(tables=RunnableLambda(lambda x: get_all_table_names()))| RunnablePassthrough.assign(relevant_tables=table_selection_chain)| RunnablePassthrough.assign(schema=lambda x: get_focused_schema(x["relevant_tables"]))| sql_generation_chain)
    logger.info("Text-to-SQL LCEL chain compiled and ready.")


# --- JSON-RPC Model ---
class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"; method: str; params: Optional[Union[Dict[str, Any], List[Any]]] = None; id: Union[str, int]

# --- Final MCP Endpoint ---
@app.post("/mcp", dependencies=[Depends(verify_token)])
async def handle_mcp_request(request: Request):
    body = await request.body()
    try:
        data = json.loads(body) if body else {}
        req = JsonRpcRequest.model_validate(data)
        method, request_id, params = req.method, req.id, req.params if isinstance(req.params, dict) else {}
    except (json.JSONDecodeError, ValidationError):
        method, request_id, params = "tools/list", "discovery", {}
    
    if method == "initialize":
        client_version = params.get("protocolVersion", "2025-06-18")
        initialize_result = {"protocolVersion": client_version, "serverInfo": {"name": "AdventureWorks Server", "version": "1.0.0"}, "capabilities": {}}
        return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "result": initialize_result})
        
    elif method in ["mcp/getTools", "tools/list"]:
        tools_list = [{"name": "text_to_sql_adventureworks", "description": "Executes a natural language query against the AdventureWorks database.", "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}]
        discovery_result = {"tools": tools_list, "cursor": None}
        return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "result": discovery_result})

    elif method == "tools/call":
        logger.info(f"Handling 'tools/call' execution request (ID: {request_id}).")
        
        user_prompt = params.get("arguments", {}).get("prompt")
        if not user_prompt:
            error_data = {"code": -32602, "message": "Invalid params: Could not find 'prompt'"}
            return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "error": error_data}, status_code=400)
            
        try:
            sql_chain = request.app.state.sql_chain
            llm_response_text = await sql_chain.ainvoke({"question": user_prompt})
            
            sql_match = re.search(r"```(sql)?\s*(.*?)\s*```", llm_response_text, re.DOTALL | re.IGNORECASE)
            sql_query = sql_match.group(2).strip() if sql_match else llm_response_text.strip()
            logger.info(f"Extracted SQL for execution: {sql_query}")

            safe_sql = sanitize_sql(sql_query)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(safe_sql)
            columns = [c[0] for c in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close(); conn.close()
            
            # 1. Making the results serializable to ensure all data types are JSON-friendly.
            serializable_results = make_serializable(results)
            
            # 2. Convert the results into a single formatted text block.
            text_output = json.dumps(serializable_results, indent=2)

            # 3. Create a standard MCP TextContent part
            content_parts = [
                {
                    "type": "text",
                    "text": text_output
                }
            ]
            
            # 4. Wrap this content in the final result object.
            result_object = {"content": content_parts}
            
            response_payload = {"jsonrpc": "2.0", "id": request_id, "result": result_object}
            return JSONResponse(content=response_payload)

        except Exception as e:
            logger.error(f"Text-to-SQL chain failed: {e}", exc_info=True)
            error_data = {"code": -32000, "message": f"Server error: {e}"}
            return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "error": error_data}, status_code=500)
    
    else:
        logger.warning(f"Received request for unknown method: {method}")
        error_data = {"code": -32601, "message": "Method not found"}
        return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "error": error_data}, status_code=404)