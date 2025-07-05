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

from mcp.server.fastmcp import FastMCP
from mcp.error import JSONRPCError
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
EXPECTED_TOKEN = os.getenv("MCP_SERVER_AUTH_TOKEN")

app = FastAPI(title="Local AdventureWorks MCP Server")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN: raise HTTPException(status_code=403, detail="Invalid token")
    return credentials

# ── Create the MCP server ────────────────────────────────────
mcp = FastMCP(
    title="AdventureWorks Server",
    version="1.0.0",
    auth_provider=None  
)


# --- Caching for Table Names ---
TableCache = namedtuple("TableCache", ["tables_str", "timestamp"])
table_cache = TableCache(None, 0)
TABLE_CACHE_TTL_SECONDS = 3600

# --- LangChain Components ---
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0)
sql_llm = ChatOpenAI(model="gpt-4", temperature=0.0)

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



# ---1. Formal MCP Resource Definition ---
@mcp.resource(
    name="db:schema/adventureworks",
    description="Provides the table and column schema for the AdventureWorksLT2019 database.",
    input_schema={
        "type":"object",
        "properties": { 
            "table_names_str":{
                "type":"string", 
                "description": "Comma-separated list of fully-qualified tables"
            } 
        },
        "required":["table_names_str"]       
    }
)
def get_database_schema_resource(table_names_str: Optional[str] = None) -> str:
    """
    Exposes the database schema as a formal MCP resource.
    Can be called with a specific list of tables or will fetch all.
    """
    logger.info("MCP resource 'db:schema/adventureworks' requested.")
    if table_names_str:
        return get_focused_schema(table_names_str)
    else:
        # For a general request, we might return a summary or the full schema
        all_tables = get_all_table_names()
        return get_focused_schema(all_tables)

# --- 2. Formal MCP Prompt Template Definition ---
@mcp.prompt(
    name="generate_sql_from_schema",
    description="Prompt to generate an SQL query given a question and schema.",
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The natural language question to answer"},
            "schema": {"type": "string", "description": "Database schema to use for context."}
        },
        "required": ["question", "schema"]
    }
)
def get_sql_generation_prompt(question: str, schema: str) -> List[Dict[str, Any]]:
    """
    Returns a structured prompt conversation for the LLM.
    """
    logger.info("MCP prompt 'generate_sql_from_schema' requested.")
    return [
        {
            "role": "system",
            "content": """
            You are an expert SQL Server AI assistant. Your primary task is to write a single, valid SQL query to answer the user's question, based on the provided schema.

            **CRITICAL INSTRUCTIONS:**
            1. Your response MUST contain only the raw SQL query, and nothing else.
            2. You MUST select not only the columns that directly answer the question, but also any other relevant columns that provide context (such as names, descriptions, or prices).
            """
        },
        {
            "role": "user",
            "content": f"Schema:\n{schema}\n\nUser Question: \"{question}\""
        }
    ]

# --- REVISED: The `tools/call` handler now uses these primitives ---
@mcp.tool(
    name="text_to_sql_adventureworks",
    description="Executes a natural language query against the AdventureWorks database.",
    input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}
)
async def text_to_sql_tool(prompt: str) -> List[Dict[str, Any]]:
    """
    This is the action the agent can invoke. In a fully compliant model,
    this tool would internally use the 'resource' and 'prompt' primitives.
    """
    logger.info(f"Executing tool 'text_to_sql_adventureworks' for prompt: '{prompt[:50]}...'")
    try:
        # For demonstration, we use the direct chain here. A more decoupled tool
        # would first make a call to read the 'db:schema/adventureworks_all' resource.
        schema = get_focused_schema(get_all_table_names())
        
        # Then it would use the 'generate_sql_from_schema_and_question' prompt
        sql_generation_prompt_template = ChatPromptTemplate.from_messages(get_sql_generation_prompt(prompt, schema))
        sql_generation_chain = sql_generation_prompt_template | sql_llm | StrOutputParser()

        llm_response_text = await sql_generation_chain.ainvoke({})
        
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
        
        # The tool directly returns the final, serializable result.
        # The FastMCP library handles wrapping this in the correct JSON-RPC response.
        return make_serializable(results)
    except ValueError as ve:
        raise JSONRPCError(code=400, message=str(ve))
    except Exception as e:
        logger.exception(e)
        raise JSONRPCError(code=500, message="Internal server error")
    
app.include_router(mcp.router, dependencies=[Depends(verify_token)])
