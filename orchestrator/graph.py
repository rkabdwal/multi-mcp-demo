import os
import json
import logging
from typing import List, Annotated
from typing_extensions import TypedDict
from functools import partial

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- Get a logger instance ---
logger = logging.getLogger(__name__)

# --- 1. Define the Agent's State ---
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]

# --- 2. Initialize Clients and Tools ---

async def load_tools_from_mcp():
    """
    Uses langchain-mcp-adapters to connect to all servers and load their tools.
    """
    connections = {}
    # Load stdio servers from the JSON config
    try:
        with open(".vscode/mcp.json") as f:
            connections.update(json.load(f)["servers"])
    except FileNotFoundError:
        print("INFO: .vscode/mcp.json not found. Skipping remote stdio servers.")

    # Load local HTTP server from environment variables
    local_name = os.getenv("LOCAL_MCP_NAME")
    if local_name:
        connections[local_name] = {
            "transport": "streamable_http",
            "url": os.getenv("LOCAL_MCP_URL"),
            "description": os.getenv("LOCAL_MCP_DESCRIPTION"),
            "headers": {"Authorization": f"Bearer {os.getenv('LOCAL_MCP_AUTH_TOKEN')}"}
        }
    
    client = MultiServerMCPClient(connections)
    # Returns a list of LangChain-compatible tools
    tools = await client.get_tools()
    print(f"Successfully loaded {len(tools)} tools from MCP servers.")
    return tools

# --- 3. Define the Graph's Nodes ---

async def agent_node(state: AgentState, *, model_with_tools):
    """
    The central "brain" of the agent. The pre-configured model_with_tools is passed in.
    """
    
     # Log the last message in the state to see what the ToolNode returned.
    if state["messages"] and len(state["messages"]) > 1:
        logger.info(f"--- AGENT NODE DEBUG --- Received Tool Output: {state['messages'][-1]}")

    
    result = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [result]}

def should_continue(state: AgentState) -> str:
    """
    This is a conditional edge. It decides where to go next.
    If the last message was a tool call, we go to the "tools" node.
    Otherwise, we are finished.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- 4. Assemble the Graph ---

async def create_agent_graph():
    """
    Factory function to create the compiled LangGraph agent.
    """
    # Create the LLM client
    llm = ChatOpenAI(model="gpt-4-turbo")
    
    # Asynchronously load all tools from the MCP servers
    tools = await load_tools_from_mcp()
    
    # Bind the available tools to the LLM.
    model_with_tools = llm.bind_tools(tools)
    
    tool_node = ToolNode(tools) # A pre-built node that executes tool calls

    #async node
    agent_node_with_tools = partial(agent_node, model_with_tools=model_with_tools)

    # Define the graph structure
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    # The `agent_node` is partially initialized with the model that already knows about the tools.
    workflow.add_node("agent", agent_node_with_tools)
    workflow.add_node("tools", tool_node)

    # Define the edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent") # After using a tool, go back to the agent to see what's next

    # Compile the graph into a runnable agent
    agent = workflow.compile()
    print("Agent graph compiled successfully.")
    return agent