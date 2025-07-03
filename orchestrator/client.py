import os
import json
import httpx

class MultiMCPClient:
    """
    A pure session manager for MCP servers. It registers servers and makes
    authenticated requests, but does not contain routing or synthesis logic.
    """
    def __init__(self):
        self.servers = []
        self._http_client = httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._http_client.aclose()

    def load_remote_servers(self, config_path: str = ".vscode/mcp.json"):
        """Loads server configurations from the JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                loaded = config.get("servers", [])
                self.servers.extend(loaded)
                print(f"Loaded {len(loaded)} remote servers from {config_path}")
        except FileNotFoundError:
            print(f"Info: Configuration file not found at {config_path}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {config_path}")

    def add_server_from_env(self):
        """Adds local server configuration from environment variables."""
        name = os.getenv("LOCAL_MCP_NAME")
        if name:
            server_config = {
                "name": name,
                "description": os.getenv("LOCAL_MCP_DESCRIPTION"),
                "url": os.getenv("LOCAL_MCP_URL"),
                "auth_token": os.getenv("LOCAL_MCP_AUTH_TOKEN")
            }
            self.servers.append(server_config)
            print(f"Loaded local server '{name}' from environment variables.")

    def get_server_by_name(self, name: str):
        """Finds a registered server by its name."""
        return next((s for s in self.servers if s['name'] == name), None)

    async def call_server(self, server_name: str, prompt: str) -> dict:
        """Makes a secure, authenticated call to a single, named MCP server."""
        server = self.get_server_by_name(server_name)
        if not server:
            return {"server": server_name, "status": "error", "data": "Server not registered."}
        
        try:
            headers = {
                'Authorization': f'Bearer {server["auth_token"]}',
                'Content-Type': 'application/json'
            }
            response = await self._http_client.post(server['url'], headers=headers, json={"prompt": prompt}, timeout=30.0)
            response.raise_for_status()
            return {"server": server['name'], "status": "success", "data": response.json()}
        except httpx.RequestError as e:
            return {"server": server['name'], "status": "error", "data": f"Request failed: {e.__class__.__name__}"}
        except httpx.HTTPStatusError as e:
            return {"server": server['name'], "status": "error", "data": f"HTTP Error: {e.response.status_code} - {e.response.text}"}