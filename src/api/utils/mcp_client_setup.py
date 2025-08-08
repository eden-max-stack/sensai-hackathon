import json
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPKnowledgeGraphClient:
    """Client to communicate with the Knowledge Graph MCP server"""
    
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Start the MCP server
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path],
        )
        
        # Create client session
        self.stdio_client = stdio_client(server_params)
        self.session = await self.stdio_client.__aenter__()
        
        # Initialize the session
        await self.session.initialize()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.stdio_client:
            await self.stdio_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def construct_knowledge_graph(self, learning_material: str, bloom_tags: Dict[str, float]) -> Dict[str, Any]:
        """Call the construct_knowledge_graph tool"""
        try:
            result = await self.session.call_tool(
                "construct_knowledge_graph",
                {
                    "learning_material": learning_material,
                    "bloom_tags": bloom_tags
                }
            )
            return json.loads(result.content[0].text)
        except Exception as e:
            raise Exception(f"Error constructing knowledge graph: {str(e)}")
    
    async def generate_entropy_scores(self, knowledge_graph: Dict[str, Any], learning_material: str, bloom_tags: Dict[str, float]) -> Dict[str, Any]:
        """Call the generate_entropy_scores tool"""
        try:
            result = await self.session.call_tool(
                "generate_entropy_scores",
                {
                    "knowledge_graph": knowledge_graph,
                    "learning_material": learning_material,
                    "bloom_tags": bloom_tags
                }
            )
            return json.loads(result.content[0].text)
        except Exception as e:
            raise Exception(f"Error generating entropy scores: {str(e)}")
    
    async def full_pipeline(self, learning_material: str, bloom_tags: Dict[str, float]) -> Dict[str, Any]:
        """Call the full_pipeline tool (alternative approach)"""
        try:
            result = await self.session.call_tool(
                "full_pipeline",
                {
                    "learning_material": learning_material,
                    "bloom_tags": bloom_tags
                }
            )
            return json.loads(result.content[0].text)
        except Exception as e:
            raise Exception(f"Error in full pipeline: {str(e)}")
