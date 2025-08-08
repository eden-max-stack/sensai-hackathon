import json
import aiohttp
from typing import Dict, Any

class MCPKnowledgeGraphClient:
    """HTTP client to communicate with the Knowledge Graph MCP server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, payload: Dict[str, Any]):
        """Enhanced request handler with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        # print(f"payload sent: {payload}")
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_detail}")
                return await response.json()
        except Exception as e:
            raise Exception(f"Request failed to {url}: {str(e)}")
    
    async def construct_knowledge_graph(self, learning_material: str, bloom_tags: Dict[str, float]) -> Dict[str, Any]:
        """Call with automatic error handling"""
        try:
            return await self._make_request(
                "construct_knowledge_graph",
                {
                    "learning_material": learning_material,
                    "bloom_tags": bloom_tags
                }
            )
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "bloom_taxonomy": bloom_tags
            }
    
    async def generate_entropy_scores(self, knowledge_graph: Dict[str, Any], learning_material: str, bloom_tags: Dict[str, float]) -> Dict[str, Any]:
        """Call the generate_entropy_scores endpoint"""
        return await self._make_request(
            "generate_entropy_scores",
            {
                "knowledge_graph": knowledge_graph,
                "learning_material": learning_material,
                "bloom_tags": bloom_tags
            }
        )
    
    async def full_pipeline(self, learning_material: str, bloom_tags: Dict[str, float]):
        """Call the full_pipeline endpoint"""
        return await self._make_request(
            "full_pipeline",
            {
                "learning_material": learning_material,
                "bloom_tags": bloom_tags
            }
        )