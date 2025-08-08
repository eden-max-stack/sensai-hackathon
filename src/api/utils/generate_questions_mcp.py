import asyncio
import json
import math
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from fastapi.params import Body
import openai
import os
from pathlib import Path
from dotenv import load_dotenv

# FastMCP imports
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    label: str
    properties: Dict[str, Any]
    entropy_score: float = 0.0

@dataclass
class KnowledgeGraphRelationship:
    """Represents a relationship in the knowledge graph"""
    source: str
    target: str
    relationship_type: str
    properties: Dict[str, Any]
    entropy_score: float = 0.0

@dataclass
class KnowledgeGraph:
    """Complete knowledge graph structure"""
    nodes: List[KnowledgeGraphNode]
    relationships: List[KnowledgeGraphRelationship]
    metadata: Dict[str, Any]

class OpenAIClient:
    """OpenAI client configured for hackathon API"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'https://agent.dev.hyperverge.org')
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def generate_completion(self, messages: List[Dict[str, str]], model: str = "openai/gpt-4o-mini") -> str:
        """Generate completion using the hackathon OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

class KnowledgeGraphConstructorAgent:
    """Agent responsible for constructing knowledge graphs from learning material"""
    
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client
    
    async def construct_knowledge_graph(self, learning_material: str, bloom_tags: Dict[str, float]) -> KnowledgeGraph:
        """Construct a knowledge graph from learning material and Bloom's taxonomy tags"""
        
        # Create prompt for KG construction
        kg_prompt = f"""
You are a Knowledge Graph Constructor AI. Your task is to analyze learning material and create a structured knowledge graph.

LEARNING MATERIAL:
{learning_material}

BLOOM'S TAXONOMY DISTRIBUTION:
{json.dumps(bloom_tags, indent=2)}

INSTRUCTIONS:
1. Identify key concepts, entities, and their relationships
2. Extract factual statements and conceptual connections
3. Consider the Bloom's taxonomy distribution to understand cognitive complexity
4. Return a structured knowledge graph in JSON format

OUTPUT FORMAT (JSON only):
{{
    "nodes": [
        {{
            "id": "unique_node_id",
            "label": "Node Label",
            "properties": {{
                "type": "concept|fact|process|principle",
                "description": "Brief description",
                "bloom_relevance": ["remembering", "understanding", "applying"],
                "complexity_level": "low|medium|high"
            }}
        }}
    ],
    "relationships": [
        {{
            "source": "node_id_1",
            "target": "node_id_2",
            "relationship_type": "defines|causes|enables|requires|contradicts|supports",
            "properties": {{
                "description": "Relationship description",
                "strength": 0.8,
                "cognitive_load": "low|medium|high"
            }}
        }}
    ],
    "metadata": {{
        "total_nodes": 0,
        "total_relationships": 0,
        "dominant_bloom_level": "understanding",
        "complexity_score": 0.7
    }}
}}

Focus on creating meaningful connections that would help generate educational questions.
"""

        messages = [
            {"role": "system", "content": "You are an expert knowledge graph constructor for educational content. Return only valid JSON."},
            {"role": "user", "content": kg_prompt}
        ]
        
        try:
            response = await self.openai_client.generate_completion(messages)
            kg_data = json.loads(response)
            
            # Convert to KnowledgeGraph object
            nodes = [KnowledgeGraphNode(**node) for node in kg_data.get('nodes', [])]
            relationships = [KnowledgeGraphRelationship(**rel) for rel in kg_data.get('relationships', [])]
            
            return KnowledgeGraph(
                nodes=nodes,
                relationships=relationships,
                metadata=kg_data.get('metadata', {})
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing KG JSON: {e}")
            # Return empty KG on error
            return KnowledgeGraph(nodes=[], relationships=[], metadata={})
        except Exception as e:
            logger.error(f"Error constructing knowledge graph: {e}")
            return KnowledgeGraph(nodes=[], relationships=[], metadata={})

class TokenLevelEntropyGenerator:
    """Agent responsible for generating token-level entropy for KG entities and relationships"""
    
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client
    
    def calculate_token_entropy(self, text: str) -> Dict[str, float]:
        """Calculate entropy for each token in the text"""
        tokens = text.lower().split()
        if not tokens:
            return {}
        
        total = len(tokens)
        counts = Counter(tokens)
        
        entropy_scores = {}
        for token, count in counts.items():
            probability = count / total
            entropy = -math.log2(probability) if probability > 0 else 0
            entropy_scores[token] = entropy
        
        return entropy_scores
    
    def calculate_contextual_entropy(self, kg: KnowledgeGraph, learning_material: str) -> Dict[str, float]:
        """Calculate contextual entropy for KG nodes based on their usage patterns"""
        
        # Extract all text content from nodes and relationships
        all_texts = []
        node_contexts = defaultdict(list)
        
        # Collect node contexts
        for node in kg.nodes:
            node_text = f"{node.label} {node.properties.get('description', '')}"
            all_texts.append(node_text)
            node_contexts[node.id].append(node_text)
        
        # Collect relationship contexts
        for rel in kg.relationships:
            rel_text = f"{rel.relationship_type} {rel.properties.get('description', '')}"
            all_texts.append(rel_text)
            node_contexts[rel.source].append(rel_text)
            node_contexts[rel.target].append(rel_text)
        
        # Add original learning material context
        all_texts.append(learning_material)
        
        # Calculate contextual entropy for each node
        contextual_entropy = {}
        
        for node_id, contexts in node_contexts.items():
            combined_context = " ".join(contexts)
            token_entropy = self.calculate_token_entropy(combined_context)
            
            # Average entropy across all tokens related to this node
            if token_entropy:
                avg_entropy = sum(token_entropy.values()) / len(token_entropy)
                contextual_entropy[node_id] = avg_entropy
            else:
                contextual_entropy[node_id] = 0.0
        
        return contextual_entropy
    
    async def generate_entropy_scores(self, kg: KnowledgeGraph, learning_material: str, bloom_tags: Dict[str, float]) -> KnowledgeGraph:
        """Generate token-level entropy scores for all entities and relationships in the KG"""
        
        # Calculate basic token entropy from learning material
        material_entropy = self.calculate_token_entropy(learning_material)
        
        # Calculate contextual entropy for KG elements
        contextual_entropy = self.calculate_contextual_entropy(kg, learning_material)
        
        # Use AI to enhance entropy calculation with semantic understanding
        entropy_prompt = f"""
You are a Token-Level Entropy Analyzer. Your task is to analyze knowledge graph elements and assign entropy scores based on:

1. Information content (rare vs common concepts)
2. Cognitive complexity 
3. Conceptual abstractness
4. Relationship strength

KNOWLEDGE GRAPH:
Nodes: {len(kg.nodes)}
Relationships: {len(kg.relationships)}

BLOOM'S TAXONOMY DISTRIBUTION:
{json.dumps(bloom_tags, indent=2)}

BASIC ENTROPY SCORES:
{json.dumps(dict(list(material_entropy.items())[:20]), indent=2)}  # First 20 for brevity

For each node ID, provide an enhanced entropy score (0.0 to 10.0) where:
- 0.0-3.0: Low entropy (common, well-defined concepts)
- 3.0-6.0: Medium entropy (moderately complex concepts)  
- 6.0-10.0: High entropy (rare, abstract, or highly complex concepts)

Consider:
- Bloom's taxonomy level relevance
- Conceptual difficulty
- Information rarity
- Cognitive load

OUTPUT FORMAT (JSON only):
{{
    "node_entropy_scores": {{
        "node_id_1": 4.5,
        "node_id_2": 7.2
    }},
    "relationship_entropy_scores": {{
        "rel_index_0": 3.1,
        "rel_index_1": 5.8
    }},
    "global_entropy_metrics": {{
        "avg_node_entropy": 5.2,
        "max_entropy_node": "node_id_2",
        "entropy_distribution": "balanced|skewed_low|skewed_high"
    }}
}}
"""

        messages = [
            {"role": "system", "content": "You are an expert entropy analyzer for educational knowledge graphs. Return only valid JSON."},
            {"role": "user", "content": entropy_prompt}
        ]
        
        try:
            response = await self.openai_client.generate_completion(messages)
            entropy_data = json.loads(response)
            
            # Apply entropy scores to nodes
            node_scores = entropy_data.get('node_entropy_scores', {})
            for node in kg.nodes:
                if node.id in node_scores:
                    node.entropy_score = node_scores[node.id]
                elif node.id in contextual_entropy:
                    node.entropy_score = contextual_entropy[node.id]
                else:
                    node.entropy_score = 1.0  # Default
            
            # Apply entropy scores to relationships
            rel_scores = entropy_data.get('relationship_entropy_scores', {})
            for i, rel in enumerate(kg.relationships):
                rel_key = f"rel_index_{i}"
                if rel_key in rel_scores:
                    rel.entropy_score = rel_scores[rel_key]
                else:
                    rel.entropy_score = 1.0  # Default
            
            # Update metadata with entropy metrics
            kg.metadata.update(entropy_data.get('global_entropy_metrics', {}))
            
            return kg
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing entropy JSON: {e}")
            # Apply basic contextual entropy on error
            for node in kg.nodes:
                node.entropy_score = contextual_entropy.get(node.id, 1.0)
            return kg
        except Exception as e:
            logger.error(f"Error generating entropy scores: {e}")
            return kg

# Initialize FastMCP
mcp = FastMCP("Knowledge Graph Entropy Server")

# Initialize agents
openai_client = OpenAIClient()
kg_agent = KnowledgeGraphConstructorAgent(openai_client)
entropy_agent = TokenLevelEntropyGenerator(openai_client)

@mcp.tool()
async def construct_knowledge_graph(learning_material: str, bloom_tags: dict):
    """
    Construct a knowledge graph from learning material and Bloom's taxonomy tags
    
    Args:
        learning_material: The learning material text to process
        bloom_tags: Bloom's taxonomy distribution scores with keys: remembering, understanding, applying, analyzing, evaluating, creating
    
    Returns:
        JSON string containing the constructed knowledge graph
    """
    try:
        print("building kg graph now....")
        kg = await kg_agent.construct_knowledge_graph(learning_material, bloom_tags)
        print("kg graph built successfully")
        # Convert to serializable format
        result = {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "properties": node.properties,
                    "entropy_score": node.entropy_score
                }
                for node in kg.nodes
            ],
            "relationships": [
                {
                    "source": rel.source,
                    "target": rel.target,
                    "relationship_type": rel.relationship_type,
                    "properties": rel.properties,
                    "entropy_score": rel.entropy_score
                }
                for rel in kg.relationships
            ],
            "metadata": kg.metadata
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error constructing knowledge graph: {str(e)}"

@mcp.tool()
async def generate_entropy_scores(knowledge_graph: dict, learning_material: str, bloom_tags: dict) -> str:
    """
    Generate token-level entropy scores for knowledge graph elements
    
    Args:
        knowledge_graph: Knowledge graph data structure
        learning_material: Original learning material text
        bloom_tags: Bloom's taxonomy distribution scores
    
    Returns:
        JSON string containing the knowledge graph with entropy scores
    """
    try:
        print("Generating entropy scores for knowledge graph...")
        # Reconstruct KG from data
        nodes = [KnowledgeGraphNode(**node) for node in knowledge_graph.get("nodes", [])]
        relationships = [KnowledgeGraphRelationship(**rel) for rel in knowledge_graph.get("relationships", [])]
        kg = KnowledgeGraph(nodes=nodes, relationships=relationships, metadata=knowledge_graph.get("metadata", {}))
        
        # Generate entropy scores
        enhanced_kg = await entropy_agent.generate_entropy_scores(kg, learning_material, bloom_tags)
        print("Entropy scores generated successfully")
        # Convert to serializable format
        result = {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "properties": node.properties,
                    "entropy_score": node.entropy_score
                }
                for node in enhanced_kg.nodes
            ],
            "relationships": [
                {
                    "source": rel.source,
                    "target": rel.target,
                    "relationship_type": rel.relationship_type,
                    "properties": rel.properties,
                    "entropy_score": rel.entropy_score
                }
                for rel in enhanced_kg.relationships
            ],
            "metadata": enhanced_kg.metadata
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error generating entropy scores: {str(e)}"

@mcp.tool()
async def full_pipeline(learning_material: str, bloom_tags: dict) -> str:
    """
    Run complete pipeline: construct KG and generate entropy scores
    
    Args:
        learning_material: The learning material text to process
        bloom_tags: Bloom's taxonomy distribution scores with keys: remembering, understanding, applying, analyzing, evaluating, creating
    
    Returns:
        JSON string containing the complete pipeline results
    """
    try:    
        # Step 1: Construct knowledge graph
        logger.info("Constructing knowledge graph...")
        kg = await kg_agent.construct_knowledge_graph(learning_material, bloom_tags)
        # Step 2: Generate entropy scores
        logger.info("Generating entropy scores...")
        enhanced_kg = await entropy_agent.generate_entropy_scores(kg, learning_material, bloom_tags)

        # Convert to serializable format
        result = {
            "pipeline_status": "completed",
            "processing_steps": [
                "Knowledge Graph Construction",
                "Token-Level Entropy Generation"
            ],
            "knowledge_graph": {
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "properties": node.properties,
                        "entropy_score": node.entropy_score
                    }
                    for node in enhanced_kg.nodes
                ],
                "relationships": [
                    {
                        "source": rel.source,
                        "target": rel.target,
                        "relationship_type": rel.relationship_type,
                        "properties": rel.properties,
                        "entropy_score": rel.entropy_score
                    }
                    for rel in enhanced_kg.relationships
                ],
                "metadata": enhanced_kg.metadata
            },
            "entropy_analysis": {
                "total_nodes": len(enhanced_kg.nodes),
                "total_relationships": len(enhanced_kg.relationships),
                "avg_node_entropy": sum(node.entropy_score for node in enhanced_kg.nodes) / len(enhanced_kg.nodes) if enhanced_kg.nodes else 0,
                "high_entropy_nodes": [
                    {"id": node.id, "label": node.label, "entropy": node.entropy_score}
                    for node in enhanced_kg.nodes if node.entropy_score > 6.0
                ]
            }
        }
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error in full pipeline: {str(e)}"
    

if __name__ == "__main__":
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI()

    # Error handler
    @app.exception_handler(Exception)
    async def universal_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(exc),
                "endpoint": str(request.url)
            }
        )

    # Convert MCP tools to FastAPI endpoints
    @app.post("/construct_knowledge_graph")
    async def construct_kg(learning_material: str, bloom_tags: dict):
        return await mcp.call_tool("construct_knowledge_graph", {
            "learning_material": learning_material,
            "bloom_tags": bloom_tags
        })

    @app.post("/generate_entropy_scores")
    async def generate_entropy(knowledge_graph: dict, learning_material: str, bloom_tags: dict):
        return await mcp.call_tool("generate_entropy_scores", {
            "knowledge_graph": knowledge_graph,
            "learning_material": learning_material,
            "bloom_tags": bloom_tags
        })

    @app.post("/full_pipeline")
    async def full_pipeline(payload: dict = Body(...)):
        print(f"learning material: {payload["learning_material"]}")
        print(f"bloom tags: {payload["bloom_tags"]}")
        return await mcp.call_tool("full_pipeline", {
            "learning_material": payload["learning_material"],
            "bloom_tags": payload["bloom_tags"]
        })

    uvicorn.run(app, host="localhost", port=8000)