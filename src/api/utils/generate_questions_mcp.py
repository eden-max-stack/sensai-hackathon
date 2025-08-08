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
from dotenv import load_dotenv

# from ..models import (
#     TaskType,
#     TaskInputType,
#     TaskAIResponseType,
#     QuestionType,
#     ChatResponseType,
# )

# FastMCP imports
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import asyncio
import json
import math
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from fastapi.params import Body
import openai
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

# Define all required models directly in the file
class TaskType(str, Enum):
    QUIZ = "quiz"
    LEARNING_MATERIAL = "learning_material"
    
    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, TaskType):
            return self.value == other.value
        return False

class QuestionType(str, Enum):
    OPEN_ENDED = "subjective"
    OBJECTIVE = "objective"
    
    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, QuestionType):
            return self.value == other.value
        return False

class TaskAIResponseType(str, Enum):
    CHAT = "chat"
    EXAM = "exam"
    
    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, TaskAIResponseType):
            return self.value == other.value
        return False

class TaskInputType(str, Enum):
    CODE = "code"
    TEXT = "text"
    AUDIO = "audio"
    
    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, TaskInputType):
            return self.value == other.value
        return False

class ChatResponseType(str, Enum):
    TEXT = "text"
    CODE = "code"
    AUDIO = "audio"

class Block(BaseModel):
    type: str
    text: str

class DraftQuestion(BaseModel):
    blocks: List[Block]
    answer: List[Block]
    type: QuestionType
    input_type: TaskInputType
    response_type: TaskAIResponseType
    context: Optional[Dict] = None
    coding_languages: Optional[List[str]] = None
    scorecard_id: Optional[int] = None
    title: Optional[str] = None

class AIChatRequest(BaseModel):
    user_response: str
    task_type: TaskType
    question: Optional[DraftQuestion] = None
    chat_history: Optional[List[Dict]] = None
    question_id: Optional[int] = None
    user_id: int
    task_id: int
    response_type: Optional[ChatResponseType] = None
    hint: Optional[str] = None


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


class QuestionBankGenerator:
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client  

    async def generate_question_bank(self, kg: KnowledgeGraph, learning_material: str, bloom_tags: dict, num_mcqs: int = 10) -> Dict[str, Any]:

        """Generate question bank from enhanced knowledge graph and learning material"""
        
        # Extract high-entropy concepts to focus on
        high_entropy_nodes = sorted(
            [node for node in kg.nodes if node.entropy_score > 6.0],
            key=lambda x: x.entropy_score,
            reverse=True
        )[:10]  # Top 10 high-entropy concepts
        
        # Prepare concept list for prompt
        concept_list = "\n".join(
            f"- {node.label} (Entropy: {node.entropy_score:.1f}, Bloom Relevance: {node.properties.get('bloom_relevance', ['N/A'])})"
            for node in high_entropy_nodes
        )
        
        # Prepare the prompt for question generation
        prompt = f"""
    You are an expert educational question generator. Create questions based on BOTH the learning material and knowledge graph.

    LEARNING MATERIAL EXCERPTS:
    {learning_material[:2000]}... [truncated for brevity]

    KEY CONCEPTS FROM KNOWLEDGE GRAPH (sorted by importance):
    {concept_list}

    BLOOM'S TAXONOMY DISTRIBUTION:
    {json.dumps(bloom_tags, indent=2)}

    INSTRUCTIONS:
    1. Generate {num_mcqs} MCQs that DIRECTLY relate to the learning material
    2. For each question:
    - Base it on specific content from the learning material
    - Connect it to knowledge graph concepts
    - For MCQs: Provide 4 plausible options with one clearly correct answer
    - Include a helpful hint that references the learning material
    - Tag with the appropriate Bloom's level
    - Prioritize high-entropy concepts but maintain broad coverage

    QUESTION CRITERIA:
    - MCQs should test understanding of key concepts
    
    - Questions should be answerable from the material
    - Avoid trivial or overly obvious questions
    - Distractors should be plausible but incorrect
    - Ensure you don't repeat any key, value pairs in the generated structure

    OUTPUT FORMAT (JSON only):
    {{
        "questions": [
            {{
                "question_id": 1,
                "question_type": "MCQ",
                "question_text": "...",
                "material_reference": "...",
                "options": ["...", "...", "...", "..."],  // Only for MCQ
                "correct_answer": "...",
                "hint": "...",
                "bloom_level": "...",
                "entropy_score": 0.0,
                "concept_id": "...",
                "concept_label": "..."
            }}
        ],
        "metadata": {{
            "material_coverage": "...",
            "bloom_distribution": {{"remembering": 0, ...}},
            "entropy_distribution": {{"low": 0, "medium": 0, "high": 0}}
        }}
    }}
    """

        messages = [
            {"role": "system", "content": "You are an expert educational question generator. Return only valid JSON that strictly follows the format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.openai_client.generate_completion(messages)
            
            # Log the raw response for debugging
            logger.info(f"Raw OpenAI response length: {len(response)}")
            logger.info(f"Raw response: {response[:500]}...")
            
            # Clean and validate JSON response
            response = response.strip()
            if not response:
                logger.error("Empty response from OpenAI")
                return {"questions": [], "metadata": {"error": "Empty response"}}
            
            # Handle potential truncation
            if response.count('"') % 2 != 0:
                logger.warning("Unmatched quotes detected, attempting to fix")
                response = response + '"'
            
            # Ensure proper JSON structure
            if not response.endswith('}'):
                logger.warning("Response appears truncated, attempting to fix")
                last_brace = response.rfind('}')
                if last_brace > 0:
                    response = response[:last_brace + 1]
            
            question_data = None
            try:
                question_data = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                # logger.error(f"Response preview: {response[:200]}...")
                question_data = response
            # Try to extract valid JSON from response
            import re
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        question_data = json.loads(match)
                        break
                    except:
                        continue
            else:
                return {"questions": [], "metadata": {"error": "Invalid JSON format"}}

            # print(f"question_data: {question_data}")
            
            # Convert to QuizTask-compatible format
            formatted_questions = []
            # print(f"question_data keys: {question_data}")
            for q in question_data.get("questions", []):
                question_type = QuestionType.OBJECTIVE if q["question_type"] == "MCQ" else QuestionType.OPEN_ENDED
                
                # Create blocks for question content
                question_blocks = [
                    Block(type="paragraph", text=q["question_text"])
                ]
                
                # Create blocks for correct answer
                answer_blocks = [
                    Block(type="paragraph", text=q["correct_answer"])
                ]
                
                # Create options blocks for MCQ
                options_blocks = []
                if q["question_type"] == "MCQ":
                    for option in q["options"]:
                        options_blocks.append(Block(type="paragraph", text=option))
                
                # Compose context blocks with hint and material reference
                context_blocks = [
                    Block(type="paragraph", text=f"Hint: {q['hint']}"),
                    Block(type="paragraph", text=f"Reference: {q.get('material_reference', '')}")
                ]
                
                formatted_question = {
                    "question_id": q["question_id"],
                    "question_type": q["question_type"].lower(),
                    "blocks": [block.model_dump() for block in question_blocks],
                    "answer": [block.model_dump() for block in answer_blocks],
                    "options": [block.model_dump() for block in options_blocks] if options_blocks else None,
                    "hint": q["hint"],
                    "bloom_level": q["bloom_level"],
                    "entropy_score": q.get("entropy_score", 0.0),
                    "concept_id": q.get("concept_id", ""),
                    "concept_label": q.get("concept_label", ""),
                    "context": [block.model_dump() for block in context_blocks]
                }
                
                formatted_questions.append(formatted_question)
            
            return {
                "questions": formatted_questions,
                "metadata": question_data.get("metadata", {})
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing question bank JSON: {e}")
            return {"questions": [], "metadata": {}}
        except Exception as e:
            logger.error(f"Error generating question bank: {e}")
            return {"questions": [], "metadata": {}}
        
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
question_bank_generator_agent = QuestionBankGenerator(openai_client)

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
async def generate_question_bank(knowledge_graph: dict, learning_material: str, bloom_tags: dict, num_mcqs: int = 10) -> str:

    """
    Generate question bank from knowledge graph
    
    Args:
        knowledge_graph: Knowledge graph data structure
        learning_material: Original learning material text
        bloom_tags: Bloom's taxonomy distribution scores
        num_mcqs: Number of multiple choice questions to generate (default: 15)
    
    Returns:
        JSON string containing the generated question bank
    """
    try:
        logger.info("Generating question bank from knowledge graph...")
        
        # Reconstruct KG from data
        nodes = [KnowledgeGraphNode(**node) for node in knowledge_graph.get("nodes", [])]
        relationships = [KnowledgeGraphRelationship(**rel) for rel in knowledge_graph.get("relationships", [])]
        enhanced_kg = KnowledgeGraph(nodes=nodes, relationships=relationships, metadata=knowledge_graph.get("metadata", {}))
        
        question_result = await question_bank_generator_agent.generate_question_bank(
            kg=enhanced_kg,
            learning_material=learning_material,
            bloom_tags=bloom_tags,
            num_mcqs=num_mcqs,
        )
        
        logger.info("Question bank generated successfully")
        return question_result
        
    except Exception as e:
        logger.error(f"Error generating question bank: {str(e)}")
        return f"Error generating question bank: {str(e)}"
    
@mcp.tool()
async def full_pipeline(learning_material: str, bloom_tags: dict):
    try:    
        # Step 1: Construct knowledge graph
        kg = await kg_agent.construct_knowledge_graph(learning_material, bloom_tags)
        
        # Step 2: Generate entropy scores
        enhanced_kg = await entropy_agent.generate_entropy_scores(kg, learning_material, bloom_tags)
        
        # Step 3: Generate question bank
        question_result = await question_bank_generator_agent.generate_question_bank(
            enhanced_kg, learning_material, bloom_tags
        )

        # Ensure question_result is a dict (it might come as JSON string)
        if isinstance(question_result, str):
            try:
                question_result = json.loads(question_result)
            except json.JSONDecodeError:
                question_result = {"questions": [], "metadata": {}}

        # Convert to serializable format with proper error handling
        result = {
            "pipeline_status": "completed",
            "knowledge_graph": {
                "nodes": [n.__dict__ for n in enhanced_kg.nodes],
                "relationships": [r.__dict__ for r in enhanced_kg.relationships],
                "metadata": enhanced_kg.metadata
            },
            "question_bank": {
                "questions": question_result.get("questions", []),
                "metadata": {
                    **question_result.get("metadata", {}),
                    "total_questions": len(question_result.get("questions", [])),
                    "mcq_count": sum(1 for q in question_result.get("questions", []) 
                              if q.get("question_type", "").lower() == "mcq"),
                    "saq_count": sum(1 for q in question_result.get("questions", []) 
                              if q.get("question_type", "").lower() == "saq")
                }
            }
        }
        print(result)
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
        return json.dumps({
            "error": str(e),
            "pipeline_status": "failed"
        }, indent=2)
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
    
    @app.post("/generate_question_bank")
    async def generate_question_bank(knowledge_graph: dict, learning_material: str, bloom_tags: dict):
        return await mcp.call_tool("generate_question_bank", {
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