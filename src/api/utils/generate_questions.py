import openai
import json
import os
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

current_file = Path(__file__)
# Navigate to the .env file location (go up to utils/, then up to api/, then to .env)
env_path = current_file.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def preprocess_learning_material(json_data):
    """
    Extract and clean content from structured learning material JSON
    """
    
    # Extract metadata
    metadata = {
        'id': json_data.id,
        'title': json_data.title,
        'type': json_data.type,
        'status': json_data.status,
        'scheduled_publish_at': json_data.scheduled_publish_at
    }
    
    # Process blocks to extract content
    processed_content = []
    
    for block in json_data.blocks:
        block_content = extract_block_content(block)
        if block_content:  # Skip empty blocks
            processed_content.append(block_content)
    
    return {
        'metadata': metadata,
        'content_blocks': processed_content,
        'full_text': ' '.join([block['text'] for block in processed_content if block['text']])
    }

def extract_block_content(block):
    """
    Extract content from individual blocks based on block type
    """
    block_type = block.type
    block_id = block.id
    position = block.position
    
    # Handle different block types
    if block_type == 'paragraph':
        text_content = extract_text_from_content(block.content)
        return {
            'id': block_id,
            'type': block_type,
            'position': position,
            'text': text_content,
            'props': block.props
        }
    
    # Add more block types as needed (heading, list, image, etc.)
    elif block_type == 'heading':
        # Handle heading blocks
        pass
    elif block_type == 'image':
        # Handle image blocks - you might want to store image URLs/descriptions
        pass
    
    return None

def extract_text_from_content(content_array):
    """
    Extract text from content array within a block
    """
    text_parts = []
    
    for content_item in content_array:
        if content_item.get('type') == 'text':
            text_parts.append(content_item.get('text'))
    
    return ' '.join(text_parts).strip()


def extract_blooms_taxonomy(learning_material: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> Dict[str, float]:
    """
    Extract Bloom's taxonomy distribution from learning material using OpenAI API
    
    Args:
        learning_material (str): The text content to analyze
        api_key (str, optional): OpenAI API key. If None, uses environment variable
        
    Returns:
        Dict[str, float]: Bloom's taxonomy distribution with values summing to 1.0
    """
    # Set up OpenAI client
    client_kwargs = {}

    if api_key:
        client_kwargs['api_key'] = api_key
    else:
        api_key_env = os.getenv('OPENAI_API_KEY')
        if not api_key_env:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env file or pass api_key parameter.")
        client_kwargs['api_key'] = api_key_env

    if base_url:
        client_kwargs['base_url'] = base_url
    else:
        base_url_env = os.getenv('OPENAI_API_BASE_URL')
        if base_url_env:
            client_kwargs['base_url'] = base_url_env
    
    client = openai.OpenAI(**client_kwargs)

    # Construct the prompt
    prompt = f"""
You are an educational content analyzer specialized in Bloom's Taxonomy classification.

Your task is to analyze the given learning material and extract learning objectives, then map them to Bloom's Taxonomy levels.

INSTRUCTIONS:
1. Extract Learning Objectives:
   - Identify key concepts and learning objectives from the text
   - Use NLP understanding and Bloom's Taxonomy verb mapping
   - Consider what students should be able to do after learning this material

2. Assign Bloom's Levels:
   - Remembering: Recall facts, basic concepts, answers
   - Understanding: Explain ideas, concepts, interpret information
   - Applying: Use information in new situations, solve problems
   - Analyzing: Draw connections, examine relationships, compare/contrast
   - Evaluating: Justify decisions, critique, assess value
   - Creating: Produce new work, combine elements, design solutions

3. Assign Weights and Normalize:
   - Map each learning objective to appropriate Bloom's level(s)
   - Calculate percentage distribution across all levels
   - Ensure all values sum to 1.0

LEARNING MATERIAL TO ANALYZE:
{learning_material}

OUTPUT FORMAT (JSON only, no explanation):
{{
    "remembering": 0.XX,
    "understanding": 0.XX,
    "applying": 0.XX,
    "analyzing": 0.XX,
    "evaluating": 0.XX,
    "creating": 0.XX
}}

Ensure all float values sum to exactly 1.0.
"""

    try:
        # Make API call to OpenAI
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",  # or "gpt-3.5-turbo" for faster/cheaper option
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert educational content analyzer. Return only valid JSON responses for Bloom's taxonomy analysis."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}  
        )
        
        # Extract and parse the response
        response_content = response.choices[0].message.content
        blooms_distribution = json.loads(response_content)
        
        # Validate and normalize the response
        validated_distribution = validate_and_normalize_blooms(blooms_distribution)
        
        return validated_distribution
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return get_default_blooms_distribution()
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return get_default_blooms_distribution()

def validate_and_normalize_blooms(blooms_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and normalize Bloom's taxonomy distribution
    """
    required_keys = ['remembering', 'understanding', 'applying', 'analyzing', 'evaluating', 'creating']
    
    # Ensure all required keys exist
    for key in required_keys:
        if key not in blooms_dict:
            blooms_dict[key] = 0.0
    
    # Convert values to float and handle any non-numeric values
    for key in required_keys:
        try:
            blooms_dict[key] = float(blooms_dict[key])
        except (ValueError, TypeError):
            blooms_dict[key] = 0.0
    
    # Calculate current sum
    current_sum = sum(blooms_dict[key] for key in required_keys)
    
    # Normalize to sum to 1.0
    if current_sum > 0:
        for key in required_keys:
            blooms_dict[key] = blooms_dict[key] / current_sum
    else:
        # If all values are 0, distribute evenly
        for key in required_keys:
            blooms_dict[key] = 1.0 / len(required_keys)
    
    # Round to avoid floating point precision issues
    for key in required_keys:
        blooms_dict[key] = round(blooms_dict[key], 4)
    
    # Final adjustment to ensure exact sum of 1.0
    total = sum(blooms_dict[key] for key in required_keys)
    if abs(total - 1.0) > 0.0001:
        # Adjust the largest value to make sum exactly 1.0
        max_key = max(required_keys, key=lambda k: blooms_dict[k])
        blooms_dict[max_key] += 1.0 - total
        blooms_dict[max_key] = round(blooms_dict[max_key], 4)
    
    return {key: blooms_dict[key] for key in required_keys}

def get_default_blooms_distribution() -> Dict[str, float]:
    """
    Return a default Bloom's taxonomy distribution in case of API failure
    """
    return {
        'remembering': 0.3,
        'understanding': 0.25,
        'applying': 0.2,
        'analyzing': 0.15,
        'evaluating': 0.05,
        'creating': 0.05
    }

# Usage example
if __name__ == "__main__":
    # Example usage
    sample_text = """
    Newton's three laws of motion describe the relationship between forces acting on a body 
    and its motion. The first law states that objects at rest stay at rest and objects in 
    motion stay in motion unless acted upon by an external force. The second law shows that 
    the acceleration of an object is directly proportional to the net force acting on it. 
    The third law states that for every action, there is an equal and opposite reaction.
    """
    
    # Set your OpenAI API key
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    try:
        blooms_tags = extract_blooms_taxonomy(sample_text)
        print("Bloom's Taxonomy Distribution:")
        for level, weight in blooms_tags.items():
            print(f"{level.capitalize()}: {weight:.3f} ({weight*100:.1f}%)")
        
        print(f"\nTotal sum: {sum(blooms_tags.values()):.3f}")
        
    except Exception as e:
        print(f"Error: {e}")