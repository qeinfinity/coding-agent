#!/usr/bin/env python3
import os
import openai
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_text_file(file_path: str) -> str:
    """Load content from a text file."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def gather_memory(layers_needed: List[int]) -> str:
    """
    Gather content from memory layers.
    
    Args:
        layers_needed: List of layer numbers to include (1-4)
    
    Returns:
        Combined content from specified memory layers
    """
    memory_content = []
    base_dir = "memory"
    
    layer_files = {
        1: "layer_1_logic.md",
        2: "layer_2_concepts.md",
        3: "layer_3_important_details.md",
        4: "layer_4_arbitrary.md"
    }
    
    for layer in layers_needed:
        if layer in layer_files:
            content = load_text_file(os.path.join(base_dir, layer_files[layer]))
            if content:
                memory_content.append(f"=== Layer {layer} Knowledge ===\n{content}\n")
    
    return "\n".join(memory_content)

def run_agent(user_query: str, api_key: Optional[str] = None) -> str:
    """
    Run the coding agent with the given query.
    
    Args:
        user_query: The user's coding question or task
        api_key: Optional OpenAI API key (can also be set via OPENAI_API_KEY env var)
    
    Returns:
        The agent's response
    """
    # Get API key from parameter or environment
    api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key_to_use:
        raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

    # Load system prompt
    system_prompt = load_text_file("prompts/system_prompt.md")
    if not system_prompt:
        raise FileNotFoundError("System prompt file not found")

    # For initial implementation, we'll always gather layers 1-3
    # Layer 4 is only loaded for specific types of queries (could be enhanced later)
    layers_to_load = [1, 2, 3]
    if any(keyword in user_query.lower() for keyword in ['legacy', 'old version', 'deprecated', 'edge case']):
        layers_to_load.append(4)

    memory_snippets = gather_memory(layers_to_load)

    # Construct the messages for the API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Relevant memory:\n{memory_snippets}"},
        {"role": "user", "content": user_query}
    ]

    try:
        # Create OpenAI client
        client = openai.OpenAI()
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Can be changed to gpt-4 or other models
            messages=messages,
            temperature=0.3  # Lower temperature for more focused coding responses
        )

        # Extract and return the response
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main function to run the agent interactively."""
    print("=== Coding Agent ===")
    print("Enter your coding questions (Ctrl+C to exit)")
    print("Example: 'How do I implement a REST API with Flask?'")
    
    while True:
        try:
            print("\nEnter your question:")
            user_input = input("> ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            result = run_agent(user_input)
            print("\n=== Agent's Response ===")
            print(result)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
