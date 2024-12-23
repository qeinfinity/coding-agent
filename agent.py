#!/usr/bin/env python3
import os
import openai
from typing import List, Optional, Dict, Tuple
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

def get_completion(messages: List[Dict[str, str]], model: str = "gpt-4") -> str:
    """
    Get completion from OpenAI API.
    
    Args:
        messages: List of message dictionaries
        model: Model to use for completion
    
    Returns:
        Model's response text
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_reflection_prompt(solution: str) -> str:
    """
    Generate a reflection prompt for analyzing a solution.
    
    Args:
        solution: The solution to analyze
    
    Returns:
        A prompt for reflection analysis
    """
    reflection_prompt = load_text_file("prompts/reflection_prompt.md")
    if not reflection_prompt:
        # Fallback reflection prompt if file doesn't exist
        reflection_prompt = """
        Analyze the solution provided across each layer:
        1. Logic Layer: Was the high-level approach sound?
        2. Concepts Layer: Were relevant patterns/principles applied?
        3. Important Details Layer: Were crucial implementation details included?
        4. Arbitrary Layer: Were edge cases considered appropriately?

        Summarize:
        1. Key strengths
        2. Areas for improvement
        3. Additional context needed (if any)
        """
    
    return f"{reflection_prompt}\n\nSolution to analyze:\n{solution}"

def run_agent_with_reflection(user_query: str, api_key: Optional[str] = None) -> Tuple[str, str]:
    """
    Run the coding agent with reflection.
    
    Args:
        user_query: The user's coding question or task
        api_key: Optional OpenAI API key
    
    Returns:
        Tuple of (initial_response, reflection)
    """
    # Get API key from parameter or environment
    api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key_to_use:
        raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

    # Load system prompt
    system_prompt = load_text_file("prompts/system_prompt.md")
    if not system_prompt:
        raise FileNotFoundError("System prompt file not found")

    # Initial layers to load
    layers_to_load = [1, 2, 3]
    if any(keyword in user_query.lower() for keyword in ['legacy', 'old version', 'deprecated', 'edge case']):
        layers_to_load.append(4)

    memory_snippets = gather_memory(layers_to_load)

    # Get initial solution
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Relevant memory:\n{memory_snippets}"},
        {"role": "user", "content": user_query}
    ]

    initial_response = get_completion(messages)
    
    # Generate reflection
    reflection_prompt = generate_reflection_prompt(initial_response)
    reflection_messages = [
        {"role": "system", "content": "You are a code review assistant performing a reflection analysis."},
        {"role": "user", "content": reflection_prompt}
    ]
    
    reflection = get_completion(reflection_messages)
    
    return initial_response, reflection

def main():
    """Main function to run the agent interactively."""
    print("=== Coding Agent ===")
    print("Enter your coding questions (Ctrl+C or type 'exit' to quit)")
    print("Example: 'How do I implement a REST API with Flask?'")
    
    while True:
        try:
            print("\nEnter your question:")
            user_input = input("> ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            initial_response, reflection = run_agent_with_reflection(user_input)
            
            print("\n=== Agent's Response ===")
            print(initial_response)
            print("\n=== Reflection Analysis ===")
            print(reflection)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()