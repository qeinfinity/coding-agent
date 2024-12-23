# Coding Agent

A Python-based coding assistant that uses a four-layer learning approach to help solve programming tasks effectively.

## Features

- Four-layer learning approach:
  1. **Logic Layer**: High-level objectives and constraints
  2. **Concepts Layer**: Domain knowledge and patterns
  3. **Important Details Layer**: Specific implementation details
  4. **Arbitrary Details Layer**: Edge cases and legacy information
- Interactive command-line interface
- Configurable memory system
- OpenAI GPT integration

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd coding-agent
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your-api-key-here`
   - Or set it as an environment variable

## Usage

1. Run the agent:
```bash
python agent.py
```

2. Enter your coding questions when prompted. For example:
   - "How do I implement a REST API with Flask?"
   - "What's the best way to handle authentication in Django?"
   - "Show me how to use SQLAlchemy with PostgreSQL"

3. Type 'exit' or press Ctrl+C to quit

## Memory System

The agent uses a layered memory system stored in markdown files:

- `memory/layer_1_logic.md`: High-level principles and constraints
- `memory/layer_2_concepts.md`: Programming concepts and patterns
- `memory/layer_3_important_details.md`: Implementation details and code snippets
- `memory/layer_4_arbitrary.md`: Edge cases and legacy information

You can customize these files to add your own knowledge base.

## Configuration

- Modify `prompts/system_prompt.md` to adjust the agent's behavior
- Edit memory files to customize the knowledge base
- Adjust model parameters in `agent.py` (e.g., temperature, model choice)

## Requirements

- Python 3.9 or higher
- OpenAI API key
- Dependencies listed in requirements.txt

## License

MIT License
