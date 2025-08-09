# Super Capable AI Agent API 🚀

A powerful Flask API that runs a **super capable AI agent** with 13+ tools, GPT-4 reasoning, and comprehensive tracing capabilities. This agent can handle complex multi-step tasks, research, analysis, coding, file operations, and much more!

## 🔥 Agent Capabilities

### 🌐 Web & Research
- **Web Search**: Real-time web search using DuckDuckGo
- **URL Fetching**: Download and analyze web content
- **Data Analysis**: Statistical analysis of numerical data

### 💻 Programming & Code
- **Python Code Execution**: Run Python code safely with timeout
- **Mathematical Calculations**: Advanced math with trigonometry, statistics
- **JSON Processing**: Parse, validate, and manipulate JSON data

### 📁 File Operations  
- **File Reading**: Read any text file contents
- **File Writing**: Create and write files
- **Directory Listing**: Browse file systems

### 🌍 Utilities
- **Time & Timezone**: Get time in any timezone
- **Weather Info**: Current weather for any location
- **Translation**: Basic text translation
- **Report Generation**: Create formatted reports

### 🧠 Advanced Reasoning
- **GPT-4 Powered**: Uses GPT-4 for superior reasoning
- **Multi-step Planning**: Handles complex workflows
- **Tool Chaining**: Combines multiple tools intelligently

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Run the Flask app:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### POST /chat

Start a new conversation or continue an existing one.

**Request body:**
- New conversation: `{"prompt": "What time is it in New York?"}`
- Follow-up: `{"prompt": "And what about London?", "session_id": "uuid-here"}`

**Response:**
```json
{"session_id": "uuid-string"}
```

### GET /get

Get the complete trace of a session.

**Parameters:**
- `session_id`: The session ID to retrieve

**Example:**
```
GET /get?session_id=12345678-1234-1234-1234-123456789abc
```

**Response:**
```json
{
  "session": {
    "session_id": "uuid",
    "initial_prompt": "What time is it in New York?",
    "status": "completed",
    "final_output": "The current time in New York is...",
    "created_at": "2024-01-01 12:00:00",
    "completed_at": "2024-01-01 12:00:05"
  },
  "traces": [
    {"event": "llm_start", "prompts": [...], ...},
    {"event": "tool_start", "tool": "get_time", ...},
    ...
  ]
}
```

### GET /health

Health check endpoint.

## 🧪 Complex Example Questions

Here are some sophisticated questions to test the agent's capabilities:

### 📊 Data Analysis & Research
```json
{"prompt": "Search the web for information about Python's performance compared to JavaScript, then create a detailed report with statistical analysis if you find any benchmark data"}
```

### 💻 Programming & Problem Solving
```json
{"prompt": "Write a Python script that calculates the Fibonacci sequence up to the 20th number, execute it, then analyze the mathematical properties of the results"}
```

### 🔍 Multi-step Investigation
```json
{"prompt": "Find the current time in New York, London, and Tokyo. Then calculate the time differences between them and create a formatted report showing when it would be business hours (9 AM - 5 PM) in all three cities simultaneously"}
```

### 📁 File Operations & Analysis
```json
{"prompt": "List all files in the current directory, read the contents of any Python files you find, and write a summary report of what the code does"}
```

### 🌐 Web Research & Processing
```json
{"prompt": "Search for information about artificial intelligence trends in 2024, fetch content from any relevant URLs you find, and generate a comprehensive analysis report"}
```

### 🧮 Mathematical & Scientific Analysis
```json
{"prompt": "Calculate the area of a circle with radius 10, then find the volume of a sphere with the same radius. Compare these values and explain the mathematical relationship"}
```

### 🔄 Complex Workflow
```json
{"prompt": "Create a Python script that generates 100 random numbers, save them to a file called 'random_data.txt', then read the file back and perform statistical analysis on the data"}
```

## Basic Usage

```bash
# Start a new conversation
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Research the latest developments in quantum computing and create a technical report"}'

# Response: {"session_id": "abc123..."}

# Get the trace (wait a moment for completion)
curl "http://localhost:5000/get?session_id=abc123..."

# Continue the conversation
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Now write Python code to simulate a simple quantum gate operation", "session_id": "abc123..."}'
```

## 🔧 Advanced Features

- **Intelligent Tool Selection**: The agent automatically chooses the best tools for each task
- **Error Recovery**: Robust error handling with fallback strategies  
- **Session Continuity**: Maintains context across multiple requests
- **Comprehensive Tracing**: Every step is logged for debugging and analysis
- **Safe Code Execution**: Python code runs in isolated environment with timeouts
