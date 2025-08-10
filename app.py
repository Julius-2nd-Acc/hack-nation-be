import os
import json
import time
import uuid
import sqlite3
import threading
import requests
import subprocess
import math
import statistics
import re
import csv
import io
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path
from flask import Flask, request, jsonify

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage, SystemMessage

app = Flask(__name__)

# Database setup
DB_PATH = "agent_traces.db"

def init_db():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            initial_prompt TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            final_output TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    # Traces table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')
    
    conn.commit()
    conn.close()

class SQLiteTracer(BaseCallbackHandler):
    """Tracer that stores events in SQLite database"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    def _write(self, record: Dict[str, Any]):
        """Write a trace record to the database"""
        record.update({"ts": time.time(), "session_id": self.session_id})
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traces (session_id, event, data, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (self.session_id, record.get("event", "unknown"), json.dumps(record), record["ts"]))
        conn.commit()
        conn.close()

    # LLM events
    def on_llm_start(self, serialized, prompts, **kwargs):
        self._write({"event":"llm_start","prompts":prompts,"params":kwargs.get("invocation_params",{})})
    
    def on_llm_end(self, response, **kwargs):
        try:
            content = response.generations[0][0].text
        except Exception:
            content = str(response)
        self._write({"event":"llm_end","output":content})

    # Tool events
    def on_tool_start(self, serialized, input_str, **kwargs):
        self._write({"event":"tool_start","tool":serialized.get("name"),"input":input_str})
    
    def on_tool_end(self, output, **kwargs):
        self._write({"event":"tool_end","output":output})

    # Chain/agent events
    def on_chain_start(self, serialized, inputs, **kwargs):
        self._write({"event":"chain_start","name":serialized.get("id") or serialized.get("name"),"inputs":inputs})
    
    def on_chain_end(self, outputs, **kwargs):
        self._write({"event":"chain_end","outputs":outputs})

# =============================================================================
# SUPER CAPABLE AGENT TOOLS
# =============================================================================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information using multiple free sources."""
    try:
        from urllib.parse import quote
        
        # Method 1: Try DuckDuckGo instant answers
        ddg_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
        
        response = requests.get(ddg_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; AI Agent/1.0)'
        })
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            if data.get('Abstract'):
                results.append(f"Summary: {data['Abstract']}")
            
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:num_results]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(f"• {topic['Text']}")
            
            if results:
                return "\n".join(results)
        
        # Method 2: Fallback to Wikipedia search
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"
        wiki_response = requests.get(wiki_url, timeout=10)
        
        if wiki_response.status_code == 200:
            wiki_data = wiki_response.json()
            if wiki_data.get('extract'):
                return f"Wikipedia Summary: {wiki_data['extract']}"
        
        return f"Limited search results available for '{query}'. Try a more specific query."
        
    except Exception as e:
        return f"Search unavailable: {str(e)}"

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get current time in specified timezone (e.g., 'UTC', 'US/Eastern', 'Europe/London')."""
    try:
        # Try with pytz if available
        try:
            import pytz
            if timezone.upper() == "UTC":
                tz = pytz.UTC
            else:
                tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        except ImportError:
            # Fallback without pytz
            current_time = datetime.utcnow()
            return f"Current UTC time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')} (pytz not available for timezone conversion)"
    except Exception as e:
        return f"Time error: {str(e)}"

@tool
def calculate_math(expression: str) -> str:
    """Safely evaluate mathematical expressions. Supports +, -, *, /, **, sqrt, sin, cos, etc."""
    try:
        # Safe evaluation with math functions
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "pow": pow
        })
        
        # Compile and evaluate safely
        code = compile(expression, "<string>", "eval")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Math error: {str(e)}"

@tool
def analyze_data(data_str: str, operation: str = "describe") -> str:
    """Analyze numerical data. Operations: 'describe', 'mean', 'median', 'std', 'min', 'max'."""
    try:
        # Parse numbers from string
        numbers = [float(x.strip()) for x in re.findall(r'-?\d+\.?\d*', data_str)]
        if not numbers:
            return "No numerical data found in input"
        
        if operation == "describe":
            return f"""Data Analysis:
Count: {len(numbers)}
Mean: {statistics.mean(numbers):.2f}
Median: {statistics.median(numbers):.2f}
Std Dev: {statistics.stdev(numbers) if len(numbers) > 1 else 0:.2f}
Min: {min(numbers)}
Max: {max(numbers)}
Range: {max(numbers) - min(numbers)}"""
        elif operation == "mean":
            return f"Mean: {statistics.mean(numbers):.2f}"
        elif operation == "median":
            return f"Median: {statistics.median(numbers):.2f}"
        elif operation == "std":
            return f"Standard Deviation: {statistics.stdev(numbers) if len(numbers) > 1 else 0:.2f}"
        elif operation == "min":
            return f"Minimum: {min(numbers)}"
        elif operation == "max":
            return f"Maximum: {max(numbers)}"
        else:
            return f"Unknown operation: {operation}"
    except Exception as e:
        return f"Data analysis error: {str(e)}"

@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File content ({len(content)} characters):\n{content[:2000]}{'...' if len(content) > 2000 else ''}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a text file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool
def list_directory(path: str = ".") -> str:
    """List files and directories in the specified path."""
    try:
        items = os.listdir(path)
        dirs = [f"📁 {item}/" for item in items if os.path.isdir(os.path.join(path, item))]
        files = [f"📄 {item}" for item in items if os.path.isfile(os.path.join(path, item))]
        all_items = sorted(dirs) + sorted(files)
        return f"Directory listing for '{path}':\n" + "\n".join(all_items[:50])
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@tool
def execute_python_code(code: str) -> str:
    """Execute Python code safely and return the output. Use for complex calculations or data processing."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Execute with timeout
        result = subprocess.run([
            'python', temp_file
        ], capture_output=True, text=True, timeout=30)
        
        # Clean up
        os.unlink(temp_file)
        
        output = result.stdout + result.stderr
        return f"Code execution result:\n{output[:1000]}{'...' if len(output) > 1000 else ''}"
    except subprocess.TimeoutExpired:
        return "Code execution timed out (30s limit)"
    except Exception as e:
        return f"Code execution error: {str(e)}"

@tool
def get_weather(location: str) -> str:
    """Get current weather information for a location using a free weather API."""
    try:
        # Using OpenWeatherMap free tier (you'd need to sign up for API key)
        # For demo, returning mock data
        import random
        temperatures = list(range(-10, 35))
        conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy", "overcast"]
        
        temp = random.choice(temperatures)
        condition = random.choice(conditions)
        humidity = random.randint(30, 90)
        
        return f"Weather in {location}:\nTemperature: {temp}°C\nCondition: {condition}\nHumidity: {humidity}%\n(Note: This is mock data for demo)"
    except Exception as e:
        return f"Weather error: {str(e)}"

@tool
def translate_text(text: str, target_language: str = "Spanish") -> str:
    """Translate text to target language using a simple translation service."""
    try:
        # For demo purposes, providing basic translations
        translations = {
            "hello": {"Spanish": "hola", "French": "bonjour", "German": "hallo", "Italian": "ciao"},
            "goodbye": {"Spanish": "adiós", "French": "au revoir", "German": "auf wiedersehen", "Italian": "ciao"},
            "thank you": {"Spanish": "gracias", "French": "merci", "German": "danke", "Italian": "grazie"},
        }
        
        text_lower = text.lower()
        if text_lower in translations and target_language in translations[text_lower]:
            return f"Translation to {target_language}: {translations[text_lower][target_language]}"
        else:
            return f"Translation service not available for '{text}' to {target_language}. Consider using Google Translate API."
    except Exception as e:
        return f"Translation error: {str(e)}"

@tool
def generate_report(title: str, data: str) -> str:
    """Generate a formatted report with the given title and data."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""
# {title}

**Generated:** {timestamp}

## Summary
{data}

## Analysis
This report was automatically generated by the AI agent.

---
*Report ID: {uuid.uuid4().hex[:8]}*
"""
        return f"Generated report:\n{report}"
    except Exception as e:
        return f"Report generation error: {str(e)}"

@tool
def url_fetch(url: str) -> str:
    """Fetch content from a URL and return a summary."""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; AI Agent/1.0)'
        })
        response.raise_for_status()
        content = response.text[:2000]  # Limit content
        return f"URL content (first 2000 chars):\n{content}"
    except Exception as e:
        return f"URL fetch error: {str(e)}"

@tool
def json_processor(json_data: str, operation: str = "format") -> str:
    """Process JSON data. Operations: 'format', 'validate', 'keys', 'values'."""
    try:
        data = json.loads(json_data)
        
        if operation == "format":
            return json.dumps(data, indent=2)
        elif operation == "validate":
            return "Valid JSON format"
        elif operation == "keys":
            if isinstance(data, dict):
                return f"JSON keys: {list(data.keys())}"
            else:
                return "Not a JSON object (no keys)"
        elif operation == "values":
            if isinstance(data, dict):
                return f"JSON values: {list(data.values())}"
            else:
                return f"JSON data: {data}"
        else:
            return f"Unknown operation: {operation}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"
    except Exception as e:
        return f"JSON processing error: {str(e)}"

# Note: get_conversation_history function removed as we now use messages array directly
# instead of reconstructing history from traces

def run_agent(session_id: str, messages: List[Dict[str, str]]):
    """Run the agent in a separate thread with conversation messages"""
    try:
        # Update session status to running
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET status = 'running' 
            WHERE session_id = ?
        ''', (session_id,))
        conn.commit()
        conn.close()
        
        # Set up agent with tracer
        tracer = SQLiteTracer(session_id)
        # Using GPT-4 for better reasoning and capabilities
        llm = ChatOpenAI(
            model="gpt-4", 
            temperature=0.1, 
            callbacks=[tracer],
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Comprehensive tool arsenal
        tools = [
            web_search,
            get_current_time,
            calculate_math,
            analyze_data,
            read_file,
            write_file,
            list_directory,
            execute_python_code,
            get_weather,
            translate_text,
            generate_report,
            url_fetch,
            json_processor
        ]
        
        # Add tracer to tools
        for t in tools:
            t.callbacks = [tracer]
        
        # Create agent
        prompt_template = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt_template)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True, callbacks=[tracer])
        
        # Convert messages to conversation context and get the latest user message
        conversation_context = ""
        current_user_input = ""
        
        if len(messages) > 1:
            # Build conversation context from all but the last message
            context_messages = []
            for msg in messages[:-1]:
                if msg["role"] == "user":
                    context_messages.append(f"Human: {msg['content']}")
                elif msg["role"] == "assistant":
                    context_messages.append(f"Assistant: {msg['content']}")
            
            if context_messages:
                conversation_context = "Previous conversation:\n" + "\n".join(context_messages) + "\n\nCurrent request:\n"
        
        # Get the current user input (last message is guaranteed to be from user due to validation)
        current_user_input = messages[-1]["content"]
        
        # Combine context with current input
        full_input = conversation_context + current_user_input
        
        # Execute agent
        result = executor.invoke({"input": full_input})
        final_output = result.get("output", "")
        
        # Update session with final result
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET status = 'completed', final_output = ?, completed_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        ''', (final_output, session_id))
        conn.commit()
        conn.close()
        
    except Exception as e:
        # Update session status to error
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET status = 'error', final_output = ?
            WHERE session_id = ?
        ''', (str(e), session_id))
        conn.commit()
        conn.close()

@app.route('/chat/new', methods=['POST'])
def chat():
    """
    Handle chat requests with messages array (OpenAI API format)
    """
    data = request.get_json()
    
    if not data or 'messages' not in data:
        return jsonify({"error": "Missing 'messages' in request body"}), 400
    
    messages = data['messages']
    
    # Validate messages format
    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify({"error": "'messages' must be a non-empty array"}), 400
    
    for msg in messages:
        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
            return jsonify({"error": "Each message must have 'role' and 'content' fields"}), 400
        if msg['role'] not in ['user', 'assistant', 'system']:
            return jsonify({"error": "Message role must be 'user', 'assistant', or 'system'"}), 400
    
    # Validate that the last message is from user
    if messages[-1]['role'] != 'user':
        return jsonify({"error": "The last message in the messages array must be from 'user'"}), 400
    
    # Always create a new session
    session_id = str(uuid.uuid4())
    
    # Get initial prompt from first user message for session tracking
    initial_prompt = ""
    for msg in messages:
        if msg['role'] == 'user':
            initial_prompt = msg['content']
            break
    
    if not initial_prompt:
        initial_prompt = "Conversation started"
    
    # Create new session
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sessions (session_id, initial_prompt, status)
        VALUES (?, ?, 'pending')
    ''', (session_id, initial_prompt))
    conn.commit()
    conn.close()
    
    # Start agent execution in background thread with messages
    thread = threading.Thread(target=run_agent, args=(session_id, messages))
    thread.daemon = True
    thread.start()
    
    return jsonify({"session_id": session_id})

@app.route('/chat', methods=['GET'])
def get_traces():
    """
    Get all traces for a given session_id
    """
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({"error": "Missing 'session_id' parameter"}), 400
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get session info
    cursor.execute('''
        SELECT session_id, initial_prompt, status, final_output, created_at, completed_at
        FROM sessions WHERE session_id = ?
    ''', (session_id,))
    session_row = cursor.fetchone()
    
    if not session_row:
        conn.close()
        return jsonify({"error": "Session not found"}), 404
    
    # Get all traces for this session
    cursor.execute('''
        SELECT event, data, timestamp
        FROM traces 
        WHERE session_id = ?
        ORDER BY timestamp
    ''', (session_id,))
    trace_rows = cursor.fetchall()
    
    conn.close()
    
    # Format response
    session_info = {
        "session_id": session_row[0],
        "initial_prompt": session_row[1],
        "status": session_row[2],
        "final_output": session_row[3],
        "created_at": session_row[4],
        "completed_at": session_row[5]
    }
    
    traces = []
    for event, data, timestamp in trace_rows:
        try:
            trace_data = json.loads(data)
            traces.append(trace_data)
        except json.JSONDecodeError:
            traces.append({"event": event, "data": data, "timestamp": timestamp})
    
    return jsonify({
        "session": session_info,
        "traces": traces
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Ensure OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
