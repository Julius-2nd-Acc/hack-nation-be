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
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app and OpenAI client
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database setup
DB_PATH = "agent_traces.db"

def init_db():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sessions table - represents a conversation thread
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Messages table - individual messages within sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            status TEXT DEFAULT 'completed' CHECK (status IN ('pending', 'running', 'completed', 'error')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')
    
    # Traces table - execution traces for assistant messages
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT NOT NULL,
            event TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (message_id) REFERENCES messages (message_id)
        )
    ''')
    
    conn.commit()
    conn.close()

class SQLiteTracer(BaseCallbackHandler):
    """Tracer that stores events in SQLite database and limits tool usage"""
    
    def __init__(self, message_id: str):
        self.message_id = message_id
        self.tool_usage_count = {}  # Track tool usage per tool name
        self.max_tool_calls = 3  # Reduced from 5 to prevent endless loops
    
    def _write(self, record: Dict[str, Any]):
        """Write a trace record to the database"""
        record.update({"ts": time.time(), "message_id": self.message_id})
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traces (message_id, event, data, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (self.message_id, record.get("event", "unknown"), json.dumps(record), record["ts"]))
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
        
        # Parse tool calls from ReAct format
        tool_call = self._parse_tool_call(content)
        self._write({"event":"llm_end","output":content,"tool_call":tool_call})
    
    def _parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from ReAct format LLM output"""
        try:
            lines = content.strip().split('\n')
            action_line = None
            input_line = None
            
            for i, line in enumerate(lines):
                if line.startswith('Action:'):
                    action_line = line
                    # Look for Action Input on next lines
                    for j in range(i+1, min(i+3, len(lines))):
                        if lines[j].startswith('Action Input:'):
                            input_line = lines[j]
                            break
                    break
            
            if action_line:
                tool_name = action_line.replace('Action:', '').strip()
                tool_input = ""
                
                if input_line:
                    tool_input = input_line.replace('Action Input:', '').strip()
                    # Remove quotes if present
                    if tool_input.startswith('"') and tool_input.endswith('"'):
                        tool_input = tool_input[1:-1]
                
                return {
                    "tool_name": tool_name,
                    "tool_input": tool_input
                }
        except Exception:
            pass
        
        return None

    # Tool events
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        
        # Check if tool has been called too many times
        current_count = self.tool_usage_count.get(tool_name, 0)
        if current_count >= self.max_tool_calls:
            error_msg = f"STOP! Tool '{tool_name}' has been called {current_count} times. Maximum allowed is {self.max_tool_calls}. You must try a completely different approach now."
            self._write({"event":"tool_limit_exceeded","tool":tool_name,"count":current_count,"limit":self.max_tool_calls})
            # Force stop by raising a critical exception
            raise RuntimeError(f"CRITICAL: Tool limit exceeded for {tool_name}. Agent must stop immediately.")
        
        # Increment usage count
        self.tool_usage_count[tool_name] = current_count + 1
        
        self._write({"event":"tool_start","tool":tool_name,"input":input_str,"usage_count":self.tool_usage_count[tool_name]})
    
    def on_tool_end(self, output, **kwargs):
        self._write({"event":"tool_end","output":output})

    # Chain/agent events
    def on_chain_start(self, serialized, inputs, **kwargs):
        name = None
        if serialized:
            name = serialized.get("id") or serialized.get("name")
        self._write({"event": "chain_start", "name": name, "inputs": inputs})
    
    def on_chain_end(self, outputs, **kwargs):
        self._write({"event":"chain_end","outputs":outputs})

# =============================================================================
# SUPER CAPABLE AGENT TOOLS
# =============================================================================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using OpenAI's web search tool and return a concise, cited answer."""
    try:
        resp = client.responses.create(
            model="gpt-4o",
            input=(
                f"Use web_search. Answer concisely, then list up to {num_results} sources with links."
                f"\nQuery: {query}"
            ),
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            max_output_tokens=600,
        )
        # Prefer SDK convenience field; fall back to manual parse
        text = getattr(resp, "output_text", "") or ""
        if not text and getattr(resp, "output", None):
            parts = []
            for item in resp.output:
                if getattr(item, "type", "") == "output_text":
                    for seg in getattr(item, "content", []) or []:
                        if hasattr(seg, "text"):
                            parts.append(seg.text)
            text = "\n".join(parts).strip()
        return text or "No results."
    except Exception as e:
        return f"Search unavailable: {e}"

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
    """Safely evaluate mathematical expressions. Supports +, -, *, /, **, sqrt, sin, cos, log, ln, etc.
    Use 'ln(x)' for natural logarithm, 'log(x)' for base-10 logarithm.
    
    IMPORTANT: Provide ONLY the mathematical expression, NO comments or extra text."""
    try:
        # Debug logging
        print(f"DEBUG: Original expression: '{expression}'")
        
        # Clean the input - remove comments and extra text
        cleaned_expression = expression.strip()
        
        # Remove comments (anything after #)
        if '#' in cleaned_expression:
            cleaned_expression = cleaned_expression.split('#')[0].strip()
        
        # Remove quotes if present (both single and double quotes)
        if (cleaned_expression.startswith('"') and cleaned_expression.endswith('"')) or \
           (cleaned_expression.startswith("'") and cleaned_expression.endswith("'")):
            cleaned_expression = cleaned_expression[1:-1]
        
        print(f"DEBUG: Cleaned expression: '{cleaned_expression}'")
        
        # Safe evaluation with math functions
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "pow": pow,
            "ln": math.log,  # Add natural logarithm as 'ln'
        })
        
        # Compile and evaluate safely
        code = compile(cleaned_expression, "<string>", "eval")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Math error: {str(e)}. Available functions: +, -, *, /, **, sqrt, sin, cos, log, ln, abs, round, min, max, etc. Make sure to provide only the mathematical expression without comments."

@tool
def analyze_data(data_str: str, operation: str = "describe") -> str:
    """Analyze numerical data and return TEXT-BASED statistics only. Operations: 'describe', 'mean', 'median', 'std', 'min', 'max'. 
    No visual charts or graphs - only text summaries and statistics."""
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
    """Execute Python code safely and return TEXT OUTPUT ONLY. 
    
    IMPORTANT INPUT FORMAT: Provide only the raw Python code, WITHOUT any markdown backticks or formatting.
    
    CORRECT: import math; print(math.sqrt(16))
    WRONG: ```python\\nimport math\\nprint(math.sqrt(16))\\n```
    
    Use for calculations, data processing, and analysis. DO NOT use for plots/charts."""
    try:
        # Clean up the input - remove common formatting issues
        cleaned_code = code.strip()
        
        # Remove markdown code block formatting if present
        if cleaned_code.startswith('```'):
            lines = cleaned_code.split('\n')
            # Remove first line if it's ```python or similar
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_code = '\n'.join(lines)
        
        # Add encoding declaration for international characters
        if '# -*- coding:' not in cleaned_code:
            cleaned_code = "# -*- coding: utf-8 -*-\n" + cleaned_code
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(cleaned_code)
            temp_file = f.name
        
        # Execute with timeout
        result = subprocess.run([
            'python', temp_file
        ], capture_output=True, text=True, timeout=30, encoding='utf-8')
        
        # Clean up
        os.unlink(temp_file)
        
        output = result.stdout + result.stderr
        
        # Check for common formatting errors and provide helpful feedback
        if "SyntaxError" in output and "```" in code:
            return "Code execution error: You included markdown backticks (```) in your code. Please provide only raw Python code without any markdown formatting."
        
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

def run_agent(session_id: str, message_id: str, user_input: str, conversation_context: str = ""):
    """Run the agent in a separate thread with message context"""
    try:
        print(f"Starting agent execution for message {message_id[:8]}... in session {session_id[:8]}...")
        print(f"User input: {user_input[:100]}...")
        
        # Update message status to running
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET status = 'running' 
            WHERE message_id = ?
        ''', (message_id,))
        conn.commit()
        conn.close()
        
        # Set up agent with tracer
        tracer = SQLiteTracer(message_id)
        llm = ChatOpenAI(
            model="gpt-4o", 
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
            url_fetch,
            json_processor
        ]
        
        # Add tracer to tools
        for t in tools:
            t.callbacks = [tracer]
        
        # Create agent
        prompt_template = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt_template)
        executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            callbacks=[tracer],
            handle_parsing_errors=True,  # Allow recovery from parsing errors
            max_iterations=5,  # Further reduced to prevent endless loops
            max_execution_time=30,  # Reduced to 30 seconds timeout
            return_intermediate_steps=False  # Simplify output
        )
        
        # Build context with conversation first, then current input, then system context
        current_date = datetime.now().strftime("%B %d, %Y")
        current_day = datetime.now().strftime("%A")
        
        # Put conversation context and user input FIRST (most important)
        main_context = conversation_context + user_input
        
        # Add system context at the end (supplementary)
        system_context = f"""

[System Information - Current date: {current_day}, {current_date}]

[Operating Guidelines: TEXT-ONLY environment]
- Provide text-based responses only
- No visual plots, charts, or graphs
- Use text tables/lists for data visualization

[CRITICAL TOOL USAGE RULES - READ CAREFULLY]

IMPORTANT: You MUST follow the ReAct format exactly:

1. TO USE A TOOL:
Thought: I need to calculate something
Action: tool_name
Action Input: raw_input_value

2. TO GIVE FINAL ANSWER (END THE CONVERSATION):
Final Answer: Your complete response here.

EXAMPLES:
✅ CORRECT TOOL USAGE:
Thought: I need to calculate the square root of 16
Action: calculate_math
Action Input: sqrt(16)

✅ CORRECT MATH INPUT:
Action Input: 1 * ln(0.9) / ln(0.9)

❌ WRONG MATH INPUT (causes errors):
Action Input: "1 * ln(0.9) / ln(0.9)"  # With quotes and comments
Action Input: "1 * ln(0.9) / ln(0.9)"  # First interval calculation

✅ CORRECT FINAL ANSWER:
Final Answer: The square root of 16 is 4.

❌ WRONG - CAUSES INFINITE LOOPS:
Thought: Here's my answer... (WITHOUT Action: or Final Answer:)

RULES:
- NEVER include markdown backticks (```) in Action Input!
- After getting your answer, use "Final Answer:" to end
- NEVER have "Thought:" without either "Action:" or "Final Answer:" following it
- If a tool fails 3+ times, try a different approach!
"""
        
        full_input = main_context + system_context
        
        # Debug logging
        print(f"Context length: {len(conversation_context)} chars")
        print(f"User input: {user_input}")
        print(f"Full input preview: {full_input[:200]}...")
        if len(conversation_context) > 0:
            print(f"Conversation context preview: {conversation_context[:150]}...")
        
        # Execute agent with fallback
        try:
            result = executor.invoke({"input": full_input})
            final_output = result.get("output", "")
            
            # Check if agent hit limits or gave irrelevant response (should be marked as error)
            is_agent_error = (
                not final_output or 
                len(final_output.strip()) < 10 or
                "iteration limit" in final_output.lower() or
                "time limit" in final_output.lower() or
                "stopped due to" in final_output.lower()
            )
            
            # Check for ReAct format loops (Invalid Format errors)
            is_format_loop = (
                final_output and
                ("Invalid Format:" in final_output or
                 "Missing 'Action:' after 'Thought:'" in final_output or
                 final_output.count("Invalid Format") > 1)
            )
            
            # Check for completely irrelevant responses (just date, etc.)
            is_irrelevant = (
                final_output and 
                len(final_output.strip()) < 100 and 
                ("current date is" in final_output.lower() or
                 final_output.strip().startswith("The current date") or
                 final_output.strip().startswith("Today is"))
            )
            
            if is_agent_error or is_irrelevant or is_format_loop:
                if not final_output or len(final_output.strip()) < 10:
                    final_output = "I encountered technical difficulties while processing your request. Please try rephrasing your question or ask something else."
                elif is_irrelevant:
                    final_output = "I apologize, but I gave an irrelevant response. Please let me try to answer your question properly."
                    raise Exception(f"Agent gave irrelevant response: {final_output}")
                elif is_format_loop:
                    final_output = "I encountered a formatting error loop and couldn't complete the task properly. Please try rephrasing your question."
                    raise Exception(f"Agent stuck in format loop: {final_output}")
                else:
                    raise Exception(f"Agent execution failed: {final_output}")
            
            print(f"Agent execution completed successfully for message {message_id[:8]}...")
            print(f"Output length: {len(final_output)} characters")
            
        except Exception as agent_error:
            print(f"Agent execution error for message {message_id[:8]}: {str(agent_error)}")
            # Provide a context-aware fallback response using direct LLM call, but mark as error
            try:
                # Build a context-aware fallback prompt
                fallback_prompt = f"""Please respond to this query in a helpful way. Use only text - no visual elements.

{conversation_context}Current query: {user_input}

Provide a relevant, helpful response that addresses the user's question in the context of our conversation."""

                fallback_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": fallback_prompt
                    }],
                    max_tokens=500
                )
                if fallback_response.choices and fallback_response.choices[0].message:
                    final_output = fallback_response.choices[0].message.content or "I'm having technical difficulties. Please try again."
                else:
                    final_output = "I'm experiencing technical difficulties. Please try again later."
                print(f"Used context-aware fallback response for message {message_id[:8]}")
            except Exception as fallback_error:
                print(f"Fallback also failed for message {message_id[:8]}: {str(fallback_error)}")
                final_output = "I'm experiencing technical difficulties and cannot process your request at the moment. Please try again later."
            
            # Mark as error since agent failed
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE messages 
                SET status = 'error', content = ?, completed_at = CURRENT_TIMESTAMP
                WHERE message_id = ?
            ''', (final_output, message_id))
            
            # Update session updated_at
            cursor.execute('''
                UPDATE sessions 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (session_id,))
            
            conn.commit()
            conn.close()
            return  # Exit early since we handled the error case
        
        # Update message with final result
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET status = 'completed', content = ?, completed_at = CURRENT_TIMESTAMP
            WHERE message_id = ?
        ''', (final_output, message_id))
        
        # Update session updated_at
        cursor.execute('''
            UPDATE sessions 
            SET updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        # Log error to console for debugging - this catches unexpected errors not handled above
        print(f"UNEXPECTED ERROR in run_agent for message {message_id}: {str(e)}")
        print(f"Session: {session_id}, User input: {user_input[:100]}...")
        
        # Update message status to error with a user-friendly message
        user_friendly_error = "I encountered an unexpected error while processing your request. Please try again later."
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET status = 'error', content = ?, completed_at = CURRENT_TIMESTAMP
            WHERE message_id = ?
        ''', (user_friendly_error, message_id))
        
        # Update session updated_at
        cursor.execute('''
            UPDATE sessions 
            SET updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()

@app.route('/chat/new', methods=['POST'])
def chat():
    """
    Handle chat requests with simplified format
    Accepts: prompt (required) and session_id (optional)
    Returns session_id and message_id for the assistant response
    """
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400
    
    prompt = data['prompt']
    session_id = data.get('session_id')  # Optional session_id for continuing conversations
    
    # Validate prompt
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "'prompt' must be a non-empty string"}), 400
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Handle existing vs new conversation
        if session_id:
            # Validate that the session exists
            cursor.execute('SELECT session_id FROM sessions WHERE session_id = ?', (session_id,))
            if not cursor.fetchone():
                return jsonify({"error": "Session not found"}), 404
        else:
            # Create new conversation
            session_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO sessions (session_id)
                VALUES (?)
            ''', (session_id,))
        
        # Store the new user message
        user_message_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO messages (message_id, session_id, role, content, status)
            VALUES (?, ?, 'user', ?, 'completed')
        ''', (user_message_id, session_id, prompt.strip()))
        
        # Create a new assistant message for the response (initially pending)
        assistant_message_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO messages (message_id, session_id, role, content, status)
            VALUES (?, ?, 'assistant', '', 'pending')
        ''', (assistant_message_id, session_id))
        
        conn.commit()
        
        # Build conversation context from database history (excluding current message and errors)
        cursor.execute('''
            SELECT role, content FROM messages 
            WHERE session_id = ? AND message_id != ? AND status = 'completed'
            ORDER BY created_at
        ''', (session_id, user_message_id))
        
        history_messages = cursor.fetchall()
        conversation_context = ""
        
        if history_messages:
            context_parts = []
            for role, content in history_messages:
                if role == "user":
                    context_parts.append(f"Human: {content}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content}")
                elif role == "system":
                    context_parts.append(f"System: {content}")
            
            conversation_context = "Previous conversation:\n" + "\n".join(context_parts) + "\n\nCurrent request:\n"
        
        # Start agent execution in background thread
        thread = threading.Thread(target=run_agent, args=(session_id, assistant_message_id, prompt.strip(), conversation_context))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "session_id": session_id, 
            "message_id": assistant_message_id
        })
        
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """
    Get all messages in a session
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get session info
    cursor.execute('''
        SELECT session_id, created_at, updated_at
        FROM sessions WHERE session_id = ?
    ''', (session_id,))
    session_row = cursor.fetchone()
    
    if not session_row:
        conn.close()
        return jsonify({"error": "Session not found"}), 404
    
    # Get all messages for this session
    cursor.execute('''
        SELECT message_id, role, content, status, created_at, completed_at
        FROM messages 
        WHERE session_id = ?
        ORDER BY created_at
    ''', (session_id,))
    message_rows = cursor.fetchall()
    
    conn.close()
    
    # Format response
    session_info = {
        "session_id": session_row[0],
        "created_at": session_row[1],
        "updated_at": session_row[2]
    }
    
    messages = []
    for message_id, role, content, status, created_at, completed_at in message_rows:
        message = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "status": status,
            "created_at": created_at,
            "completed_at": completed_at
        }
        messages.append(message)
    
    return jsonify({
        "session": session_info,
        "messages": messages
    })

@app.route('/messages/<message_id>/traces', methods=['GET'])
def get_message_traces(message_id):
    """
    Get all traces for a specific message (assistant messages only)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get message info
    cursor.execute('''
        SELECT message_id, session_id, role, content, status, created_at, completed_at
        FROM messages WHERE message_id = ?
    ''', (message_id,))
    message_row = cursor.fetchone()
    
    if not message_row:
        conn.close()
        return jsonify({"error": "Message not found"}), 404
    
    # Only assistant messages have traces
    if message_row[2] != 'assistant':  # role is at index 2
        conn.close()
        return jsonify({"error": "Only assistant messages have execution traces"}), 400
    
    # Get all traces for this message
    cursor.execute('''
        SELECT event, data, timestamp
        FROM traces 
        WHERE message_id = ?
        ORDER BY timestamp
    ''', (message_id,))
    trace_rows = cursor.fetchall()
    
    conn.close()
    
    # Format response
    message_info = {
        "message_id": message_row[0],
        "session_id": message_row[1],
        "role": message_row[2],
        "content": message_row[3],
        "status": message_row[4],
        "created_at": message_row[5],
        "completed_at": message_row[6]
    }
    
    traces = []
    for event, data, timestamp in trace_rows:
        try:
            trace_data = json.loads(data)
            traces.append(trace_data)
        except json.JSONDecodeError:
            traces.append({"event": event, "data": data, "timestamp": timestamp})
    
    return jsonify({
        "message": message_info,
        "traces": traces
    })

# Legacy endpoint for backward compatibility (deprecated)
@app.route('/chat', methods=['GET'])
def get_legacy_traces():
    """
    Legacy endpoint - redirects to new conversation-based API
    """
    session_id = request.args.get('session_id')
    if session_id:
        return jsonify({
            "error": "This endpoint is deprecated. Use /sessions/{session_id} instead.",
            "migration_note": "The new API uses session_id and message_id for better tracking"
        }), 410  # Gone
    else:
        return jsonify({"error": "Missing 'session_id' parameter"}), 400

@app.route('/', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Ensure OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True, reloader_type='stat')
