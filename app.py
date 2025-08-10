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


# Import agent tools from lib.tools.agent_tools
from lib.tools.agent_tools import *

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

        now = datetime.now().astimezone()
        current_time = now.strftime("%H:%M:%S %Z")

        # Put conversation context and user input FIRST (most important)
        main_context = conversation_context + user_input
        
        # Add system context at the end (supplementary)
        system_context = f"""

[System Information - Current date: {current_day}, {current_date}, Current system time: {current_time}]

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


# Register routes from routes.api_routes
from routes.api_routes import register_routes

if __name__ == '__main__':
    # Initialize database
    init_db()
    # Register routes
    register_routes(app)
    # Ensure OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True, reloader_type='stat')
