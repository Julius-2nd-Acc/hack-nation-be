from langchain.callbacks.base import BaseCallbackHandler
import time
import sqlite3
import json
from .db import DB_PATH

class SQLiteTracer(BaseCallbackHandler):
    """Tracer that stores events in SQLite database and limits tool usage"""
    def __init__(self, message_id: str):
        self.message_id = message_id
        self.tool_usage_count = {}
        self.max_tool_calls = 3

    def _write(self, record):
        record.update({"ts": time.time(), "message_id": self.message_id})
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traces (message_id, event, data, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (self.message_id, record.get("event", "unknown"), json.dumps(record), record["ts"]))
        conn.commit()
        conn.close()

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._write({"event":"llm_start","prompts":prompts,"params":kwargs.get("invocation_params",{})})

    def on_llm_end(self, response, **kwargs):
        try:
            content = response.generations[0][0].text
        except Exception:
            content = str(response)
        tool_call = self._parse_tool_call(content)
        self._write({"event":"llm_end","output":content,"tool_call":tool_call})

    def _parse_tool_call(self, content: str):
        try:
            lines = content.strip().split('\n')
            action_line = None
            input_line = None
            for i, line in enumerate(lines):
                if line.startswith('Action:'):
                    action_line = line
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
                    if tool_input.startswith('"') and tool_input.endswith('"'):
                        tool_input = tool_input[1:-1]
                return {"tool_name": tool_name, "tool_input": tool_input}
        except Exception:
            pass
        return None
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        current_count = self.tool_usage_count.get(tool_name, 0)
        if current_count >= self.max_tool_calls:
            self._write({"event":"tool_limit_exceeded","tool":tool_name,"count":current_count,"limit":self.max_tool_calls})
            raise RuntimeError(f"CRITICAL: Tool limit exceeded for {tool_name}. Agent must stop immediately.")
        self.tool_usage_count[tool_name] = current_count + 1
        self._write({"event":"tool_start","tool":tool_name,"input":input_str,"usage_count":self.tool_usage_count[tool_name]})

    def on_tool_end(self, output, **kwargs):
        self._write({"event":"tool_end","output":output})

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = None
        if serialized:
            name = serialized.get("id") or serialized.get("name")
        self._write({"event": "chain_start", "name": name, "inputs": inputs})
        
    def on_chain_end(self, outputs, **kwargs):
        self._write({"event":"chain_end","outputs":outputs})
