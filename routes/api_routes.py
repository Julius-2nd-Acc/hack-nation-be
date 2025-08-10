
import sqlite3
import uuid
import json
from flask import request, jsonify
import sys
from lib.agent_runner import run_agent
from lib.tracing import SQLiteTracer
from lib.utils import replace_current_request, replace_before_system_info
import tempfile
import os
from . import *

DB_PATH = "agent_traces.db"

def register_routes(app):

	@app.route('/chat/replay', methods=['POST'])
	def chat_replay():
		sys.modules.pop('app', None)  # Prevent circular import if present
		data = request.get_json()
		history = data.get('history')
		traces = data.get('traces')
		user_prompt = data.get('user_prompt') or None
		
		if not traces or not isinstance(traces, list) or len(traces) == 0:
			return jsonify({"error": "You must provide a non-empty 'traces' array in the request body."}), 400
		last_event = traces[-1]
		if last_event["event"] not in ("chain_start", "llm_start"):
			return jsonify({"error": "Last trace event must be 'chain_start' or 'llm_start'."}), 400
		# Extract replay input
		if last_event["event"] == "chain_start":
			replay_input = last_event.get("inputs", {}).get("input", "")
			
            # replace if user_prompt is there
			if user_prompt:
				if replay_input.startswith("Previous conversation:\nHuman:"):
					replay_input = replace_current_request(replay_input, user_prompt)
				else:
					replay_input = replace_before_system_info(replay_input, user_prompt)
		elif last_event["event"] == "llm_start":
			prompts = last_event.get("prompts", [])
			replay_input = prompts[0] if prompts else ""
		else:
			replay_input = ""
		# Reconstruct conversation context from history if provided
		conversation_context = ""
		if history and isinstance(history, list):
			context_parts = []
			for msg in history:
				role = msg.get("role")
				content = msg.get("content", "")
				if role == "user":
					context_parts.append(f"Human: {content}")
				elif role == "assistant":
					context_parts.append(f"Assistant: {content}")
				elif role == "system":
					context_parts.append(f"System: {content}")
			if context_parts:
				conversation_context = "Previous conversation:\n" + "\n".join(context_parts) + "\n\nCurrent request:\n"
		# Use a temp session_id and message_id for replay
		import uuid
		session_id = str(uuid.uuid4())
		message_id = str(uuid.uuid4())
		# Use a temp DB for traces (in-memory)
		temp_db_path = os.path.join(tempfile.gettempdir(), f"replay_traces_{message_id}.db")
		# Initialize temp DB schema
		import sqlite3
		conn = sqlite3.connect(temp_db_path)
		cursor = conn.cursor()
		cursor.execute('''CREATE TABLE IF NOT EXISTS traces (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			message_id TEXT,
			event TEXT,
			data TEXT,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
		)''')
		conn.commit()
		conn.close()
		# Run the agent synchronously, capturing traces in temp DB
		tracer = SQLiteTracer(message_id, db_path=temp_db_path)
		try:
			output = run_agent(session_id, message_id, replay_input, conversation_context, tracer=tracer)
			# Collect traces from temp DB
			conn = sqlite3.connect(temp_db_path)
			cursor = conn.cursor()
			cursor.execute('SELECT event, data, timestamp FROM traces WHERE message_id = ? ORDER BY timestamp', (message_id,))
			trace_rows = cursor.fetchall()
			traces_out = []
			for event, data, timestamp in trace_rows:
				try:
					trace_data = json.loads(data)
					traces_out.append(trace_data)
				except Exception:
					traces_out.append({"event": event, "data": data, "timestamp": timestamp})
			conn.close()
		finally:
			if os.path.exists(temp_db_path):
				os.remove(temp_db_path)
		return jsonify({
			"replay_input": replay_input,
			"replay_start_event": last_event["event"],
			"output": output,
			"traces": traces_out
		})
	
	@app.route('/chat/new', methods=['POST'])
	def chat():
		data = request.get_json()
		if not data or 'prompt' not in data:
			return jsonify({"error": "Missing 'prompt' in request body"}), 400
		prompt = data['prompt']
		session_id = data.get('session_id')
		if not isinstance(prompt, str) or not prompt.strip():
			return jsonify({"error": "'prompt' must be a non-empty string"}), 400
		conn = sqlite3.connect(DB_PATH)
		cursor = conn.cursor()
		try:
			if session_id:
				cursor.execute('SELECT session_id FROM sessions WHERE session_id = ?', (session_id,))
				if not cursor.fetchone():
					return jsonify({"error": "Session not found"}), 404
			else:
				session_id = str(uuid.uuid4())
				cursor.execute('''
					INSERT INTO sessions (session_id)
					VALUES (?)
				''', (session_id,))
			user_message_id = str(uuid.uuid4())
			cursor.execute('''
				INSERT INTO messages (message_id, session_id, role, content, status)
				VALUES (?, ?, 'user', ?, 'completed')
			''', (user_message_id, session_id, prompt.strip()))
			assistant_message_id = str(uuid.uuid4())
			cursor.execute('''
				INSERT INTO messages (message_id, session_id, role, content, status)
				VALUES (?, ?, 'assistant', '', 'pending')
			''', (assistant_message_id, session_id))
			conn.commit()
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
			from app import run_agent
			import threading
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
		conn = sqlite3.connect(DB_PATH)
		cursor = conn.cursor()
		cursor.execute('''
			SELECT session_id, created_at, updated_at
			FROM sessions WHERE session_id = ?
		''', (session_id,))
		session_row = cursor.fetchone()
		if not session_row:
			conn.close()
			return jsonify({"error": "Session not found"}), 404
		cursor.execute('''
			SELECT message_id, role, content, status, created_at, completed_at
			FROM messages 
			WHERE session_id = ?
			ORDER BY created_at
		''', (session_id,))
		message_rows = cursor.fetchall()
		conn.close()
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
		conn = sqlite3.connect(DB_PATH)
		cursor = conn.cursor()
		cursor.execute('''
			SELECT message_id, session_id, role, content, status, created_at, completed_at
			FROM messages WHERE message_id = ?
		''', (message_id,))
		message_row = cursor.fetchone()
		if not message_row:
			conn.close()
			return jsonify({"error": "Message not found"}), 404
		if message_row[2] != 'assistant':
			conn.close()
			return jsonify({"error": "Only assistant messages have execution traces"}), 400
		cursor.execute('''
			SELECT event, data, timestamp
			FROM traces 
			WHERE message_id = ?
			ORDER BY timestamp
		''', (message_id,))
		trace_rows = cursor.fetchall()
		conn.close()
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
	
	@app.route('/sessions/<session_id>/traces', methods=['GET'])
	def get_session_traces(session_id):
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
		# Get all message_ids for this session
		cursor.execute('''
			SELECT message_id FROM messages WHERE session_id = ?
		''', (session_id,))
		message_ids = [row[0] for row in cursor.fetchall()]
		# Get all traces for these message_ids
		traces = []
		for message_id in message_ids:
			cursor.execute('''
				SELECT event, data, timestamp FROM traces WHERE message_id = ? ORDER BY timestamp
			''', (message_id,))
			trace_rows = cursor.fetchall()
			for event, data, timestamp in trace_rows:
				try:
					trace_data = json.loads(data)
					traces.append(trace_data)
				except json.JSONDecodeError:
					traces.append({"event": event, "data": data, "timestamp": timestamp})
		conn.close()
		session_info = {
			"session_id": session_row[0],
			"created_at": session_row[1],
			"updated_at": session_row[2]
		}
		return jsonify({
			"session": session_info,
			"traces": traces
		})

	@app.route('/chat', methods=['GET'])
	def get_legacy_traces():
		session_id = request.args.get('session_id')
		if session_id:
			return jsonify({
				"error": "This endpoint is deprecated. Use /sessions/{session_id} instead.",
				"migration_note": "The new API uses session_id and message_id for better tracking"
			}), 410
		else:
			return jsonify({"error": "Missing 'session_id' parameter"}), 400

	@app.route('/', methods=['GET'])
	def health():
		return jsonify({"status": "healthy"})
