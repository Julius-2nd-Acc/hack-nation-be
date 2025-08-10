
import sqlite3
import uuid
import json
from flask import request, jsonify
from . import *

DB_PATH = "agent_traces.db"

def register_routes(app):
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
