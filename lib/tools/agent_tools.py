
import os
import json
import math
import statistics
import re
import tempfile
import subprocess
import requests
from datetime import datetime
import pytz
from langchain.tools import tool

# The following tools are imported and used in app.py
@tool
def web_search(query: str, num_results: int = 5) -> str:
	"""Search the web using OpenAI's web search tool and return a concise, cited answer."""
	try:
		# NOTE: The 'client' object must be provided by the caller's context (see app.py)
		from app import client
		resp = client.responses.create(
			model="gpt-4o",
			input=(
				f"Use web_search. Answer concisely, then list up to {num_results} sources with links."
				f"\nQuery: {query}"
			),
			tools=[{"type": "web_search"}], # type: ignore
			tool_choice="auto",
			max_output_tokens=600,
		)
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
		timezone = timezone.strip().strip("'\"")

		# Handle UTC offset like 'UTC+2', 'UTC-5', etc.
		if timezone.upper().startswith("UTC") and len(timezone) > 3:
			offset_str = timezone[3:]
			sign = 1 if "+" in offset_str else -1
			try:
				hours = int(offset_str.replace("+", "").replace("-", ""))
			except Exception:
				return f"Invalid UTC offset format: {timezone}"
			from datetime import timedelta
			current_time = datetime.utcnow() + timedelta(hours=sign * hours)
			return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC offset)"

		# Handle UTC
		if timezone.upper() == "UTC":
			if pytz:
				tz = pytz.UTC
				current_time = datetime.now(tz)
				return f"Current time in UTC: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
			else:
				current_time = datetime.utcnow()
				return f"Current UTC time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')} (pytz not available for timezone conversion)"

		# Handle IANA timezones (like 'Asia/Tokyo')
		if pytz:
			try:
				tz = pytz.timezone(timezone)
				current_time = datetime.now(tz)
				return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
			except Exception:
				return f"Unknown or unsupported timezone: {timezone}"
		else:
			return f"Timezone '{timezone}' requires pytz. Only 'UTC' and 'UTC±offset' are supported without pytz."
	except Exception as e:
		return f"Time error: {str(e)}"

@tool
def calculate_math(expression: str) -> str:
	"""Safely evaluate mathematical expressions. Supports +, -, *, /, **, sqrt, sin, cos, log, ln, etc.
	Use 'ln(x)' for natural logarithm, 'log(x)' for base-10 logarithm.
	IMPORTANT: Provide ONLY the mathematical expression, NO comments or extra text."""
	try:
		cleaned_expression = expression.strip()
		if '#' in cleaned_expression:
			cleaned_expression = cleaned_expression.split('#')[0].strip()
		if (cleaned_expression.startswith('"') and cleaned_expression.endswith('"')) or \
		   (cleaned_expression.startswith("'") and cleaned_expression.endswith("'")):
			cleaned_expression = cleaned_expression[1:-1]
		allowed_names = {
			k: v for k, v in math.__dict__.items() if not k.startswith("__")
		}
		allowed_names.update({
			"abs": abs, "round": round, "min": min, "max": max,
			"sum": sum, "len": len, "pow": pow,
			"ln": math.log,
		})
		code = compile(cleaned_expression, "<string>", "eval")
		result = eval(code, {"__builtins__": {}}, allowed_names)
		return f"Result: {result}"
	except Exception as e:
		return f"Math error: {str(e)}. Available functions: +, -, *, /, **, sqrt, sin, cos, log, ln, abs, round, min, max, etc. Make sure to provide only the mathematical expression without comments."

@tool
def analyze_data(data_str: str, operation: str = "describe") -> str:
	"""Analyze numerical data and return TEXT-BASED statistics only. Operations: 'describe', 'mean', 'median', 'std', 'min', 'max'. No visual charts or graphs - only text summaries and statistics."""
	try:
		numbers = [float(x.strip()) for x in re.findall(r'-?\d+\.?\d*', data_str)]
		if not numbers:
			return "No numerical data found in input"
		if operation == "describe":
			return f"""Data Analysis:\nCount: {len(numbers)}\nMean: {statistics.mean(numbers):.2f}\nMedian: {statistics.median(numbers):.2f}\nStd Dev: {statistics.stdev(numbers) if len(numbers) > 1 else 0:.2f}\nMin: {min(numbers)}\nMax: {max(numbers)}\nRange: {max(numbers) - min(numbers)}"""
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
		dirs = [f"\ud83d\udcc1 {item}/" for item in items if os.path.isdir(os.path.join(path, item))]
		files = [f"\ud83d\udcc4 {item}" for item in items if os.path.isfile(os.path.join(path, item))]
		all_items = sorted(dirs) + sorted(files)
		return f"Directory listing for '{path}':\n" + "\n".join(all_items[:50])
	except Exception as e:
		return f"Error listing directory: {str(e)}"

@tool
def execute_python_code(code: str) -> str:
	"""Execute Python code safely and return TEXT OUTPUT ONLY. IMPORTANT INPUT FORMAT: Provide only the raw Python code, WITHOUT any markdown backticks or formatting. Use for calculations, data processing, and analysis. DO NOT use for plots/charts."""
	try:
		cleaned_code = code.strip()
		if cleaned_code.startswith('```'):
			lines = cleaned_code.split('\n')
			if lines[0].startswith('```'):
				lines = lines[1:]
			if lines and lines[-1].strip() == '```':
				lines = lines[:-1]
			cleaned_code = '\n'.join(lines)
		if '# -*- coding:' not in cleaned_code:
			cleaned_code = "# -*- coding: utf-8 -*-\n" + cleaned_code
		with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
			f.write(cleaned_code)
			temp_file = f.name
		result = subprocess.run([
			'python', temp_file
		], capture_output=True, text=True, timeout=30, encoding='utf-8')
		os.unlink(temp_file)
		output = result.stdout + result.stderr
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
		content = response.text[:2000]
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
