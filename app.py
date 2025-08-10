import os

from flask import Flask
from lib.db import init_db
from lib.config import client
from lib.agent_runner import run_agent
from routes.api_routes import register_routes

app = Flask(__name__)

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
