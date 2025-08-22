import os
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory, Response
from utils import ask_groq_stream, perform_web_search, clear_chat_history, execute_mcp_tools, MCP_SERVERS
import json

app = Flask(__name__)

@app.route("/")
def index():
    session_id = str(uuid.uuid4())
    return render_template("index.html", session_id=session_id)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route("/clear-history", methods=["POST"])
def clear_history():
    data = request.json
    session_id = data.get("session_id", "")
    
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    
    try:
        clear_chat_history(session_id)
        return jsonify({"status": "success", "message": "Chat history cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/mcp/servers", methods=["GET"])
def get_mcp_servers():
    """Get available MCP servers"""
    try:
        return jsonify({"servers": MCP_SERVERS, "status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/mcp/execute", methods=["POST"])
def execute_mcp():
    """Execute MCP tools directly"""
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Missing query"}), 400
    
    try:
        results = execute_mcp_tools(query)
        return jsonify({"results": results, "status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/ask-stream", methods=["GET", "POST"])
def ask_stream():
    if request.method == "GET":
        # Handle GET request (for EventSource)
        question = request.args.get('question', '')
        session_id = request.args.get('session_id', '')
        use_web_search = request.args.get('web_search', 'true').lower() == 'true'
    else:
        # Handle POST request
        data = request.json
        question = data.get("question", "")
        session_id = data.get("session_id", "")
        use_web_search = data.get("web_search", True)
    
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    def generate():
        try:
            # Perform web search only if web search is enabled
            web_results = []
            if use_web_search:
                web_results = perform_web_search(question)
            
            # Stream response from Groq with MCP integration
            for chunk in ask_groq_stream(session_id, question, web_results, use_web_search):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)