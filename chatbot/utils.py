import os
from groq import Groq
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize clients and models
groq_client = Groq(api_key=GROQ_API_KEY)

# Web search settings
MAX_SEARCH_RESULTS = 5

# In-memory storage for chat history
chat_history = {}

# MCP Servers configuration
MCP_SERVERS = {
    "web_search": {
        "enabled": True,
        "name": "Web Search",
        "description": "Perform real-time web searches"
    },
    "calculator": {
        "enabled": True,
        "name": "Calculator",
        "description": "Perform mathematical calculations"
    },
    "time_date": {
        "enabled": True,
        "name": "Time & Date",
        "description": "Get current time and date information"
    },
    "json_processor": {
        "enabled": True,
        "name": "JSON Processor",
        "description": "Process and analyze JSON data"
    }
}

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return ''.join(char for char in text if char.isprintable())

def perform_web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list:
    """Perform web search using Serper API and return results"""
    try:
        if not SERPER_API_KEY:
            print("Serper API key not found. Please set SERPER_API_KEY in your .env file.")
            return []
        
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": max_results
        }
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Extract organic results
        results = []
        if 'organic' in data:
            for item in data['organic']:
                results.append({
                    'title': item.get('title', ''),
                    'href': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
        
        return results
        
    except Exception as e:
        print(f"Web search error: {str(e)}")
        return []

def fetch_web_content(url: str) -> str:
    """Fetch and clean content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Simple HTML content extraction
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return clean_text(text[:3000])  # Limit content length
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return ""

def get_domain_name(url: str) -> str:
    """Extract domain name from URL"""
    try:
        parsed_uri = urlparse(url)
        domain = parsed_uri.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return url

def clear_chat_history(session_id: str):
    """Clear chat history for a specific session"""
    global chat_history
    if session_id in chat_history:
        del chat_history[session_id]
    return True

# MCP Server Functions
def mcp_calculator(expression: str) -> str:
    """MCP Calculator - Evaluate mathematical expressions"""
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        # Use numpy for safe evaluation
        result = eval(expression, {"__builtins__": None}, {
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
            "pi": np.pi, "e": np.e
        })
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

def mcp_time_date() -> str:
    """MCP Time & Date - Get current time and date"""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"

def mcp_json_process(json_data: str, operation: str = "analyze") -> str:
    """MCP JSON Processor - Analyze or manipulate JSON data"""
    try:
        data = json.loads(json_data)
        
        if operation == "analyze":
            if isinstance(data, dict):
                keys = list(data.keys())
                return f"JSON object with keys: {keys}"
            elif isinstance(data, list):
                return f"JSON array with {len(data)} items"
            else:
                return f"JSON value: {type(data).__name__}"
        
        elif operation == "pretty":
            return json.dumps(data, indent=2)
            
        else:
            return "Available operations: analyze, pretty"
            
    except Exception as e:
        return f"Error processing JSON: {str(e)}"

def execute_mcp_tools(query: str) -> List[Dict[str, Any]]:
    """Execute MCP tools based on query content"""
    results = []
    
    # Calculator MCP
    if any(word in query.lower() for word in ["calculate", "math", "equation", "solve", "+", "-", "*", "/"]):
        try:
            # Extract mathematical expression
            import re
            math_pattern = r"(\d+\.?\d*[\s*+\-/\\*^()\s\d+\.?\d*]*)"
            matches = re.findall(math_pattern, query)
            if matches:
                expression = matches[0].strip()
                result = mcp_calculator(expression)
                results.append({
                    "tool": "calculator",
                    "result": result,
                    "expression": expression
                })
        except Exception as e:
            results.append({
                "tool": "calculator",
                "result": f"Error: {str(e)}",
                "expression": "N/A"
            })
    
    # Time & Date MCP
    if any(word in query.lower() for word in ["time", "date", "now", "current", "today", "what day"]):
        result = mcp_time_date()
        results.append({
            "tool": "time_date",
            "result": result
        })
    
    # JSON Processor MCP
    if "json" in query.lower():
        # Look for JSON data in the query
        json_pattern = r"(\{.*\}|\[.*\])"
        matches = re.findall(json_pattern, query, re.DOTALL)
        if matches:
            json_data = matches[0]
            result = mcp_json_process(json_data)
            results.append({
                "tool": "json_processor",
                "result": result,
                "data_preview": json_data[:100] + "..." if len(json_data) > 100 else json_data
            })
    
    return results

def ask_groq_stream(session_id: str, question: str, web_results: list = None, use_web_search: bool = True):
    """Stream Groq model response with optional web results, MCP tools, and chat history"""
    history_text = ""
    if session_id in chat_history:
        for q, a in chat_history[session_id][-5:]:
            history_text += f"Q: {q}\nA: {a}\n\n"
    
    # Execute MCP tools
    mcp_results = execute_mcp_tools(question)
    mcp_context = ""
    if mcp_results:
        mcp_context = "\nMCP TOOL RESULTS:\n"
        for i, result in enumerate(mcp_results):
            mcp_context += f"[Tool {i+1}: {result['tool']}]\n{result['result']}\n\n"
    
    # Prepare web context if available and web search is enabled
    web_context = ""
    sources = set()
    
    if use_web_search and web_results and len(web_results) > 0:
        web_context = "\nWEB SEARCH RESULTS:\n"
        for i, result in enumerate(web_results):
            if i >= MAX_SEARCH_RESULTS:
                break
                
            # Use the snippet from search results
            content = result.get('snippet', '')
            if not content:
                content = fetch_web_content(result.get('href', ''))
            
            if content:
                domain = get_domain_name(result.get('href', ''))
                title = result.get('title', 'No title')
                web_context += f"[Source {i+1}: {title} - {domain}]\n{content}\n\n"
                sources.add(domain)
    
    # Different prompts based on whether web search is enabled
    if use_web_search:
        prompt = f"""You are an AI assistant that answers questions using web search results and MCP tools when available.
You must follow these rules:
1. Use information from the WEB SEARCH RESULTS and MCP TOOL RESULTS below when relevant
2. If conversation starts with Hi, Hello, or Hey, respond with a friendly greeting
3. If conversation starts with What, Where, When, Who, or Why, respond with a detailed answer
4. If using web search results, cite your sources with numbers in brackets like [1], [2], etc.
5. If using MCP tools, mention that you used specialized tools for the response
6. If the answer isn't in the web results, use your general knowledge
7. Be concise but helpful
8. Always provide accurate information

{mcp_context}

{web_context}

CHAT HISTORY:
{history_text}

QUESTION:
{question}

ANSWER:"""
    else:
        prompt = f"""You are an AI assistant that answers questions based on your general knowledge and MCP tools.
You must follow these rules:
1. Use your general knowledge and MCP TOOL RESULTS to answer questions
2. If conversation starts with Hi, Hello, or Hey, respond with a friendly greeting
3. If conversation starts with What, Where, When, Who, or Why, respond with a detailed answer
4. If using MCP tools, mention that you used specialized tools for the response
5. Be concise but helpful
6. Always provide accurate information

{mcp_context}

CHAT HISTORY:
{history_text}

QUESTION:
{question}

ANSWER:"""
    
    try:
        full_response = ""
        stream = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield {"token": token, "complete": False}
        
        # Add source references if we have them and web search is enabled
        if use_web_search and sources:
            source_text = f"\n\nSources: {', '.join(sources)}"
            full_response += source_text
            yield {"token": source_text, "complete": False}
        
        # Add MCP tool references if we used them
        if mcp_results:
            tools_used = ", ".join([result['tool'] for result in mcp_results])
            tools_text = f"\n\nTools used: {tools_used}"
            full_response += tools_text
            yield {"token": tools_text, "complete": False}
        
        # Save to chat history
        if session_id not in chat_history:
            chat_history[session_id] = []
        chat_history[session_id].append((question, full_response))
        
        yield {"complete": True}
    
    except Exception as e:
        yield {"error": str(e), "complete": True}