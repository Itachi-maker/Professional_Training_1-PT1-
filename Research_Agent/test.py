from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

app = Flask(__name__)
CORS(app)

# Simple HTML page embedded in the app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Research Agent Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .form-group { margin: 15px 0; }
        input, button { padding: 10px; margin: 5px 0; }
        input[type="text"] { width: 300px; }
        button { background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #dee2e6; }
        .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .loading { color: #007bff; }
    </style>
</head>
<body>
    <h1>Research Agent Test</h1>
    
    <div class="form-group">
        <input type="text" id="company" placeholder="Company Name (e.g., Google)" />
    </div>
    
    <div class="form-group">
        <input type="text" id="role" placeholder="Job Role (e.g., Software Engineer)" />
    </div>
    
    <button onclick="search()">Search</button>
    
    <div id="result"></div>
    
    <script>
        async function search() {
            const company = document.getElementById('company').value;
            const role = document.getElementById('role').value;
            const resultDiv = document.getElementById('result');
            
            if (!company || !role) {
                resultDiv.innerHTML = '<div class="result error">Please fill both fields</div>';
                return;
            }
            
            resultDiv.innerHTML = '<div class="result loading">Searching...</div>';
            
            try {
                const response = await fetch('/api/test-search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ company, role })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `<div class="result">
                        <h3>Results for ${data.company} - ${data.role}</h3>
                        <pre>${data.summary}</pre>
                    </div>`;
                } else {
                    resultDiv.innerHTML = `<div class="result error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">Connection Error: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

class SimpleResearchAgent:
    def __init__(self):
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.tavily_url = "https://api.tavily.com/search"
    
    def search_tavily(self, query: str) -> list:
        """Simple Tavily search"""
        if not self.tavily_api_key:
            return []
        
        try:
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "max_results": 3
            }
            
            response = requests.post(self.tavily_url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def simple_research(self, company: str, role: str) -> str:
        """Simple research function"""
        print(f"Researching: {company} - {role}")
        
        # Single search to test connectivity
        query = f"{company} {role} job requirements"
        results = self.search_tavily(query)
        
        if not results:
            return "No search results found. Please check your API key and internet connection."
        
        # Simple summary
        summary = f"Found {len(results)} results for {company} - {role}:\n\n"
        
        for i, result in enumerate(results[:3], 1):
            summary += f"{i}. {result.get('title', 'No title')}\n"
            summary += f"   {result.get('content', 'No content')[:200]}...\n\n"
        
        return summary

# Initialize agent
agent = SimpleResearchAgent()

@app.route('/')
def home():
    return HTML_TEMPLATE

@app.route('/api/test-search', methods=['POST'])
def test_search():
    try:
        data = request.get_json()
        print(f"Received request: {data}")
        
        if not data or not data.get('company') or not data.get('role'):
            return jsonify({'success': False, 'error': 'Missing company or role'}), 400
        
        if not agent.tavily_api_key:
            return jsonify({'success': False, 'error': 'Tavily API key not configured'}), 500
        
        company = data['company'].strip()
        role = data['role'].strip()
        
        summary = agent.simple_research(company, role)
        
        return jsonify({
            'success': True,
            'company': company,
            'role': role,
            'summary': summary
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'tavily_configured': bool(agent.tavily_api_key)})

if __name__ == '__main__':
    print("🚀 Starting Simple Research Agent Test Server...")
    print("📍 Open: http://localhost:5000")
    print("🔑 Tavily API Key:", "✅ Configured" if agent.tavily_api_key else "❌ Missing")
    
    app.run(debug=True, host='0.0.0.0', port=5000)