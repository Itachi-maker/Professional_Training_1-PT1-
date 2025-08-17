from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import json
import time
import re
from typing import Dict, List, Any
from bs4 import BeautifulSoup
import html

load_dotenv()

app = Flask(__name__)
CORS(app)

class ContentCleaner:
    """Utility class to clean and extract meaningful content"""
    
    @staticmethod
    def clean_html_content(content: str) -> str:
        """Clean HTML and extract meaningful text"""
        if not content:
            return ""
        
        # Remove HTML tags if present
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Decode HTML entities
        content = html.unescape(content)
        
        # Remove common web artifacts
        content = re.sub(r'Skip to content|expand_more|open_in_new', '', content)
        content = re.sub(r'#{1,6}\s*', '', content)  # Remove markdown headers
        content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)  # Remove markdown bold
        content = re.sub(r'Blog\s*\*{3,}', '', content)
        content = re.sub(r'Privacy.*?Te\.\.\.', '', content)
        content = re.sub(r'Related information.*?expand_more', '', content)
        
        # Clean up whitespace and special characters
        content = re.sub(r'\s+', ' ', content)
        content = content.replace('â', "'").replace('â€™', "'")
        content = content.strip()
        
        return content
    
    @staticmethod
    def extract_company_size(text: str) -> str:
        """Extract company size information"""
        text_lower = text.lower()
        
        # Look for employee numbers
        size_patterns = [
            r'(\d{1,3}[,\d]*)\s*employees',
            r'employs?\s*(\d{1,3}[,\d]*)',
            r'workforce\s*of\s*(\d{1,3}[,\d]*)',
            r'team\s*of\s*(\d{1,3}[,\d]*)',
            r'(\d{1,3}[,\d]*)\s*people'
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, text_lower)
            if match:
                number = match.group(1)
                return f"Approximately {number} employees"
        
        # Look for size descriptions
        if any(word in text_lower for word in ['large', 'multinational', 'global']):
            return "Large multinational corporation"
        elif any(word in text_lower for word in ['medium', 'mid-size']):
            return "Medium-sized company"
        elif any(word in text_lower for word in ['startup', 'small']):
            return "Small to medium company"
        
        return "No reliable data available"
    
    @staticmethod
    def extract_domain_industry(text: str, company_name: str) -> str:
        """Extract company domain/industry information"""
        text_lower = text.lower()
        company_lower = company_name.lower()
        
        # Technology companies
        tech_keywords = ['technology', 'software', 'cloud', 'internet', 'ai', 'artificial intelligence', 'machine learning', 'data']
        if any(keyword in text_lower for keyword in tech_keywords):
            return "Technology and software services"
        
        # Other industries
        if any(word in text_lower for word in ['finance', 'financial', 'banking']):
            return "Financial services"
        elif any(word in text_lower for word in ['retail', 'e-commerce', 'shopping']):
            return "Retail and e-commerce"
        elif any(word in text_lower for word in ['healthcare', 'medical', 'pharmaceutical']):
            return "Healthcare and medical"
        elif any(word in text_lower for word in ['automotive', 'cars', 'vehicles']):
            return "Automotive industry"
        elif any(word in text_lower for word in ['energy', 'oil', 'renewable']):
            return "Energy sector"
        
        return "Technology and innovation"
    
    @staticmethod
    def extract_headquarters(text: str) -> str:
        """Extract headquarters information"""
        # Common location patterns
        location_patterns = [
            r'headquarters?\s+(?:in\s+|at\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+[A-Z]{2,})',
            r'based\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+[A-Z]{2,})',
            r'located\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+[A-Z]{2,})'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Known company headquarters
        company_hqs = {
            'google': 'Mountain View, California, USA',
            'microsoft': 'Redmond, Washington, USA', 
            'apple': 'Cupertino, California, USA',
            'amazon': 'Seattle, Washington, USA',
            'meta': 'Menlo Park, California, USA',
            'tesla': 'Austin, Texas, USA',
            'netflix': 'Los Gatos, California, USA'
        }
        
        text_lower = text.lower()
        for company, hq in company_hqs.items():
            if company in text_lower:
                return hq
        
        return "No reliable data available"
    
    @staticmethod
    def extract_salary_info(text: str, role: str) -> str:
        """Extract salary information"""
        # Salary patterns
        salary_patterns = [
            r'\$(\d{1,3}[,\d]*)\s*[-–]\s*\$(\d{1,3}[,\d]*)',
            r'\$(\d{1,3}[,\d]*)[kK]?\s*[-–]\s*\$(\d{1,3}[,\d]*)[kK]?',
            r'salary\s*:?\s*\$(\d{1,3}[,\d]*)',
            r'compensation\s*:?\s*\$(\d{1,3}[,\d]*)'
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    return f"${match.group(1)} - ${match.group(2)} USD"
                else:
                    return f"${match.group(1)} USD"
        
        # Role-based estimates
        role_lower = role.lower()
        if 'data scientist' in role_lower:
            return "$120,000 - $200,000 USD (estimated range)"
        elif 'software engineer' in role_lower:
            return "$130,000 - $250,000 USD (estimated range)"
        elif 'product manager' in role_lower:
            return "$140,000 - $220,000 USD (estimated range)"
        
        return "No reliable data available"
    
    @staticmethod
    def extract_skills(text: str, role: str) -> str:
        """Extract required skills"""
        text_lower = text.lower()
        role_lower = role.lower()
        
        # Common skills by role
        if 'data scientist' in role_lower:
            skills = []
            if any(skill in text_lower for skill in ['python', 'r ', 'sql']):
                skills.append('Python, R, SQL')
            if any(skill in text_lower for skill in ['machine learning', 'ml', 'statistics']):
                skills.append('Machine Learning')
            if any(skill in text_lower for skill in ['analytics', 'analysis', 'statistical']):
                skills.append('Statistical Analysis')
            if any(skill in text_lower for skill in ['visualization', 'tableau', 'power bi']):
                skills.append('Data Visualization')
            
            if skills:
                return ', '.join(skills)
            else:
                return "Python, R, SQL, Machine Learning, Statistics"
        
        elif 'software engineer' in role_lower:
            skills = []
            if any(skill in text_lower for skill in ['java', 'python', 'javascript', 'c++', 'go']):
                skills.append('Programming Languages')
            if any(skill in text_lower for skill in ['system design', 'architecture', 'scalability']):
                skills.append('System Design')
            if any(skill in text_lower for skill in ['algorithms', 'data structures']):
                skills.append('Algorithms & Data Structures')
            
            if skills:
                return ', '.join(skills)
            else:
                return "Programming, System Design, Algorithms"
        
        return "No reliable data available"
    
    @staticmethod
    def extract_experience(text: str) -> str:
        """Extract experience requirements"""
        # Experience patterns
        exp_patterns = [
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:experience|work)',
            r'minimum\s+of\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?',
            r'(\d+)[\-–](\d+)\s*years?'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text.lower())
            if match:
                if len(match.groups()) == 2:
                    return f"{match.group(1)}-{match.group(2)} years experience"
                else:
                    return f"{match.group(1)}+ years experience"
        
        return "No reliable data available"

class GeminiAPIClient:
    """Handles Google Gemini API integration for summarization"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.cleaner = ContentCleaner()
    
    def summarize_research(self, company: str, role: str, search_results: List[Dict]) -> Dict[str, str]:
        """Use Gemini to summarize research findings"""
        # Clean search results first
        cleaned_results = []
        for result in search_results[:10]:
            cleaned_content = self.cleaner.clean_html_content(result.get('content', ''))
            if len(cleaned_content) > 50:  # Only include substantial content
                cleaned_results.append({
                    'title': result.get('title', ''),
                    'content': cleaned_content,
                    'url': result.get('url', '')
                })
        
        if not self.api_key or not cleaned_results:
            return self._create_structured_summary(company, role, cleaned_results)
        
        # Prepare context for Gemini
        context = self._prepare_context(cleaned_results)
        
        prompt = f"""
        You are a professional research assistant. Based on the following search results, create a clean summary about {company} and the {role} position.

        Search Results:
        {context}

        Please provide EXACTLY this format (replace [...] with actual information or "No reliable data available"):

        Size: [Specific employee count or company size]
        Domain: [Industry and main business areas]
        Headquarters: [Company headquarters location]
        Latest News: [Recent developments from 2024-2025]

        Skills: [Required technical and soft skills, separated by commas]
        Experience: [Years of experience required]
        Salary Range: [Expected salary range in USD]
        Special Notes: [Benefits, remote work, company culture notes]

        Rules:
        1. Extract only factual information from the search results
        2. Be concise - max 2 sentences per field
        3. If no information is found, write "No reliable data available"
        4. Do not include URLs, HTML, or source references
        5. Focus on the most recent and relevant information
        """
        
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                'contents': [{
                    'parts': [{
                        'text': prompt
                    }]
                }]
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                summary_text = result['candidates'][0]['content']['parts'][0]['text']
                return self._parse_gemini_response(summary_text)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
        
        # Fallback to structured summary
        return self._create_structured_summary(company, role, cleaned_results)
    
    def _prepare_context(self, cleaned_results: List[Dict]) -> str:
        """Prepare cleaned results as context"""
        context = ""
        for i, result in enumerate(cleaned_results[:8], 1):
            context += f"\n{i}. {result.get('title', 'N/A')}\n"
            context += f"   {result.get('content', 'N/A')[:300]}...\n"
        return context
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, str]:
        """Parse Gemini's structured response"""
        try:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            
            company_info = {}
            role_info = {}
            current_section = 'company'
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key in ['size', 'domain', 'headquarters', 'latest news']:
                        company_info[key] = value
                    elif key in ['skills', 'experience', 'salary range', 'special notes']:
                        role_info[key] = value
            
            # Format the results
            company_overview = ""
            company_overview += f"Size: {company_info.get('size', 'No reliable data available')}\n"
            company_overview += f"Domain: {company_info.get('domain', 'No reliable data available')}\n"
            company_overview += f"Headquarters: {company_info.get('headquarters', 'No reliable data available')}\n"
            company_overview += f"Latest News: {company_info.get('latest news', 'No reliable data available')}"
            
            role_requirements = ""
            role_requirements += f"Skills: {role_info.get('skills', 'No reliable data available')}\n"
            role_requirements += f"Experience: {role_info.get('experience', 'No reliable data available')}\n"
            role_requirements += f"Salary Range: {role_info.get('salary range', 'No reliable data available')}\n"
            role_requirements += f"Special Notes: {role_info.get('special notes', 'No reliable data available')}"
            
            return {
                'company_overview': company_overview,
                'role_requirements': role_requirements
            }
            
        except Exception as e:
            print(f"Failed to parse Gemini response: {e}")
            return {
                'company_overview': "No reliable data available",
                'role_requirements': "No reliable data available"
            }
    
    def _create_structured_summary(self, company: str, role: str, cleaned_results: List[Dict]) -> Dict[str, str]:
        """Create structured summary using local processing"""
        all_content = " ".join([result.get('content', '') for result in cleaned_results])
        
        # Extract company information
        size = self.cleaner.extract_company_size(all_content)
        domain = self.cleaner.extract_domain_industry(all_content, company)
        headquarters = self.cleaner.extract_headquarters(all_content)
        
        # Extract role information
        skills = self.cleaner.extract_skills(all_content, role)
        experience = self.cleaner.extract_experience(all_content)
        salary = self.cleaner.extract_salary_info(all_content, role)
        
        # Latest news - look for recent information
        news = "No reliable data available"
        if any(year in all_content for year in ['2024', '2025', 'recent']):
            news_sentences = [s for s in all_content.split('.') if any(year in s for year in ['2024', '2025', 'recent'])]
            if news_sentences:
                news = news_sentences[0][:200] + "..." if len(news_sentences[0]) > 200 else news_sentences[0]
        
        company_overview = f"Size: {size}\nDomain: {domain}\nHeadquarters: {headquarters}\nLatest News: {news}"
        
        role_requirements = f"Skills: {skills}\nExperience: {experience}\nSalary Range: {salary}\nSpecial Notes: No reliable data available"
        
        return {
            'company_overview': company_overview,
            'role_requirements': role_requirements
        }

class ResearchAgent:
    def __init__(self):
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.tavily_url = "https://api.tavily.com/search"
        self.gemini_client = GeminiAPIClient()
    
    def search_tavily(self, query: str, max_results: int = 4) -> List[Dict]:
        """Search using Tavily API"""
        try:
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": [],
                "exclude_domains": []
            }
            
            response = requests.post(self.tavily_url, json=payload, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', [])
            
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    def research_company_and_role(self, company: str, role: str) -> Dict[str, str]:
        """Main research function"""
        print(f"Starting research for {company} - {role}")
        
        # Targeted search queries
        queries = [
            f"{company} company employees size headquarters 2024",
            f"{company} industry business model technology",
            f'"{role}" job requirements {company} skills experience',
            f'"{role}" salary compensation {company} 2024',
            f"{company} recent news developments 2024 2025"
        ]
        
        all_results = []
        
        for query in queries:
            print(f"Searching: {query}")
            results = self.search_tavily(query, max_results=3)
            all_results.extend(results)
            time.sleep(1)  # Rate limiting
        
        print(f"Found {len(all_results)} total results")
        
        # Remove duplicates
        unique_results = []
        seen_urls = set()
        for result in all_results:
            url = result.get('url', '')
            content = result.get('content', '')
            if url and url not in seen_urls and len(content) > 100:
                unique_results.append(result)
                seen_urls.add(url)
        
        print(f"Processing {len(unique_results)} unique results")
        
        # Summarize with Gemini
        if unique_results:
            summary = self.gemini_client.summarize_research(company, role, unique_results)
        else:
            summary = {
                'company_overview': "Size: No reliable data available\nDomain: No reliable data available\nHeadquarters: No reliable data available\nLatest News: No reliable data available",
                'role_requirements': "Skills: No reliable data available\nExperience: No reliable data available\nSalary Range: No reliable data available\nSpecial Notes: No reliable data available"
            }
        
        return summary

# Initialize research agent
research_agent = ResearchAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        print("Search endpoint called")
        data = request.get_json()
        
        if not data or 'company' not in data or 'role' not in data:
            return jsonify({'error': 'Missing company or role in request'}), 400
        
        company = data['company'].strip()
        role = data['role'].strip()
        
        if not company or not role:
            return jsonify({'error': 'Company and role cannot be empty'}), 400
        
        if not research_agent.tavily_api_key:
            return jsonify({'error': 'Tavily API key not configured'}), 500
        
        print(f"Researching: {company} - {role}")
        summary = research_agent.research_company_and_role(company, role)
        
        return jsonify({
            'success': True,
            'company_overview': summary.get('company_overview'),
            'role_requirements': summary.get('role_requirements'),
            'company': company,
            'role': role
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Starting Research Agent Backend...")
    print("API Configuration:")
    print(f"  - Tavily: {'✅' if os.getenv('TAVILY_API_KEY') else '❌'}")
    print(f"  - Gemini: {'✅' if os.getenv('GEMINI_API_KEY') else '❌ (will use local processing)'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)