"""
AI Medical Chatbot Backend
=========================
Handles LangChain pipeline, Gemini API calls, and disease prediction logic.
Uses Google Generative AI (Gemini) for symptom analysis and disease prediction.
Optimized for reliable API usage with proper error handling and model selection.
"""

import os
import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class PredictionResult:
    """Data class to structure prediction results"""
    diseases: List[str]
    confidence_level: str
    needs_body_metrics: bool
    additional_questions: List[str]
    recommendations: List[str]
    disclaimer: str

class MedicalChatbot:
    """
    Main chatbot class that handles medical symptom analysis and disease prediction
    Optimized for reliable Gemini API usage
    """
    
    def __init__(self):
        """Initialize the medical chatbot with optimized Gemini API setup"""
        self.api_key = self._load_api_key()
        self.llm = self._initialize_llm_with_fallback()
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Reduced to save tokens
            return_messages=True,
            memory_key="chat_history"
        )
        self.conversation_chain = self._create_conversation_chain()
        self.user_data = {}
        
    def _load_api_key(self) -> str:
        """Load and validate Gemini API key from environment"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY not found or not configured in .env file. "
                "Please get your API key from https://makersuite.google.com/app/apikey "
                "and update the .env file."
            )
        return api_key
    
    def _initialize_llm_with_fallback(self) -> ChatGoogleGenerativeAI:
        """Initialize Gemini with multiple model fallbacks for reliability"""
        
        # Models in order of preference (most quota-friendly first)
        model_options = [
            {
                "name": "gemini-1.5-flash",
                "description": "Fast, efficient, best for free tier",
                "temperature": 0.2,
                "max_tokens": 512
            },
            {
                "name": "gemini-1.0-pro",
                "description": "Stable, good for medical analysis",
                "temperature": 0.3,
                "max_tokens": 600
            },
            {
                "name": "gemini-pro",
                "description": "Standard model",
                "temperature": 0.3,
                "max_tokens": 600
            }
        ]
        
        for model_config in model_options:
            try:
                logger.info(f"Attempting to initialize {model_config['name']}")
                
                llm = ChatGoogleGenerativeAI(
                    model=model_config["name"],
                    google_api_key=self.api_key,
                    temperature=model_config["temperature"],
                    max_output_tokens=model_config["max_tokens"],
                    request_timeout=60,
                    max_retries=2,
                    # Additional parameters for better reliability
                    top_p=0.8,
                    top_k=40,
                )
                
                # Test the model with a simple query
                test_response = llm.invoke("Say 'Model working'")
                logger.info(f"✅ Successfully initialized {model_config['name']}")
                return llm
                
            except Exception as e:
                logger.warning(f"❌ Failed to initialize {model_config['name']}: {str(e)}")
                continue
        
        raise Exception("❌ Unable to initialize any Gemini model. Please check your API key and try again later.")
    
    def _create_conversation_chain(self) -> ConversationChain:
        """Create optimized LangChain conversation chain for medical analysis"""
        
        # Optimized medical prompt for better API efficiency
        system_prompt = """You are a professional medical AI assistant. Analyze symptoms and provide structured medical insights.

IMPORTANT: Always follow this EXACT response format:

**POSSIBLE CONDITIONS:**
1. [Most likely condition]
2. [Second most likely condition]
3. [Third condition if applicable]

**CONFIDENCE LEVEL:** [High/Medium/Low]

**BODY METRICS NEEDED:** [Yes/No - only if weight/height relevant for diagnosis]

**RECOMMENDATIONS:**
• [Immediate care advice]
• [Self-care recommendations]  
• [When to seek professional help]
• Consult healthcare provider for proper diagnosis

**URGENCY:** [Routine/Urgent/Emergency]

Guidelines:
- Focus on most common conditions matching the symptoms
- Consider symptom duration and severity
- Recommend body metrics only for metabolic, cardiovascular, or weight-related conditions
- Be specific but concise
- Always recommend professional medical consultation

DISCLAIMER: This is informational only, not medical advice."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        return ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=False
        )
    
    def analyze_symptoms(self, symptoms: str, user_context: Dict = None) -> PredictionResult:
        """
        Main method to analyze symptoms with enhanced error handling
        """
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Prepare concise input to minimize token usage
                input_text = self._prepare_optimized_input(symptoms, user_context)
                
                # Make API call with timeout
                logger.info(f"Analyzing symptoms (attempt {attempt + 1}/{max_retries})")
                response = self.conversation_chain.predict(input=input_text)
                
                # Parse and structure the response
                result = self._parse_structured_response(response)
                
                logger.info("✅ Successfully analyzed symptoms")
                return result
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"❌ Attempt {attempt + 1} failed: {error_str}")
                
                # Handle specific error types
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (base_delay ** (attempt + 1)) + (attempt * 10)
                        logger.info(f"⏳ Rate limit hit. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return self._create_quota_error_result(symptoms)
                
                elif "timeout" in error_str.lower():
                    if attempt < max_retries - 1:
                        logger.info(f"⏳ Timeout occurred. Retrying in {base_delay * (attempt + 1)} seconds...")
                        time.sleep(base_delay * (attempt + 1))
                        continue
                
                # For final attempt or non-recoverable errors
                if attempt == max_retries - 1:
                    return self._create_error_result(error_str)
                
        return self._create_error_result("Maximum retries exceeded")
    
    def _prepare_optimized_input(self, symptoms: str, user_context: Dict = None) -> str:
        """Prepare optimized input to minimize token usage while maintaining accuracy"""
        
        # Clean and optimize symptom description
        symptoms_clean = symptoms.strip()
        if len(symptoms_clean) > 300:
            symptoms_clean = symptoms_clean[:300] + "..."
        
        input_parts = [f"Symptoms: {symptoms_clean}"]
        
        # Add relevant context only
        if user_context:
            for key, value in user_context.items():
                if value and key in ['age', 'gender', 'duration', 'severity']:
                    input_parts.append(f"{key.title()}: {value}")
        
        return " | ".join(input_parts)
    
    def _parse_structured_response(self, response: str) -> PredictionResult:
        """Parse structured response from Gemini API"""
        
        try:
            diseases = []
            confidence_level = "Medium"
            needs_body_metrics = False
            recommendations = []
            urgency = "Routine"
            
            lines = response.strip().split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Identify sections
                if "**POSSIBLE CONDITIONS:**" in line or "**CONDITIONS:**" in line:
                    current_section = "conditions"
                    continue
                elif "**CONFIDENCE LEVEL:**" in line or "**CONFIDENCE:**" in line:
                    current_section = "confidence"
                    # Extract confidence from the same line
                    if "High" in line:
                        confidence_level = "High"
                    elif "Low" in line:
                        confidence_level = "Low"
                    else:
                        confidence_level = "Medium"
                    continue
                elif "**BODY METRICS NEEDED:**" in line or "**BODY METRICS:**" in line:
                    current_section = "metrics"
                    needs_body_metrics = "Yes" in line or "yes" in line.lower()
                    continue
                elif "**RECOMMENDATIONS:**" in line:
                    current_section = "recommendations"
                    continue
                elif "**URGENCY:**" in line:
                    current_section = "urgency"
                    if "Emergency" in line:
                        urgency = "Emergency"
                    elif "Urgent" in line:
                        urgency = "Urgent"
                    continue
                
                # Extract content based on current section
                if current_section == "conditions":
                    # Extract numbered conditions
                    if re.match(r'^\d+\.', line):
                        condition = re.sub(r'^\d+\.\s*', '', line).strip()
                        if condition:
                            diseases.append(condition)
                
                elif current_section == "recommendations":
                    # Extract bullet point recommendations
                    if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                        rec = re.sub(r'^[•\-\*]\s*', '', line).strip()
                        if rec:
                            recommendations.append(rec)
            
            # Fallback parsing if structured parsing fails
            if not diseases:
                diseases = self._extract_conditions_fallback(response)
            
            if not recommendations:
                recommendations = self._extract_recommendations_fallback(response)
            
            # Ensure we have at least some results
            if not diseases:
                diseases = ["Further evaluation needed"]
            
            if not recommendations:
                recommendations = ["Consult healthcare provider for proper diagnosis"]
            
            return PredictionResult(
                diseases=diseases[:4],  # Limit to top 4
                confidence_level=confidence_level,
                needs_body_metrics=needs_body_metrics,
                additional_questions=[],
                recommendations=recommendations[:6],  # Limit recommendations
                disclaimer="⚠️ This is AI-generated medical information for educational purposes only. Always consult qualified healthcare professionals for proper diagnosis and treatment."
            )
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._create_parsing_error_result(response)
    
    def _extract_conditions_fallback(self, response: str) -> List[str]:
        """Fallback method to extract medical conditions from response"""
        conditions = []
        
        # Common medical condition patterns
        medical_keywords = [
            'infection', 'influenza', 'flu', 'covid', 'pneumonia', 'bronchitis',
            'migraine', 'headache', 'sinusitis', 'gastroenteritis', 'allergies',
            'hypertension', 'diabetes', 'asthma', 'arthritis', 'dermatitis'
        ]
        
        response_lower = response.lower()
        
        for keyword in medical_keywords:
            if keyword in response_lower:
                # Try to extract the full condition name around the keyword
                sentences = response.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        # Clean up the sentence and extract potential condition
                        clean_sentence = sentence.strip()
                        if len(clean_sentence.split()) <= 8:  # Reasonable condition name length
                            conditions.append(clean_sentence.title())
                        break
        
        return list(set(conditions))[:4]  # Remove duplicates and limit
    
    def _extract_recommendations_fallback(self, response: str) -> List[str]:
        """Fallback method to extract recommendations from response"""
        recommendations = []
        
        # Look for recommendation keywords
        rec_keywords = ['recommend', 'should', 'consider', 'try', 'take', 'seek', 'consult']
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in rec_keywords):
                if 10 <= len(sentence) <= 100:  # Reasonable recommendation length
                    recommendations.append(sentence)
        
        # Add default recommendations if none found
        if not recommendations:
            recommendations = [
                "Monitor symptoms and track any changes",
                "Stay hydrated and get adequate rest",
                "Consult healthcare provider if symptoms persist or worsen",
                "Seek immediate care if symptoms become severe"
            ]
        
        return recommendations[:6]
    
    def _create_quota_error_result(self, symptoms: str) -> PredictionResult:
        """Create result for quota exceeded errors with basic symptom analysis"""
        
        symptoms_lower = symptoms.lower()
        basic_conditions = []
        basic_recommendations = []
        
        # Basic pattern matching for common symptom combinations
        if "headache" in symptoms_lower and "fever" in symptoms_lower:
            basic_conditions = [
                "Viral Upper Respiratory Infection",
                "Influenza (Flu)",
                "COVID-19",
                "Sinus Infection"
            ]
            basic_recommendations = [
                "Monitor fever - seek care if >103°F (39.4°C)",
                "Use acetaminophen/ibuprofen for symptom relief",
                "Stay hydrated with plenty of fluids",
                "Rest and consider COVID-19 testing",
                "Consult healthcare provider if symptoms worsen"
            ]
        
        elif "cough" in symptoms_lower and "fever" in symptoms_lower:
            basic_conditions = [
                "Viral Respiratory Infection",
                "COVID-19",
                "Bacterial Infection",
                "Bronchitis"
            ]
            basic_recommendations = [
                "Monitor breathing and temperature",
                "Stay hydrated and rest",
                "Consider COVID-19 testing",
                "Seek care for difficulty breathing",
                "Consult healthcare provider"
            ]
        
        else:
            basic_conditions = ["Requires medical evaluation"]
            basic_recommendations = [
                "Document all symptoms with timeline",
                "Monitor symptom progression",
                "Consult healthcare provider for proper evaluation",
                "Seek immediate care for severe symptoms"
            ]
        
        return PredictionResult(
            diseases=basic_conditions,
            confidence_level="Low",
            needs_body_metrics=False,
            additional_questions=[],
            recommendations=basic_recommendations,
            disclaimer="⚠️ API quota temporarily exceeded. This is basic analysis - please consult healthcare provider for proper medical evaluation."
        )
    
    def _create_error_result(self, error_message: str) -> PredictionResult:
        """Create error result when analysis fails"""
        return PredictionResult(
            diseases=["Unable to analyze symptoms"],
            confidence_level="Low", 
            needs_body_metrics=False,
            additional_questions=[],
            recommendations=[
                f"Technical error: {error_message[:100]}",
                "Please try again in a few moments",
                "For immediate concerns, contact healthcare provider",
                "For emergencies, call emergency services"
            ],
            disclaimer="⚠️ Technical error occurred. Please consult healthcare provider for medical advice."
        )
    
    def _create_parsing_error_result(self, response: str) -> PredictionResult:
        """Create result when response parsing fails but we have a response"""
        return PredictionResult(
            diseases=["Response parsing error"],
            confidence_level="Low",
            needs_body_metrics=False,
            additional_questions=[],
            recommendations=[
                "AI response could not be properly formatted",
                "Raw response received but not structured",
                "Please try rephrasing your symptoms",
                "Consult healthcare provider for reliable analysis"
            ],
            disclaimer="⚠️ AI response formatting error. Please consult healthcare provider for medical advice."
        )
    
    def add_body_metrics(self, weight: float, height: float, unit_system: str = "metric") -> str:
        """Add body metrics and get BMI analysis"""
        try:
            # Convert to metric if needed
            if unit_system.lower() == "imperial":
                weight_kg = weight * 0.453592
                height_m = height * 0.0254
            else:
                weight_kg = weight
                height_m = height / 100
            
            # Calculate BMI
            bmi = weight_kg / (height_m ** 2)
            
            # Store in user data
            self.user_data.update({
                'weight': weight_kg,
                'height': height_m * 100,
                'bmi': round(bmi, 1),
                'unit_system': unit_system
            })
            
            # Create BMI analysis prompt
            bmi_prompt = f"""Analyze this BMI data in context of previous symptoms:

BMI: {bmi:.1f}
Weight: {weight_kg:.1f} kg  
Height: {height_m*100:.1f} cm

Provide:
1. BMI category assessment
2. How this relates to the reported symptoms
3. Health recommendations based on BMI
4. Any weight-related risk factors"""
            
            # Get BMI analysis
            response = self.conversation_chain.predict(input=bmi_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error processing body metrics: {e}")
            return f"Unable to process body metrics: {str(e)}. Please consult healthcare provider."
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        history = []
        if hasattr(self.memory, 'chat_memory'):
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "assistant", "content": message.content})
        return history
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.memory.clear()
        self.user_data.clear()
        logger.info("Conversation cleared")

# Factory function
def create_chatbot_instance() -> MedicalChatbot:
    """Create chatbot instance with proper error handling"""
    try:
        return MedicalChatbot()
    except Exception as e:
        logger.error(f"Failed to create chatbot: {e}")
        raise

# Validation functions
def validate_symptoms_input(symptoms: str) -> Tuple[bool, str]:
    """Validate symptoms input"""
    if not symptoms or not symptoms.strip():
        return False, "Please describe your symptoms."
    
    if len(symptoms.strip()) < 5:
        return False, "Please provide more detailed description."
    
    # Check for emergency keywords
    emergency_words = ['suicide', 'kill myself', 'end my life']
    if any(word in symptoms.lower() for word in emergency_words):
        return False, "For mental health emergencies, please contact crisis services immediately: 988 (US), 116 123 (UK), or local emergency services."
    
    return True, "Valid input"

def validate_body_metrics(weight: str, height: str, unit_system: str = "metric") -> Tuple[bool, str, float, float]:
    """Validate body metrics"""
    try:
        weight_val = float(weight)
        height_val = float(height)
        
        if unit_system.lower() == "metric":
            if not (20 <= weight_val <= 300):
                return False, "Weight should be between 20-300 kg", 0, 0
            if not (100 <= height_val <= 250):
                return False, "Height should be between 100-250 cm", 0, 0
        else:
            if not (44 <= weight_val <= 660):
                return False, "Weight should be between 44-660 lbs", 0, 0
            if not (39 <= height_val <= 98):
                return False, "Height should be between 39-98 inches", 0, 0
        
        return True, "Valid metrics", weight_val, height_val
        
    except ValueError:
        return False, "Please enter valid numbers", 0, 0
