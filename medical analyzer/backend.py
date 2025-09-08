"""
AI Medical Chatbot Backend
=========================
Handles LangChain pipeline, Gemini API calls, and disease prediction logic.
Uses Google Generative AI (Gemini) for symptom analysis and disease prediction.
"""

import os
import logging
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
    """
    
    def __init__(self):
        """Initialize the medical chatbot with Gemini API and LangChain setup"""
        self.api_key = self._load_api_key()
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        self.conversation_chain = self._create_conversation_chain()
        self.user_data = {}  # Store user information during conversation
        
    def _load_api_key(self) -> str:
        """Load and validate Gemini API key from environment"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )
        return api_key
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize Google Generative AI (Gemini) model"""
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # Using the best available Gemini model
                google_api_key=self.api_key,
                temperature=0.3,  # Lower temperature for more consistent medical responses
                max_output_tokens=1024
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            raise
    
    def _create_conversation_chain(self) -> ConversationChain:
        """Create LangChain conversation chain with medical-specific prompt"""
        
        # Define the system prompt for medical symptom analysis
        system_prompt = """You are an AI medical assistant designed to help analyze symptoms and suggest possible conditions. 
        
        IMPORTANT DISCLAIMERS:
        - You are NOT a replacement for professional medical advice
        - Always recommend consulting with a healthcare provider
        - Your suggestions are for informational purposes only
        
        Your capabilities:
        1. Analyze user-described symptoms
        2. Suggest possible conditions based on symptoms
        3. Ask for additional information when needed (age, weight, height, duration, etc.)
        4. Provide general health recommendations
        5. Identify when body metrics (weight/height) are relevant for assessment
        
        CONDITIONS THAT MAY REQUIRE BODY METRICS:
        - Obesity-related conditions
        - Malnutrition or eating disorders
        - Diabetes and metabolic disorders
        - Cardiovascular issues
        - Growth and development concerns
        - Joint and mobility problems
        
        Response Format:
        - List 2-4 most likely conditions
        - Provide confidence level (High/Medium/Low)
        - Ask for body metrics if relevant
        - Suggest immediate actions if urgent
        - Always include medical disclaimer
        
        Be empathetic, professional, and thorough in your responses."""
        
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
        Main method to analyze symptoms and predict possible diseases
        
        Args:
            symptoms (str): User-described symptoms
            user_context (Dict): Additional user information (age, weight, height, etc.)
        
        Returns:
            PredictionResult: Structured prediction results
        """
        try:
            # Prepare input with context if available
            input_text = self._prepare_input(symptoms, user_context)
            
            # Get response from LangChain conversation chain
            response = self.conversation_chain.predict(input=input_text)
            
            # Parse and structure the response
            result = self._parse_response(response)
            
            logger.info(f"Successfully analyzed symptoms: {symptoms[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing symptoms: {e}")
            return self._create_error_result(str(e))
    
    def _prepare_input(self, symptoms: str, user_context: Dict = None) -> str:
        """Prepare input text with symptoms and user context"""
        input_parts = [f"Symptoms: {symptoms}"]
        
        if user_context:
            for key, value in user_context.items():
                if value:
                    input_parts.append(f"{key.title()}: {value}")
        
        return "\n".join(input_parts)
    
    def _parse_response(self, response: str) -> PredictionResult:
        """Parse LLM response and structure it into PredictionResult"""
        
        # Simple parsing logic - in production, you might want more sophisticated parsing
        lines = response.strip().split('\n')
        
        diseases = []
        confidence_level = "Medium"
        needs_body_metrics = False
        additional_questions = []
        recommendations = []
        
        # Look for keywords in response to determine structure
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if body metrics are mentioned
            body_metric_keywords = ['weight', 'height', 'bmi', 'body mass', 'obesity', 'underweight']
            if any(keyword in line_lower for keyword in body_metric_keywords):
                needs_body_metrics = True
            
            # Extract diseases/conditions (simple heuristic)
            condition_keywords = ['condition', 'disease', 'disorder', 'syndrome', 'possible', 'likely']
            if any(keyword in line_lower for keyword in condition_keywords) and ':' in line:
                potential_disease = line.split(':')[-1].strip()
                if potential_disease and len(potential_disease.split()) <= 5:
                    diseases.append(potential_disease)
            
            # Extract confidence level
            if 'confidence' in line_lower:
                if 'high' in line_lower:
                    confidence_level = "High"
                elif 'low' in line_lower:
                    confidence_level = "Low"
            
            # Extract recommendations
            if line_lower.startswith('recommend') or 'should' in line_lower:
                recommendations.append(line.strip())
        
        # Fallback: if no diseases found, extract from full response
        if not diseases:
            # Simple extraction of potential medical terms
            medical_terms = ['diabetes', 'hypertension', 'infection', 'inflammation', 
                           'allergy', 'migraine', 'anxiety', 'depression', 'arthritis']
            for term in medical_terms:
                if term in response.lower():
                    diseases.append(term.title())
        
        # Ensure we have at least some diseases
        if not diseases:
            diseases = ["Condition requires further evaluation"]
        
        return PredictionResult(
            diseases=diseases[:4],  # Limit to 4 conditions
            confidence_level=confidence_level,
            needs_body_metrics=needs_body_metrics,
            additional_questions=additional_questions,
            recommendations=recommendations if recommendations else ["Consult with a healthcare provider"],
            disclaimer="⚠️ This is not professional medical advice. Please consult a qualified healthcare provider for proper diagnosis and treatment."
        )
    
    def _create_error_result(self, error_message: str) -> PredictionResult:
        """Create error result when analysis fails"""
        return PredictionResult(
            diseases=["Unable to analyze symptoms"],
            confidence_level="Low",
            needs_body_metrics=False,
            additional_questions=[],
            recommendations=[f"Error occurred: {error_message}", "Please try again or consult a healthcare provider"],
            disclaimer="⚠️ Technical error occurred. Please consult a healthcare provider for medical advice."
        )
    
    def add_body_metrics(self, weight: float, height: float, unit_system: str = "metric") -> str:
        """
        Add body metrics to user context and provide BMI analysis
        
        Args:
            weight (float): User weight
            height (float): User height
            unit_system (str): "metric" (kg, cm) or "imperial" (lbs, inches)
        
        Returns:
            str: Analysis including BMI and health implications
        """
        try:
            # Convert to metric if needed
            if unit_system.lower() == "imperial":
                weight_kg = weight * 0.453592  # lbs to kg
                height_m = height * 0.0254  # inches to meters
            else:
                weight_kg = weight
                height_m = height / 100  # cm to meters
            
            # Calculate BMI
            bmi = weight_kg / (height_m ** 2)
            
            # Store in user data
            self.user_data.update({
                'weight': weight_kg,
                'height': height_m * 100,  # Store as cm
                'bmi': round(bmi, 1),
                'unit_system': unit_system
            })
            
            # Get BMI analysis from LLM
            bmi_prompt = f"""
            Analyze this BMI information:
            - BMI: {bmi:.1f}
            - Weight: {weight_kg:.1f} kg
            - Height: {height_m*100:.1f} cm
            
            Provide:
            1. BMI category (underweight, normal, overweight, obese)
            2. Health implications related to the previous symptoms discussed
            3. Recommendations based on BMI and symptoms
            """
            
            response = self.conversation_chain.predict(input=bmi_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error processing body metrics: {e}")
            return f"Error processing body metrics: {e}"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        history = []
        if hasattr(self.memory, 'chat_memory'):
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "assistant", "content": message.content})
        return history
    
    def clear_conversation(self):
        """Clear conversation history and user data"""
        self.memory.clear()
        self.user_data.clear()
        logger.info("Conversation history cleared")

# Utility functions for the frontend
def create_chatbot_instance() -> MedicalChatbot:
    """Factory function to create chatbot instance"""
    try:
        return MedicalChatbot()
    except Exception as e:
        logger.error(f"Failed to create chatbot instance: {e}")
        raise

def validate_symptoms_input(symptoms: str) -> Tuple[bool, str]:
    """Validate user symptoms input"""
    if not symptoms or not symptoms.strip():
        return False, "Please describe your symptoms."
    
    if len(symptoms.strip()) < 10:
        return False, "Please provide more detailed description of your symptoms."
    
    # Check for inappropriate content (basic filter)
    inappropriate_words = ['suicide', 'kill', 'die', 'death']
    if any(word in symptoms.lower() for word in inappropriate_words):
        return False, "For urgent mental health concerns, please contact emergency services or a crisis hotline immediately."
    
    return True, "Valid input"

def validate_body_metrics(weight: str, height: str, unit_system: str = "metric") -> Tuple[bool, str, float, float]:
    """Validate body metrics input"""
    try:
        weight_val = float(weight)
        height_val = float(height)
        
        # Validate ranges based on unit system
        if unit_system.lower() == "metric":
            if not (20 <= weight_val <= 300):  # kg
                return False, "Weight should be between 20-300 kg", 0, 0
            if not (100 <= height_val <= 250):  # cm
                return False, "Height should be between 100-250 cm", 0, 0
        else:  # imperial
            if not (44 <= weight_val <= 660):  # lbs
                return False, "Weight should be between 44-660 lbs", 0, 0
            if not (39 <= height_val <= 98):  # inches
                return False, "Height should be between 39-98 inches", 0, 0
        
        return True, "Valid metrics", weight_val, height_val
        
    except ValueError:
        return False, "Please enter valid numbers for weight and height", 0, 0