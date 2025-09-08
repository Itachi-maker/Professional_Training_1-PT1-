"""
AI Medical Chatbot Frontend
===========================
Dark-themed Gradio interface that interacts with the backend.
Provides an intuitive chat interface for symptom analysis.
"""

import gradio as gr
import logging
from typing import List, Tuple, Dict, Any
import traceback

# Import backend functionality
from backend import (
    create_chatbot_instance, 
    validate_symptoms_input, 
    validate_body_metrics,
    PredictionResult
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatbotUI:
    """Main UI class for the medical chatbot interface"""
    
    def __init__(self):
        """Initialize the UI with chatbot backend"""
        try:
            self.chatbot = create_chatbot_instance()
            self.conversation_history = []
            logger.info("Medical Chatbot UI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chatbot UI: {e}")
            self.chatbot = None
    
    def process_symptoms(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
        """
        Process user symptoms and return chatbot response
        
        Args:
            message (str): User's symptom description
            history (List[List[str]]): Chat history
        
        Returns:
            Tuple[List[List[str]], str]: Updated history and empty string for input
        """
        if not self.chatbot:
            error_msg = "❌ Chatbot not initialized. Please check your API key configuration."
            history.append([message, error_msg])
            return history, ""
        
        try:
            # Validate input
            is_valid, validation_msg = validate_symptoms_input(message)
            if not is_valid:
                history.append([message, f"❌ {validation_msg}"])
                return history, ""
            
            # Analyze symptoms
            result = self.chatbot.analyze_symptoms(message, self.chatbot.user_data)
            
            # Format response
            response = self._format_prediction_response(result)
            
            # Add to history
            history.append([message, response])
            
            # Store in conversation history
            self.conversation_history = history
            
            return history, ""
            
        except Exception as e:
            logger.error(f"Error processing symptoms: {e}")
            error_response = f"❌ Sorry, an error occurred while analyzing your symptoms: {str(e)}\n\nPlease try again or consult a healthcare provider."
            history.append([message, error_response])
            return history, ""
    
    def _format_prediction_response(self, result: PredictionResult) -> str:
        """Format the prediction result into a readable response"""
        
        response_parts = []
        
        # Header
        response_parts.append("🔍 **Symptom Analysis Results**\n")
        
        # Possible conditions
        if result.diseases:
            response_parts.append("**Possible Conditions:**")
            for i, disease in enumerate(result.diseases, 1):
                response_parts.append(f"  {i}. {disease}")
            response_parts.append("")
        
        # Confidence level
        confidence_emoji = {
            "High": "🟢",
            "Medium": "🟡", 
            "Low": "🔴"
        }
        emoji = confidence_emoji.get(result.confidence_level, "🟡")
        response_parts.append(f"**Confidence Level:** {emoji} {result.confidence_level}\n")
        
        # Body metrics request
        if result.needs_body_metrics:
            response_parts.append("📏 **Additional Information Needed:**")
            response_parts.append("Your symptoms may be related to body weight and height factors.")
            response_parts.append("Please provide your weight and height below for a more accurate analysis.\n")
        
        # Recommendations
        if result.recommendations:
            response_parts.append("💡 **Recommendations:**")
            for rec in result.recommendations:
                if rec.strip():
                    response_parts.append(f"  • {rec}")
            response_parts.append("")
        
        # Disclaimer
        response_parts.append(f"⚠️ **Important:** {result.disclaimer}")
        
        return "\n".join(response_parts)
    
    def process_body_metrics(self, weight: str, height: str, unit_system: str, history: List[List[str]]) -> Tuple[List[List[str]], str, str]:
        """
        Process body metrics and provide BMI analysis
        
        Args:
            weight (str): Weight input
            height (str): Height input  
            unit_system (str): "Metric" or "Imperial"
            history (List[List[str]]): Chat history
        
        Returns:
            Tuple: Updated history, empty weight input, empty height input
        """
        if not self.chatbot:
            error_msg = "❌ Chatbot not initialized. Please check your API key configuration."
            history.append(["Body metrics submitted", error_msg])
            return history, "", ""
        
        try:
            # Validate inputs
            is_valid, validation_msg, weight_val, height_val = validate_body_metrics(
                weight, height, unit_system.lower()
            )
            
            if not is_valid:
                history.append(["Body metrics submitted", f"❌ {validation_msg}"])
                return history, weight, height
            
            # Process metrics
            response = self.chatbot.add_body_metrics(
                weight_val, height_val, unit_system.lower()
            )
            
            # Format the response
            user_message = f"Weight: {weight} {'kg' if unit_system.lower() == 'metric' else 'lbs'}, Height: {height} {'cm' if unit_system.lower() == 'metric' else 'inches'}"
            formatted_response = f"📊 **BMI Analysis:**\n\n{response}"
            
            history.append([user_message, formatted_response])
            
            return history, "", ""
            
        except Exception as e:
            logger.error(f"Error processing body metrics: {e}")
            error_response = f"❌ Error processing body metrics: {str(e)}"
            history.append(["Body metrics submitted", error_response])
            return history, weight, height
    
    def clear_conversation(self, history: List[List[str]]) -> List[List[str]]:
        """Clear the conversation history"""
        if self.chatbot:
            self.chatbot.clear_conversation()
        self.conversation_history = []
        return []
    
    def get_example_symptoms(self) -> List[List[str]]:
        """Get example symptoms for user guidance"""
        examples = [
            ["I have been experiencing frequent headaches, dizziness, and fatigue for the past week. The headaches are usually in the morning and I feel nauseous sometimes."],
            ["I've been having chest pain and shortness of breath, especially when climbing stairs. I also feel my heart racing sometimes."],
            ["I have a persistent cough with yellow mucus, fever of 101°F, and body aches for 3 days. I'm also feeling very tired."],
            ["I've been experiencing abdominal pain, bloating, and irregular bowel movements for the past month. Sometimes I feel nauseous after eating."]
        ]
        return examples

def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface"""
    
    # Initialize UI
    ui = MedicalChatbotUI()
    
    # Custom CSS for dark theme and styling
    custom_css = """
    /* Dark theme customizations */
    .gradio-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #ffffff;
    }
    
    .gr-box {
        background: rgba(45, 45, 45, 0.8);
        border: 1px solid #404040;
        border-radius: 12px;
    }
    
    .gr-button {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .gr-button:hover {
        background: linear-gradient(135deg, #357abd 0%, #2968a3 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    }
    
    .gr-textbox {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #404040;
        color: #ffffff;
        border-radius: 8px;
    }
    
    .gr-chatbot {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid #404040;
        border-radius: 12px;
    }
    
    .message {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 12px;
        margin: 8px;
    }
    
    /* Header styling */
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #4a90e2, #50c878);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #4a90e2;
        border-bottom: 2px solid #4a90e2;
        padding-bottom: 8px;
    }
    
    /* Warning and info boxes */
    .warning {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: #ffc107;
    }
    
    .info {
        background: rgba(74, 144, 226, 0.1);
        border: 1px solid #4a90e2;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: #4a90e2;
    }
    """
    
    # Create the interface
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
        ).set(
            body_background_fill="*neutral_950",
            body_text_color="*neutral_50",
            background_fill_primary="*neutral_900",
            background_fill_secondary="*neutral_800",
            border_color_primary="*neutral_600",
        ),
        css=custom_css,
        title="AI Medical Chatbot"
    ) as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🏥 AI Medical Symptom Analyzer</h1>
            <p style="font-size: 1.2rem; color: #b0b0b0; margin-top: 10px;">
                Describe your symptoms and get AI-powered health insights
            </p>
        </div>
        """)
        
        # Warning notice
        gr.HTML("""
        <div class="warning">
            <strong>⚠️ Medical Disclaimer:</strong> This AI tool is for informational purposes only 
            and is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult with qualified healthcare providers for medical concerns.
        </div>
        """)
        
        with gr.Row():
            # Main chat interface
            with gr.Column(scale=2):
                gr.HTML("<h3>💬 Symptom Analysis Chat</h3>")
                
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="Your conversation will appear here...",
                    show_label=False,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Describe your symptoms in detail (e.g., 'I have headaches, fever, and fatigue for 3 days')",
                        lines=3,
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            # Body metrics panel
            with gr.Column(scale=1):
                gr.HTML("<h3>📏 Body Metrics (Optional)</h3>")
                gr.HTML("""
                <div class="info">
                    <strong>ℹ️ When needed:</strong> The AI will request your weight and height 
                    if your symptoms might be related to BMI or metabolic factors.
                </div>
                """)
                
                unit_system = gr.Radio(
                    choices=["Metric", "Imperial"],
                    value="Metric",
                    label="Unit System"
                )
                
                with gr.Row():
                    weight_input = gr.Textbox(
                        label="Weight",
                        placeholder="e.g., 70 (kg) or 154 (lbs)"
                    )
                    height_input = gr.Textbox(
                        label="Height", 
                        placeholder="e.g., 175 (cm) or 69 (inches)"
                    )
                
                submit_metrics_btn = gr.Button("Submit Metrics", variant="primary")
                
                # Help section
                gr.HTML("<h3>💡 How to Use</h3>")
                gr.HTML("""
                <div style="background: rgba(255, 255, 255, 0.05); padding: 16px; border-radius: 8px;">
                    <ol>
                        <li><strong>Describe symptoms:</strong> Be specific about what you're experiencing, when it started, and how severe it is.</li>
                        <li><strong>Review analysis:</strong> The AI will suggest possible conditions and recommendations.</li>
                        <li><strong>Add metrics if needed:</strong> If requested, provide your weight and height for BMI analysis.</li>
                        <li><strong>Consult a doctor:</strong> Always follow up with professional medical care.</li>
                    </ol>
                </div>
                """)
        
                # Example symptoms section
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3>📝 Example Symptoms</h3>")
                        example_symptoms = [
                            ["I have been experiencing frequent headaches, dizziness, and fatigue for the past week. The headaches are usually in the morning and I feel nauseous sometimes."],
                            ["I've been having chest pain and shortness of breath, especially when climbing stairs. I also feel my heart racing sometimes."],
                            ["I have a persistent cough with yellow mucus, fever of 101°F, and body aches for 3 days. I'm also feeling very tired."],
                            ["I've been experiencing abdominal pain, bloating, and irregular bowel movements for the past month. Sometimes I feel nauseous after eating."]
                        ]
                        examples = gr.Examples(
                            examples=example_symptoms,
                            inputs=[msg_input],
                            label="Click on an example to try:"
                        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #404040;">
            <p style="color: #888;">
                🤖 Powered by Google Gemini AI & LangChain | 
                Built with ❤️ for educational purposes
            </p>
        </div>
        """)
        
        # Event handlers
        def handle_submit(message, history):
            return ui.process_symptoms(message, history)
        
        def handle_metrics_submit(weight, height, unit, history):
            return ui.process_body_metrics(weight, height, unit, history)
        
        def handle_clear(history):
            return ui.clear_conversation(history)
        
        # Connect events
        msg_input.submit(
            fn=handle_submit,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        submit_btn.click(
            fn=handle_submit,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        submit_metrics_btn.click(
            fn=handle_metrics_submit,
            inputs=[weight_input, height_input, unit_system, chatbot],
            outputs=[chatbot, weight_input, height_input]
        )
        
        clear_btn.click(
            fn=handle_clear,
            inputs=[chatbot],
            outputs=[chatbot]
        )
    
    return interface

def main():
    """Main function to launch the application"""
    try:
        # Create and launch interface
        interface = create_interface()
        
        # Launch with custom settings
        interface.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,       # Default Gradio port
            share=False,            # Set to True to create public link
            debug=False,            # Set to True for debugging
            show_error=True,        # Show errors in interface
            inbrowser=True          # Auto-open in browser
        )
        
    except Exception as e:
        logger.error(f"Failed to launch interface: {e}")
        print(f"❌ Error launching application: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
