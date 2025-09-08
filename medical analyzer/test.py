#!/usr/bin/env python3
"""
AI Medical Chatbot - Quick Launcher
===================================
Simple script to launch the medical chatbot application with proper error handling.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'gradio',
        'langchain',
        'langchain_google_genai',
        'python-dotenv',
        'google-generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"] + missing_packages
            )
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            sys.exit(1)

def check_env_file():
    """Check if .env file exists and has API key"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env file not found!")
        print("📝 Creating .env template...")
        
        env_template = """# AI Medical Chatbot Environment Variables
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Optional Configuration
LOG_LEVEL=INFO
"""
        with open(".env", "w") as f:
            f.write(env_template)
        
        print("✅ .env file created. Please add your Gemini API key and restart.")
        print("🔗 Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Check if API key is set
    with open(".env", "r") as f:
        content = f.read()
        if "your_gemini_api_key_here" in content:
            print("❌ Please replace 'your_gemini_api_key_here' with your actual Gemini API key in .env file")
            print("🔗 Get your API key from: https://makersuite.google.com/app/apikey")
            sys.exit(1)
        elif "GEMINI_API_KEY=" in content:
            print("✅ .env file found with API key")
        else:
            print("❌ GEMINI_API_KEY not found in .env file")
            sys.exit(1)

def check_files():
    """Check if all required files exist"""
    required_files = ["backend.py", "frontend.py"]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file missing: {file}")
            sys.exit(1)
        else:
            print(f"✅ {file} found")

def launch_application():
    """Launch the medical chatbot application"""
    print("\n🚀 Launching AI Medical Chatbot...")
    print("📱 The application will open in your default web browser")
    print("🌐 URL: http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        # Import and run the application
        from frontend import main
        main()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"Error launching application: {e}")
        print(f"\n❌ Failed to launch application: {e}")
        print("📋 Check the error message above and ensure all requirements are met")

def main():
    """Main launcher function"""
    print("🏥 AI Medical Symptom Analyzer - Launcher")
    print("=" * 50)
    
    # Run all checks
    print("\n🔍 Running system checks...")
    check_python_version()
    check_files()
    check_dependencies()
    check_env_file()
    
    print("\n✅ All checks passed!")
    print("-" * 30)
    
    # Launch application
    launch_application()

if __name__ == "__main__":
    main()
