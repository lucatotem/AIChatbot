#!/usr/bin/env python3
"""
Setup script for AI Chatbot Backend
This script helps you set up the environment and test the installation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📋 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_environment():
    """Set up the Python environment."""
    print("🚀 Setting up AI Chatbot Backend Environment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("📝 Creating .env file...")
        shutil.copy(".env.example", ".env")
        print("✅ .env file created from template")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("💡 Try using: python -m pip install -r requirements.txt")
        return False
    
    # Create necessary directories
    os.makedirs("documents", exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)
    print("✅ Created necessary directories")
    
    return True

def test_installation():
    """Test if the installation works."""
    print("\n🧪 Testing Installation")
    print("=" * 30)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        import fastapi
        import uvicorn
        import transformers
        import langchain
        import sentence_transformers
        print("✅ All required packages imported successfully")
        
        # Test model loading (just check if we can import)
        print("🤖 Testing Hugging Face transformers...")
        from transformers import pipeline
        print("✅ Transformers library working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main setup function."""
    print("🎯 AI Chatbot Backend Setup")
    print("This script will help you set up your RAG-powered chatbot backend.")
    print()
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed. Please check the errors above.")
        return 1
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed. Please check the errors above.")
        return 1
    
    # Success message
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your resume and documents to the 'documents/' folder")
    print("2. Update the sample information in documents/sample_resume.txt")
    print("3. Run the backend: python main.py")
    print("4. Test the API at: http://localhost:8000/docs")
    print("5. Use the frontend example in frontend_example.md")
    print("\n📚 Check README.md for more detailed instructions.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
