#!/usr/bin/env python3
"""
Quick Start Script for ARGO System
Installs dependencies and runs basic components
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    # Basic packages needed
    basic_packages = [
        "streamlit",
        "pandas",
        "mysql-connector-python",
        "sqlalchemy",
        "langchain",
        "langchain-google-genai",
        "langchain-community",
        "python-dotenv"
    ]
    
    for package in basic_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("✅ Basic dependencies installed")

def setup_config():
    """Setup basic configuration"""
    print("🔧 Setting up configuration...")
    
    # Create .streamlit directory if it doesn't exist
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    # Create a basic secrets.toml if it doesn't exist
    secrets_file = streamlit_dir / "secrets.toml"
    if not secrets_file.exists():
        secrets_content = """[database]
db_username = "root"
db_password = "Arman123?"

[api_keys]
google_api_key = "your_google_api_key_here"
"""
        with open(secrets_file, "w") as f:
            f.write(secrets_content)
        
        print("✅ Created basic secrets.toml")
        print("⚠️ Please edit .streamlit/secrets.toml with your Google API key")

def run_chatbot():
    """Run the Streamlit chatbot"""
    print("🚀 Starting Streamlit chatbot...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "chatbot.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped")
    except Exception as e:
        print(f"❌ Error running chatbot: {e}")

def main():
    print("🌊 ARGO Oceanographic Data System - Quick Start")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    
    print(f"✅ Python {sys.version.split()[0]} found")
    
    # Install dependencies
    install_dependencies()
    
    # Setup configuration
    setup_config()
    
    print("\n" + "=" * 60)
    print("🎉 Setup complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Edit .streamlit/secrets.toml with your Google API key")
    print("2. Ensure MySQL is running with the argo_data database")
    print("3. The chatbot will start automatically...")
    print("=" * 60)
    
    input("Press Enter to start the chatbot (after setting up your API key)...")
    
    # Run the chatbot
    run_chatbot()

if __name__ == "__main__":
    main()