"""
Simple Run Script for ARGO Oceanographic Data System
Runs the Streamlit chatbot using Google Gemini API
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_requirements():
    """Check if basic requirements are met"""
    logger = logging.getLogger(__name__)
    
    # Check if .env file exists
    if not Path(".env").exists() and not Path(".env.template").exists():
        logger.error("No .env or .env.template file found!")
        return False
    
    # Check if chatbot.py exists
    if not Path("chatbot.py").exists():
        logger.error("chatbot.py not found!")
        return False
    
    # Check MySQL connection (optional)
    try:
        import mysql.connector
        # You can add a test connection here if needed
        logger.info("✅ MySQL connector available")
    except ImportError:
        logger.warning("MySQL connector not available - some features may not work")
    
    return True

def install_basic_dependencies():
    """Install only the essential dependencies for the chatbot"""
    logger = logging.getLogger(__name__)
    
    essential_packages = [
        "streamlit>=1.28.0",
        "google-generativeai>=0.3.0", 
        "langchain-google-genai>=1.0.0",
        "langchain-community>=0.0.10",
        "sqlalchemy>=2.0.0",
        "mysql-connector-python>=8.0.33",
        "python-dotenv>=1.0.0"
    ]
    
    try:
        for package in essential_packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logger.info("✅ Essential dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    logger = logging.getLogger(__name__)
    
    try:
        from dotenv import load_dotenv
        
        # Try to load .env file
        if Path(".env").exists():
            load_dotenv(".env")
            logger.info("✅ Loaded .env file")
        elif Path(".env.template").exists():
            load_dotenv(".env.template")
            logger.info("✅ Loaded .env.template file")
        
        # Check if Google API key is available
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key or google_api_key == "your_google_api_key_here":
            logger.warning("⚠️  Google API key not properly set in environment")
            logger.info("Please set GOOGLE_API_KEY in your .env file")
            return False
        
        logger.info("✅ Google API key found")
        return True
        
    except ImportError:
        logger.warning("python-dotenv not available, skipping .env loading")
        return True

def create_streamlit_secrets():
    """Create Streamlit secrets file from environment"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create .streamlit directory if it doesn't exist
        streamlit_dir = Path(".streamlit")
        streamlit_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        from dotenv import load_dotenv
        if Path(".env").exists():
            load_dotenv(".env")
        elif Path(".env.template").exists():
            load_dotenv(".env.template")
        
        # Create secrets.toml
        secrets_content = f'''[secrets]
db_username = "{os.getenv('DB_USER', 'root')}"
db_password = "{os.getenv('DB_PASSWORD', 'Arman123?')}"
google_api_key = "{os.getenv('GOOGLE_API_KEY', '')}"
'''
        
        secrets_file = streamlit_dir / "secrets.toml"
        with open(secrets_file, "w") as f:
            f.write(secrets_content)
        
        logger.info("✅ Created Streamlit secrets.toml")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create Streamlit secrets: {e}")
        return False

def run_streamlit_chatbot():
    """Run the Streamlit chatbot"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Streamlit chatbot...")
        logger.info("The chatbot will be available at: http://localhost:8501")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "chatbot.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        logger.info("Shutting down chatbot...")
    except Exception as e:
        logger.error(f"Failed to run Streamlit: {e}")

def main():
    """Main execution function"""
    logger = setup_logging()
    
    print("🌊 ARGO Oceanographic Data Chatbot")
    print("=" * 50)
    
    # Check basic requirements
    if not check_requirements():
        logger.error("Requirements check failed!")
        sys.exit(1)
    
    # Install essential dependencies
    logger.info("Installing essential dependencies...")
    if not install_basic_dependencies():
        logger.error("Failed to install dependencies!")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed!")
        logger.info("Please check your .env file and ensure GOOGLE_API_KEY is set")
        sys.exit(1)
    
    # Create Streamlit secrets
    if not create_streamlit_secrets():
        logger.error("Failed to create Streamlit secrets!")
        sys.exit(1)
    
    # Run the chatbot
    logger.info("All checks passed! Starting chatbot...")
    run_streamlit_chatbot()

if __name__ == "__main__":
    main()