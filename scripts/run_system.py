#!/usr/bin/env python3
"""
Complete ARGO Oceanographic Data System Runner
Handles both the advanced RAG system and the existing Streamlit chatbot
"""

import os
import sys
import subprocess
import logging
import time
import signal
from pathlib import Path
from typing import Optional, List
import webbrowser
from threading import Thread

def setup_logging():
    """Setup logging for the runner"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('system_runner.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class SystemRunner:
    def __init__(self):
        self.logger = setup_logging()
        self.processes = []
        self.running = False
        
    def check_requirements(self) -> bool:
        """Check if all requirements are met"""
        self.logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("Python 3.8 or higher is required")
            return False
        
        # Check if requirements.txt exists
        if not Path("requirements.txt").exists():
            self.logger.error("requirements.txt not found")
            return False
        
        # Check if MySQL is accessible (basic check)
        try:
            import mysql.connector
            # Try to connect with default settings
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="Arman123?",  # Default from template
                database="argo_data"
            )
            conn.close()
            self.logger.info("✅ MySQL connection successful")
        except Exception as e:
            self.logger.warning(f"MySQL connection failed: {e}")
            self.logger.info("Please ensure MySQL is running and credentials are correct")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install all required dependencies"""
        self.logger.info("Installing dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            self.logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Setup environment files"""
        self.logger.info("Setting up environment...")
        
        # Create .env file if it doesn't exist
        if not Path(".env").exists():
            if Path(".env.template").exists():
                import shutil
                shutil.copy(".env.template", ".env")
                self.logger.info("Created .env file from template")
                self.logger.warning("Please edit .env file with your API keys")
            else:
                self.logger.error(".env.template not found")
                return False
        
        # Create Streamlit secrets if it doesn't exist
        secrets_path = Path(".streamlit/secrets.toml")
        if not secrets_path.exists():
            template_path = Path(".streamlit/secrets.toml.template")
            if template_path.exists():
                import shutil
                shutil.copy(template_path, secrets_path)
                self.logger.info("Created Streamlit secrets.toml from template")
                self.logger.warning("Please edit .streamlit/secrets.toml with your credentials")
            else:
                self.logger.error("Streamlit secrets template not found")
                return False
        
        return True
    
    def run_data_ingestion(self) -> bool:
        """Run initial data ingestion if database is empty"""
        self.logger.info("Checking if data ingestion is needed...")
        
        try:
            import mysql.connector
            
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="Arman123?",
                database="argo_data"
            )
            cursor = conn.cursor()
            
            # Check if argo_profiles table has data
            cursor.execute("SELECT COUNT(*) FROM argo_profiles")
            count = cursor.fetchone()[0]
            
            conn.close()
            
            if count == 0:
                self.logger.info("Database is empty. Running data ingestion...")
                
                # Run limited data ingestion for demo
                from data import ArgoDataIngestion, ArgoDataProcessor, ArgoDatabase, ArgoConfig
                
                config = ArgoConfig()
                ingestion_system = ArgoDataIngestion(config)
                
                # Get index and limit to small sample for demo
                df_index = ingestion_system.fetch_global_index()
                filtered = ingestion_system.filter_profiles(df_index)
                
                # Limit to 5 profiles for quick demo
                sample = filtered.head(5)
                self.logger.info(f"Processing {len(sample)} sample profiles...")
                
                downloaded_files = ingestion_system.download_profiles_batch(sample)
                
                processor = ArgoDataProcessor(config)
                combined = processor.process_all_files(downloaded_files)
                
                if not combined.empty:
                    from datetime import datetime
                    run_id = f"demo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    database = ArgoDatabase(config)
                    success = database.save_to_database(combined, run_id)
                    
                    if success:
                        self.logger.info(f"✅ Sample data ingestion completed: {len(combined)} records")
                        return True
                    else:
                        self.logger.error("Data ingestion failed")
                        return False
                else:
                    self.logger.warning("No data was processed")
                    return False
            else:
                self.logger.info(f"✅ Database already has {count} records")
                return True
                
        except Exception as e:
            self.logger.error(f"Data ingestion check/run failed: {e}")
            return False
    
    def start_fastapi_server(self) -> subprocess.Popen:
        """Start the FastAPI server"""
        self.logger.info("Starting FastAPI server...")
        
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "api:app",
                "--host", "127.0.0.1",
                "--port", "8000",
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            self.logger.info("✅ FastAPI server started on http://127.0.0.1:8000")
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to start FastAPI server: {e}")
            return None
    
    def start_streamlit_app(self) -> subprocess.Popen:
        """Start the Streamlit chatbot"""
        self.logger.info("Starting Streamlit chatbot...")
        
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "chatbot.py",
                "--server.port", "8501",
                "--server.address", "127.0.0.1"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            self.logger.info("✅ Streamlit app started on http://127.0.0.1:8501")
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to start Streamlit app: {e}")
            return None
    
    def wait_for_server(self, url: str, timeout: int = 30) -> bool:
        """Wait for a server to be ready"""
        import requests
        
        for _ in range(timeout):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        
        return False
    
    def open_browsers(self):
        """Open browsers to the applications"""
        time.sleep(3)  # Wait a bit for servers to start
        
        self.logger.info("Opening applications in browser...")
        
        # Open Streamlit chatbot
        try:
            webbrowser.open("http://127.0.0.1:8501")
            self.logger.info("🌐 Opened Streamlit chatbot")
        except:
            pass
        
        # Open FastAPI docs
        try:
            webbrowser.open("http://127.0.0.1:8000/docs")
            self.logger.info("🌐 Opened FastAPI documentation")
        except:
            pass
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutting down all services...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up all running processes"""
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        self.running = False
        self.logger.info("✅ All services stopped")
    
    def run_system(self):
        """Run the complete system"""
        self.logger.info("🚀 Starting ARGO Oceanographic Data System")
        self.logger.info("=" * 60)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Setup steps
        steps = [
            ("Checking requirements", self.check_requirements),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up environment", self.setup_environment),
            ("Running data ingestion", self.run_data_ingestion),
        ]
        
        for step_name, step_function in steps:
            self.logger.info(f"Step: {step_name}")
            try:
                if not step_function():
                    self.logger.error(f"Step failed: {step_name}")
                    return False
            except Exception as e:
                self.logger.error(f"Step error ({step_name}): {e}")
                return False
        
        self.logger.info("All setup steps completed!")
        self.logger.info("=" * 60)
        
        # Start services
        self.logger.info("Starting services...")
        
        # Start FastAPI server
        fastapi_process = self.start_fastapi_server()
        if not fastapi_process:
            self.logger.error("Failed to start FastAPI server")
            return False
        
        # Start Streamlit app
        streamlit_process = self.start_streamlit_app()
        if not streamlit_process:
            self.logger.error("Failed to start Streamlit app")
            self.cleanup()
            return False
        
        self.running = True
        
        # Open browsers in a separate thread
        browser_thread = Thread(target=self.open_browsers)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Display information
        self.logger.info("=" * 60)
        self.logger.info("🎉 ARGO System is now running!")
        self.logger.info("=" * 60)
        self.logger.info("📱 Streamlit Chatbot: http://127.0.0.1:8501")
        self.logger.info("🚀 FastAPI Server: http://127.0.0.1:8000")
        self.logger.info("📚 API Documentation: http://127.0.0.1:8000/docs")
        self.logger.info("=" * 60)
        self.logger.info("Press Ctrl+C to stop all services")
        self.logger.info("=" * 60)
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
                
                # Check if processes are still running
                for process in self.processes[:]:
                    if process.poll() is not None:
                        self.logger.warning(f"Process {process.pid} has stopped")
                        self.processes.remove(process)
                
                if not self.processes:
                    self.logger.error("All processes have stopped")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        
        self.cleanup()
        return True

def main():
    """Main entry point"""
    runner = SystemRunner()
    
    print("🌊 ARGO Oceanographic Data System")
    print("Complete Natural Language Interface for Ocean Data")
    print("=" * 60)
    
    try:
        success = runner.run_system()
        if success:
            print("\n✅ System ran successfully")
        else:
            print("\n❌ System failed to start")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        runner.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()