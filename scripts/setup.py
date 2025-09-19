"""
Setup and Installation Guide for ARGO Oceanographic Data RAG System
Complete system requirements, installation, and configuration instructions
"""

import os

# Requirements.txt content
REQUIREMENTS = """
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
xarray>=2022.6.0
mysql-connector-python>=8.0.33
netCDF4>=1.6.0

# Vector database dependencies
chromadb>=0.4.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4

# LLM API dependencies
openai>=1.0.0
anthropic>=0.3.0

# Web API dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# Additional utilities
python-dotenv>=1.0.0
asyncio-compat>=0.1.2
typing-extensions>=4.5.0
"""

# Environment configuration template
ENV_TEMPLATE = """
# Database Configuration
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=argo_data

# LLM API Keys (choose one or both)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Vector Database Configuration
VECTOR_DB_PATH=./argo_vector_db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
MAX_QUERY_LENGTH=500
RATE_LIMIT_PER_MINUTE=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=argo_rag.log
"""

# Docker configuration
DOCKERFILE = """
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and logs
RUN mkdir -p /app/argo_data_downloads /app/argo_vector_db /app/logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKER_COMPOSE = """
version: '3.8'

services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: Arman123?
      MYSQL_DATABASE: argo_data
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      timeout: 20s
      retries: 10

  argo-rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=mysql
      - DB_USER=root
      - DB_PASSWORD=Arman123?
      - DB_NAME=argo_data
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./argo_data_downloads:/app/argo_data_downloads
      - ./argo_vector_db:/app/argo_vector_db
      - ./logs:/app/logs
    depends_on:
      mysql:
        condition: service_healthy
    restart: unless-stopped

volumes:
  mysql_data:
"""

def create_requirements_file():
    """Create requirements.txt file"""
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS.strip())
    print("✅ Created requirements.txt")

def create_env_file():
    """Create .env template file"""
    with open(".env.template", "w") as f:
        f.write(ENV_TEMPLATE.strip())
    print("✅ Created .env.template")
    print("📝 Please copy .env.template to .env and fill in your API keys")

def create_docker_files():
    """Create Docker configuration files"""
    with open("Dockerfile", "w") as f:
        f.write(DOCKERFILE.strip())
    
    with open("docker-compose.yml", "w") as f:
        f.write(DOCKER_COMPOSE.strip())
    
    print("✅ Created Dockerfile and docker-compose.yml")

def create_startup_script():
    """Create system startup and initialization script"""
    startup_script = '''#!/usr/bin/env python3
"""
ARGO RAG System Startup and Initialization Script
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for the startup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")
    print(f"✅ Python version: {sys.version}")

def install_dependencies():
    """Install required Python packages"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check if requirements.txt exists
        if not Path("requirements.txt").exists():
            logger.error("requirements.txt not found. Run setup.py first.")
            return False
        
        # Install dependencies
        logger.info("Installing Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_database():
    """Setup MySQL database and tables"""
    logger = logging.getLogger(__name__)
    
    try:
        import mysql.connector
        from data import ArgoDatabase, ArgoConfig
        
        # Test database connection
        config = ArgoConfig()
        database = ArgoDatabase(config)
        database.create_enhanced_schema()
        
        logger.info("✅ Database setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        logger.info("Please ensure MySQL is running and credentials are correct")
        return False

def initialize_vector_database():
    """Initialize vector database"""
    logger = logging.getLogger(__name__)
    
    try:
        from vector_db import ArgoVectorDB, VectorDBConfig
        
        config = VectorDBConfig()
        vector_db = ArgoVectorDB(config)
        
        # Test vector database
        test_results = vector_db.search_semantic("test query", n_results=1)
        logger.info("✅ Vector database initialized")
        return True
        
    except Exception as e:
        logger.error(f"Vector database initialization failed: {e}")
        return False

def test_llm_connection():
    """Test LLM API connection"""
    logger = logging.getLogger(__name__)
    
    try:
        from sql_generator import NaturalLanguageToSQL, SQLGeneratorConfig, LLMProvider
        
        # Test OpenAI connection if API key is available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            config = SQLGeneratorConfig(
                llm_provider=LLMProvider.OPENAI,
                openai_api_key=openai_key
            )
            sql_generator = NaturalLanguageToSQL(config)
            result = sql_generator.generate_sql("test query")
            logger.info("✅ OpenAI API connection working")
            return True
        
        # Test Anthropic connection if API key is available
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            config = SQLGeneratorConfig(
                llm_provider=LLMProvider.ANTHROPIC,
                anthropic_api_key=anthropic_key
            )
            sql_generator = NaturalLanguageToSQL(config)
            result = sql_generator.generate_sql("test query")
            logger.info("✅ Anthropic API connection working")
            return True
        
        logger.warning("No LLM API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return False
        
    except Exception as e:
        logger.error(f"LLM API test failed: {e}")
        return False

def run_data_ingestion():
    """Run initial data ingestion"""
    logger = logging.getLogger(__name__)
    
    try:
        from data import ArgoDataIngestion, ArgoDataProcessor, ArgoDatabase, ArgoConfig
        
        logger.info("Starting data ingestion...")
        config = ArgoConfig()
        
        # Run ingestion pipeline
        ingestion_system = ArgoDataIngestion(config)
        df_index = ingestion_system.fetch_global_index()
        filtered = ingestion_system.filter_profiles(df_index)
        
        # Limit to a small sample for initial setup
        sample_size = min(10, len(filtered))
        filtered_sample = filtered.head(sample_size)
        
        downloaded_files = ingestion_system.download_profiles_batch(filtered_sample)
        
        processor = ArgoDataProcessor(config)
        combined = processor.process_all_files(downloaded_files)
        
        if not combined.empty:
            from datetime import datetime
            run_id = f"startup_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            database = ArgoDatabase(config)
            success = database.save_to_database(combined, run_id)
            
            if success:
                # Initialize vector database with sample data
                from vector_db import ArgoVectorDB, VectorDBConfig
                vector_config = VectorDBConfig()
                vector_db = ArgoVectorDB(vector_config)
                vector_db.ingest_dataframe(combined)
                
                logger.info(f"✅ Initial data ingestion completed: {len(combined)} records")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    logger = logging.getLogger(__name__)
    
    try:
        import uvicorn
        from api import create_app
        
        app = create_app()
        
        logger.info("Starting API server...")
        logger.info("API will be available at: http://127.0.0.1:8000")
        logger.info("API documentation at: http://127.0.0.1:8000/docs")
        
        uvicorn.run(
            "api:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return False

def main():
    """Main startup sequence"""
    logger = setup_logging()
    
    print("🚀 ARGO Oceanographic Data RAG System Startup")
    print("=" * 50)
    
    # Check system requirements
    try:
        check_python_version()
    except RuntimeError as e:
        logger.error(e)
        sys.exit(1)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("✅ Environment variables loaded")
    except ImportError:
        logger.warning("python-dotenv not available, skipping .env file loading")
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
    
    # Installation and setup steps
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up database", setup_database),
        ("Initializing vector database", initialize_vector_database),
        ("Testing LLM connection", test_llm_connection),
        ("Running initial data ingestion", run_data_ingestion),
    ]
    
    for step_name, step_function in steps:
        logger.info(f"Starting: {step_name}")
        try:
            success = step_function()
            if not success:
                logger.warning(f"Step failed: {step_name}")
        except Exception as e:
            logger.error(f"Step error ({step_name}): {e}")
    
    # Start API server
    logger.info("All setup steps completed!")
    logger.info("Starting API server...")
    start_api_server()

if __name__ == "__main__":
    main()
'''
    
    with open("startup.py", "w") as f:
        f.write(startup_script)
    
    # Make it executable on Unix systems
    try:
        os.chmod("startup.py", 0o755)
    except:
        pass
    
    print("✅ Created startup.py script")

def create_readme():
    """Create comprehensive README file"""
    readme_content = '''# ARGO Oceanographic Data RAG System

A comprehensive Retrieval Augmented Generation (RAG) system for natural language querying of ARGO oceanographic data.

## Features

- 🌊 **Data Ingestion**: Download and process ARGO NetCDF files from IFREMER FTP
- 📊 **Dual Database Storage**: MySQL for structured data, ChromaDB for vector embeddings
- 🤖 **Natural Language Processing**: Convert questions to SQL using OpenAI/Anthropic APIs
- 🔍 **Semantic Search**: Find similar oceanographic patterns using vector similarity
- 🚀 **RESTful API**: FastAPI backend with comprehensive endpoints
- 📈 **Hybrid Querying**: Combine analytical SQL with semantic vector search

## System Architecture

```
Natural Language Query
        ↓
Query Classification (Semantic/Analytical/Hybrid)
        ↓
┌─────────────────┬─────────────────┐
│   Vector Search │   SQL Generation│
│   (ChromaDB)    │   (LLM → MySQL) │
└─────────────────┴─────────────────┘
        ↓
Result Combination & Ranking
        ↓
Structured Response with Summary
```

## Quick Start

### Prerequisites

- Python 3.8+
- MySQL 8.0+
- OpenAI or Anthropic API key

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd argo-rag-system
   python setup.py  # Creates requirements.txt and config files
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and database credentials
   ```

4. **Run startup script**:
   ```bash
   python startup.py
   ```

### Docker Deployment

```bash
# Set your API keys in .env file first
docker-compose up -d
```

## API Usage

### Basic Query
```bash
curl -X POST "http://localhost:8000/query" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "Show temperature profiles from tropical regions"}'
```

### Query Examples

| Query Type | Example | Description |
|------------|---------|-------------|
| **Semantic** | "Find data similar to warm tropical waters" | Vector similarity search |
| **Analytical** | "What's the average salinity at 500m depth?" | SQL aggregation |
| **Hybrid** | "Compare oxygen levels in tropical vs temperate regions" | Combined approach |

### API Endpoints

- `POST /query` - Submit natural language queries
- `GET /health` - System health check
- `POST /batch-query` - Multiple queries
- `GET /schema` - Database schema info
- `POST /ingest` - Trigger data ingestion

## Configuration

### Vector Database Settings
```python
VectorDBConfig(
    db_type="chroma",
    embedding_model="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)
```

### SQL Generator Settings
```python
SQLGeneratorConfig(
    llm_provider=LLMProvider.OPENAI,
    model_name="gpt-3.5-turbo",
    temperature=0.1
)
```

## Data Processing Pipeline

1. **Download**: Fetch ARGO data from IFREMER FTP
2. **Parse**: Extract oceanographic measurements from NetCDF
3. **Store**: Save to MySQL with quality control
4. **Embed**: Generate semantic embeddings for vector search
5. **Index**: Create searchable vector database

## Query Processing Flow

### Semantic Queries
- Generate embeddings for user query
- Search vector database for similar content
- Return ranked results by similarity

### Analytical Queries
- Parse natural language to identify intent
- Generate SQL query using LLM
- Execute against MySQL database
- Return structured data

### Hybrid Queries
- Execute both semantic and analytical searches
- Combine and rank results
- Provide comprehensive response

## Development

### Project Structure
```
├── data.py           # Data ingestion and processing
├── vector_db.py      # Vector database integration
├── sql_generator.py  # Natural language to SQL
├── rag_system.py     # RAG orchestration
├── api.py            # FastAPI web interface
├── startup.py        # System initialization
└── setup.py          # Installation setup
```

### Adding New Features

1. **New Data Sources**: Extend `ArgoDataIngestion` class
2. **Custom Embeddings**: Implement new `EmbeddingGenerator`
3. **Query Types**: Add to `QueryClassifier`
4. **API Endpoints**: Extend FastAPI routes

## Monitoring and Logging

- System health checks via `/health` endpoint
- Comprehensive logging to files and console
- Processing metadata tracking in database
- Performance metrics for query processing

## Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check MySQL service
sudo systemctl status mysql
# Verify credentials in .env file
```

**Vector Database Errors**
```bash
# Reset ChromaDB
rm -rf ./argo_vector_db
python -c "from vector_db import ArgoVectorDB, VectorDBConfig; ArgoVectorDB(VectorDBConfig())"
```

**LLM API Issues**
```bash
# Verify API keys
echo $OPENAI_API_KEY
# Test connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

## Performance Optimization

### Database Optimization
- Use indexed columns for filtering
- Implement data partitioning for large datasets
- Regular table maintenance and optimization

### Vector Search Optimization
- Tune embedding model for domain-specific content
- Implement approximate nearest neighbor search
- Cache frequently accessed embeddings

### API Performance
- Implement request caching
- Use async processing for batch queries
- Load balancing for high traffic

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ARGO Program for providing oceanographic data
- IFREMER for data hosting and access
- OpenAI/Anthropic for LLM capabilities
- ChromaDB team for vector database technology

## Support

For questions and support:
- 📧 Create an issue on GitHub
- 📚 Check the documentation
- 💬 Join our community discussions
'''
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README.md")

def main():
    """Main setup function"""
    print("🚀 Setting up ARGO Oceanographic Data RAG System")
    print("=" * 60)
    
    # Create all necessary files
    create_requirements_file()
    create_env_file()
    create_docker_files()
    create_startup_script()
    create_readme()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Ensure MySQL is running")
    print("3. Run: pip install -r requirements.txt")
    print("4. Run: python startup.py")
    print("\nAPI will be available at: http://127.0.0.1:8000")
    print("Documentation at: http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    main()