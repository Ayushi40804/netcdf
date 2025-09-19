# ARGO NetCDF Processor

A comprehensive system for processing ARGO oceanographic data from NetCDF files and providing AI-powered natural language querying capabilities.

## 🌊 Overview

This project provides a complete pipeline for:
- **Data Ingestion**: Converting ARGO ocean float data from NetCDF format to structured databases
- **AI-Powered Querying**: Using Google Gemini LLM with RAG (Retrieval Augmented Generation) for natural language queries
- **Dual Database System**: SQLite for structured data + ChromaDB for semantic vector search
- **Web Interface**: Streamlit-based chatbot for interactive data exploration

## 📁 Project Structure

```
argo_netcdf_processor/
├── src/                     # Core application modules
│   ├── data.py             # ARGO data ingestion and processing
│   ├── sql_generator.py    # Natural language to SQL conversion
│   ├── vector_db.py        # Vector database and semantic search
│   ├── rag_system.py       # RAG system combining SQL + vector search
│   └── api.py              # FastAPI backend endpoints
├── config/                  # Configuration files
│   ├── .env                # Environment variables (API keys, DB config)
│   ├── .env.template       # Template for environment variables
│   └── .streamlit/         # Streamlit configuration
├── scripts/                 # Utility scripts and setup
│   ├── setup.py            # System installation and configuration
│   ├── run_chatbot.py      # Streamlit chatbot launcher
│   ├── run_system.py       # Complete system runner
│   ├── quick_start.py      # Quick demo setup
│   └── setup_and_run.bat   # Windows batch file for easy setup
├── docs/                    # Documentation
├── tests/                   # Unit tests
├── chatbot.py              # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ayushi40804/netcdf.git
   cd argo_netcdf_processor
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy template and edit with your API key
   cp config/.env.template config/.env
   # Edit config/.env and add your Google Gemini API key
   ```

5. **Run the application**
   ```bash
   streamlit run chatbot.py --server.port 8501
   ```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Database Configuration (SQLite - no setup required)
DATABASE_PATH=argo_data.sqlite

# Optional: Advanced configuration
VECTOR_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Streamlit Configuration
The application uses Streamlit secrets for secure configuration. Update `config/.streamlit/secrets.toml`:

```toml
[api_keys]
google_api_key = "your_google_gemini_api_key"

[database]
db_path = "argo_data.sqlite"
```

## 📊 Features

### Data Processing
- **NetCDF Ingestion**: Automatic download and processing of ARGO float data
- **Quality Control**: Built-in data validation and filtering
- **Batch Processing**: Efficient handling of large datasets
- **Multiple Formats**: Support for various ARGO data formats

### AI-Powered Querying
- **Natural Language**: Ask questions in plain English
- **Smart Query Routing**: Automatically chooses between SQL and semantic search
- **Context Awareness**: Maintains conversation context
- **Error Handling**: Graceful handling of complex queries

### Database Systems
- **SQLite**: Fast, file-based SQL database for structured queries
- **ChromaDB**: Vector database for semantic similarity search
- **Hybrid Search**: Combines both approaches for comprehensive results

## 🎯 Example Queries

- "What is the average temperature for float 5904297?"
- "Show me salinity measurements from the Indian Ocean"
- "Find temperature anomalies in recent data"
- "List all floats with their GPS coordinates"
- "What's the deepest measurement in the database?"

## 🛠️ Development

### Project Components

1. **Data Ingestion (`src/data.py`)**
   - ARGO data download from IFREMER FTP
   - NetCDF file parsing with xarray
   - Database schema management
   - Quality control and validation

2. **SQL Generator (`src/sql_generator.py`)**
   - Google Gemini integration for natural language processing
   - SQL query generation and validation
   - Error handling and query optimization

3. **Vector Database (`src/vector_db.py`)**
   - ChromaDB setup and management
   - Sentence transformer embeddings
   - Semantic similarity search

4. **RAG System (`src/rag_system.py`)**
   - Query classification and routing
   - Context management
   - Response synthesis

5. **API Backend (`src/api.py`)**
   - FastAPI endpoints
   - RESTful API design
   - Authentication and rate limiting

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ARGO Project for providing oceanographic data
- Google for Gemini API
- Streamlit team for the amazing framework
- ChromaDB for vector database capabilities

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review example queries in the application

---

**Happy Ocean Data Exploration! 🌊🔍**