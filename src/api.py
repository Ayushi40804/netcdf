"""
FastAPI Backend for ARGO Oceanographic Data RAG System
Provides RESTful API endpoints for natural language queries
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

# Import our RAG system
from rag_system import ArgoRAGSystem, RAGConfig, QueryType
from vector_db import VectorDBConfig
from sql_generator import SQLGeneratorConfig, LLMProvider

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for natural language queries"""
    query: str = Field(..., description="Natural language query about oceanographic data")
    max_results: Optional[int] = Field(20, description="Maximum number of results to return")
    include_metadata: Optional[bool] = Field(True, description="Include metadata in response")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters to apply")

class QueryResponse(BaseModel):
    """Response model for query results"""
    success: bool
    query: str
    processing_time_seconds: float
    summary: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Response model for health check"""
    overall_status: str
    timestamp: str
    components: Dict[str, Dict[str, str]]

class DataIngestionRequest(BaseModel):
    """Request model for data ingestion"""
    data_source: str = Field(..., description="Source of data to ingest")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")

class BatchQueryRequest(BaseModel):
    """Request model for batch queries"""
    queries: List[str] = Field(..., description="List of natural language queries")
    max_results_per_query: Optional[int] = Field(10, description="Max results per query")

# Global variables for the RAG system
rag_system: Optional[ArgoRAGSystem] = None
app_config: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global rag_system, app_config
    
    logger = logging.getLogger(__name__)
    logger.info("Starting ARGO RAG API...")
    
    try:
        # Load configuration
        app_config = load_configuration()
        
        # Initialize RAG system
        rag_config = create_rag_config(app_config)
        rag_system = ArgoRAGSystem(rag_config)
        
        # Health check
        health = rag_system.health_check()
        if health["overall_status"] != "healthy":
            logger.warning(f"RAG system health check: {health}")
        
        logger.info("ARGO RAG API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ARGO RAG API...")

# Create FastAPI app with lifespan
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="ARGO Oceanographic Data RAG API",
        description="Natural language interface for ARGO oceanographic data using RAG (Retrieval Augmented Generation)",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app = None

def load_configuration() -> Dict[str, Any]:
    """Load application configuration"""
    config = {
        "vector_db": {
            "db_type": "chroma",
            "embedding_model": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "db_path": "./argo_vector_db"
        },
        "sql_generator": {
            "llm_provider": "openai",
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": "gpt-3.5-turbo",
            "db_host": os.getenv("DB_HOST", "localhost"),
            "db_user": os.getenv("DB_USER", "root"),
            "db_password": os.getenv("DB_PASSWORD", "Arman123?"),
            "db_name": os.getenv("DB_NAME", "argo_data")
        },
        "rag_system": {
            "max_vector_results": 20,
            "max_sql_results": 1000,
            "similarity_threshold": 0.7
        },
        "api": {
            "max_query_length": 500,
            "rate_limit_per_minute": 60,
            "enable_batch_queries": True,
            "max_batch_size": 10
        }
    }
    
    return config

def create_rag_config(app_config: Dict[str, Any]) -> RAGConfig:
    """Create RAG configuration from app config"""
    vector_config = VectorDBConfig(
        db_type=app_config["vector_db"]["db_type"],
        embedding_model=app_config["vector_db"]["embedding_model"],
        model_name=app_config["vector_db"]["model_name"],
        db_path=app_config["vector_db"]["db_path"]
    )
    
    sql_config = SQLGeneratorConfig(
        llm_provider=LLMProvider(app_config["sql_generator"]["llm_provider"]),
        openai_api_key=app_config["sql_generator"]["openai_api_key"],
        model_name=app_config["sql_generator"]["model_name"],
        db_host=app_config["sql_generator"]["db_host"],
        db_user=app_config["sql_generator"]["db_user"],
        db_password=app_config["sql_generator"]["db_password"],
        db_name=app_config["sql_generator"]["db_name"]
    )
    
    rag_config = RAGConfig(
        vector_db_config=vector_config,
        sql_config=sql_config,
        max_vector_results=app_config["rag_system"]["max_vector_results"],
        max_sql_results=app_config["rag_system"]["max_sql_results"],
        similarity_threshold=app_config["rag_system"]["similarity_threshold"]
    )
    
    return rag_config

def get_rag_system() -> ArgoRAGSystem:
    """Dependency to get RAG system instance"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system

def validate_query(query: str) -> str:
    """Validate and sanitize query"""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    query = query.strip()
    max_length = app_config.get("api", {}).get("max_query_length", 500)
    
    if len(query) > max_length:
        raise HTTPException(
            status_code=400, 
            detail=f"Query too long. Maximum {max_length} characters allowed"
        )
    
    return query

# API Endpoints
if FASTAPI_AVAILABLE:
    
    @app.get("/", summary="API Information")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "ARGO Oceanographic Data RAG API",
            "version": "1.0.0",
            "description": "Natural language interface for ARGO oceanographic data",
            "endpoints": {
                "query": "/query - Submit natural language queries",
                "health": "/health - System health check",
                "batch": "/batch-query - Submit multiple queries",
                "schema": "/schema - Database schema information"
            }
        }
    
    @app.post("/query", response_model=QueryResponse, summary="Submit Natural Language Query")
    async def query_data(
        request: QueryRequest,
        rag_system: ArgoRAGSystem = Depends(get_rag_system)
    ):
        """
        Submit a natural language query about oceanographic data.
        
        The system will:
        1. Classify the query type (semantic, analytical, or hybrid)
        2. Search vector database for semantically similar data
        3. Generate and execute SQL queries for analytical data
        4. Combine and rank results
        5. Return structured response with summary
        """
        try:
            # Validate query
            validated_query = validate_query(request.query)
            
            # Process query through RAG system
            result = await rag_system.query_async(validated_query)
            
            # Apply request-specific filters
            if request.max_results:
                # Limit results
                if "combined_results" in result.get("results", {}):
                    result["results"]["combined_results"] = result["results"]["combined_results"][:request.max_results]
            
            if not request.include_metadata:
                # Remove metadata if not requested
                result.pop("metadata", None)
                if "results" in result:
                    for result_type in ["vector_results", "sql_results"]:
                        if result_type in result["results"]:
                            for item in result["results"][result_type].get("results", []):
                                item.pop("metadata", None)
            
            return QueryResponse(**result)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")
    
    @app.get("/health", response_model=HealthResponse, summary="System Health Check")
    async def health_check(rag_system: ArgoRAGSystem = Depends(get_rag_system)):
        """
        Check the health status of all system components.
        
        Returns status for:
        - Vector database connection and search
        - SQL generator and database connection
        - LLM API connectivity
        """
        try:
            health = rag_system.health_check()
            return HealthResponse(**health)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    
    @app.post("/batch-query", summary="Submit Multiple Queries")
    async def batch_query(
        request: BatchQueryRequest,
        background_tasks: BackgroundTasks,
        rag_system: ArgoRAGSystem = Depends(get_rag_system)
    ):
        """
        Submit multiple natural language queries in a single request.
        Results are processed asynchronously and returned when complete.
        """
        if not app_config.get("api", {}).get("enable_batch_queries", True):
            raise HTTPException(status_code=403, detail="Batch queries are disabled")
        
        max_batch_size = app_config.get("api", {}).get("max_batch_size", 10)
        if len(request.queries) > max_batch_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size too large. Maximum {max_batch_size} queries allowed"
            )
        
        try:
            # Process queries concurrently
            tasks = []
            for query in request.queries:
                validated_query = validate_query(query)
                task = asyncio.create_task(rag_system.query_async(validated_query))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Format batch response
            batch_response = {
                "total_queries": len(request.queries),
                "successful": 0,
                "failed": 0,
                "results": []
            }
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    batch_response["failed"] += 1
                    batch_response["results"].append({
                        "query_index": i,
                        "query": request.queries[i],
                        "success": False,
                        "error": str(result)
                    })
                else:
                    batch_response["successful"] += 1
                    # Limit results per query
                    if request.max_results_per_query:
                        if "combined_results" in result.get("results", {}):
                            result["results"]["combined_results"] = result["results"]["combined_results"][:request.max_results_per_query]
                    
                    batch_response["results"].append({
                        "query_index": i,
                        **result
                    })
            
            return batch_response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")
    
    @app.get("/schema", summary="Database Schema Information")
    async def get_schema(rag_system: ArgoRAGSystem = Depends(get_rag_system)):
        """
        Get information about the database schema and available data fields.
        Useful for understanding what data is available for querying.
        """
        try:
            schema_info = rag_system.sql_generator.schema_manager.schema_info
            return {
                "schema": schema_info,
                "query_examples": [
                    "Show temperature data from March 2023",
                    "What's the average salinity at 500 meters depth?",
                    "Find profiles with high dissolved oxygen levels",
                    "Compare data between different ARGO floats",
                    "Show seasonal temperature variations"
                ],
                "available_fields": {
                    "geographic": ["latitude", "longitude"],
                    "temporal": ["date"],
                    "physical": ["depth", "temperature", "salinity", "dissolved_oxygen"],
                    "metadata": ["float_id", "profile_index", "level_index"]
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Schema retrieval error: {str(e)}")
    
    @app.post("/ingest", summary="Trigger Data Ingestion")
    async def trigger_ingestion(
        request: DataIngestionRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Trigger data ingestion process in the background.
        This endpoint initiates the process of downloading and processing new ARGO data.
        """
        def run_ingestion():
            """Background task for data ingestion"""
            try:
                # Import and run data ingestion
                from data import ArgoDataIngestion, ArgoDataProcessor, ArgoDatabase, ArgoConfig
                
                config = ArgoConfig()
                if request.config_overrides:
                    for key, value in request.config_overrides.items():
                        setattr(config, key, value)
                
                # Run ingestion pipeline
                ingestion_system = ArgoDataIngestion(config)
                df_index = ingestion_system.fetch_global_index()
                filtered = ingestion_system.filter_profiles(df_index)
                downloaded_files = ingestion_system.download_profiles_batch(filtered)
                
                processor = ArgoDataProcessor(config)
                combined = processor.process_all_files(downloaded_files)
                
                if not combined.empty:
                    run_id = f"api_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    database = ArgoDatabase(config)
                    success = database.save_to_database(combined, run_id)
                    
                    if success:
                        # Update vector database
                        rag_system.vector_db.ingest_dataframe(combined)
                        logging.info(f"Data ingestion completed successfully: {run_id}")
                    else:
                        logging.error(f"Data ingestion failed: {run_id}")
                
            except Exception as e:
                logging.error(f"Background ingestion error: {e}")
        
        # Add to background tasks
        background_tasks.add_task(run_ingestion)
        
        return {
            "message": "Data ingestion started in background",
            "timestamp": datetime.now().isoformat(),
            "source": request.data_source
        }
    
    @app.get("/query-types", summary="Get Query Type Examples")
    async def get_query_types():
        """
        Get examples of different query types supported by the system.
        """
        return {
            "query_types": {
                "semantic": {
                    "description": "Find similar data patterns or characteristics",
                    "examples": [
                        "Find data similar to tropical ocean conditions",
                        "Show profiles with warm water characteristics",
                        "Find measurements from deep ocean environments"
                    ],
                    "keywords": ["similar", "like", "find", "show me", "characteristics"]
                },
                "analytical": {
                    "description": "Statistical analysis and aggregations",
                    "examples": [
                        "What's the average temperature at 1000m depth?",
                        "Count the number of profiles per month",
                        "Calculate maximum salinity values by region"
                    ],
                    "keywords": ["average", "count", "sum", "maximum", "minimum", "statistics"]
                },
                "hybrid": {
                    "description": "Combination of semantic and analytical approaches",
                    "examples": [
                        "Compare average temperatures in tropical vs temperate regions",
                        "Find and analyze oxygen-rich water masses",
                        "Show temperature trends in similar oceanographic conditions"
                    ],
                    "keywords": ["compare", "analyze", "trends", "patterns"]
                }
            }
        }

# Development server setup
def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required to run the API server")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return app

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        exit(1)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the development server
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )