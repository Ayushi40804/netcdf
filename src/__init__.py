"""
ARGO NetCDF Processor

A comprehensive system for processing ARGO oceanographic data and providing
AI-powered natural language querying capabilities.
"""

__version__ = "1.0.0"
__author__ = "ARGO NetCDF Team"
__email__ = "support@argo-netcdf.com"

# Package imports
from .data import ArgoDataIngestion, ArgoDataProcessor, ArgoDatabase
from .vector_db import ArgoVectorDB, ArgoEmbeddingGenerator
from .sql_generator import ArgoSQLGenerator, LLMProvider
from .rag_system import ArgoRAGSystem

__all__ = [
    "ArgoDataIngestion",
    "ArgoDataProcessor", 
    "ArgoDatabase",
    "ArgoVectorDB",
    "ArgoEmbeddingGenerator",
    "ArgoSQLGenerator",
    "LLMProvider",
    "ArgoRAGSystem"
]