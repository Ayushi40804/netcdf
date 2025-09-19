"""
Vector Database Integration for ARGO Oceanographic Data
Provides semantic search capabilities for oceanographic data using embeddings
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu")

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available. Install with: pip install sentence-transformers")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

@dataclass
class VectorDBConfig:
    """Configuration for vector database operations"""
    db_type: str = "chroma"  # "chroma" or "faiss"
    embedding_model: str = "sentence-transformers"  # "sentence-transformers" or "openai"
    model_name: str = "all-MiniLM-L6-v2"  # For sentence-transformers
    db_path: str = "./argo_vector_db"
    collection_name: str = "argo_oceanographic_data"
    chunk_size: int = 1000
    embedding_dimension: int = 384  # For all-MiniLM-L6-v2
    openai_api_key: Optional[str] = None

class ArgoEmbeddingGenerator:
    """Generate embeddings for oceanographic data descriptions"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on configuration"""
        if self.config.embedding_model == "sentence-transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("SentenceTransformers is required but not available")
            
            self.model = SentenceTransformer(self.config.model_name)
            self.logger.info(f"Initialized SentenceTransformer model: {self.config.model_name}")
            
        elif self.config.embedding_model == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI is required but not available")
            
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            
            openai.api_key = self.config.openai_api_key
            self.logger.info("Initialized OpenAI embeddings")
        
        else:
            raise ValueError(f"Unsupported embedding model: {self.config.embedding_model}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.config.embedding_model == "sentence-transformers":
            return self.model.encode(texts, convert_to_numpy=True)
        
        elif self.config.embedding_model == "openai":
            # Note: OpenAI API calls should be batched and rate-limited in production
            embeddings = []
            for text in texts:
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embeddings.append(response['data'][0]['embedding'])
            return np.array(embeddings)
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text])[0]

class ArgoDataDescriptionGenerator:
    """Generate rich text descriptions of oceanographic data for embedding"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_profile_description(self, row: pd.Series) -> str:
        """Create a rich text description for an oceanographic profile"""
        description_parts = []
        
        # Basic location and time information
        description_parts.append(f"Oceanographic profile from ARGO float {row.get('float_id', 'unknown')}")
        
        if 'date' in row:
            description_parts.append(f"collected on {row['date']}")
        
        if 'latitude' in row and 'longitude' in row:
            lat_dir = "North" if row['latitude'] >= 0 else "South"
            lon_dir = "East" if row['longitude'] >= 0 else "West"
            description_parts.append(
                f"at location {abs(row['latitude']):.3f}°{lat_dir}, "
                f"{abs(row['longitude']):.3f}°{lon_dir}"
            )
        
        if 'depth' in row:
            description_parts.append(f"at depth {row['depth']:.1f} meters")
        
        # Oceanographic measurements
        measurements = []
        if 'temperature' in row and pd.notna(row['temperature']):
            measurements.append(f"temperature {row['temperature']:.3f}°C")
        
        if 'salinity' in row and pd.notna(row['salinity']):
            measurements.append(f"salinity {row['salinity']:.3f} PSU")
        
        if 'dissolved_oxygen' in row and pd.notna(row['dissolved_oxygen']):
            measurements.append(f"dissolved oxygen {row['dissolved_oxygen']:.3f} μmol/kg")
        
        if measurements:
            description_parts.append(f"with measured {', '.join(measurements)}")
        
        # Environmental context
        if 'depth' in row:
            depth = row['depth']
            if depth < 100:
                description_parts.append("in the surface mixed layer")
            elif depth < 1000:
                description_parts.append("in the thermocline region")
            else:
                description_parts.append("in the deep ocean")
        
        # Seasonal context if date available
        if 'date' in row:
            month = pd.to_datetime(row['date']).month
            if month in [12, 1, 2]:
                season = "winter"
            elif month in [3, 4, 5]:
                season = "spring"
            elif month in [6, 7, 8]:
                season = "summer"
            else:
                season = "autumn"
            description_parts.append(f"during {season} conditions")
        
        return ". ".join(description_parts) + "."
    
    def create_aggregated_description(self, df: pd.DataFrame) -> str:
        """Create description for aggregated data (e.g., by region, time period)"""
        description_parts = []
        
        # Spatial coverage
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_range = f"{df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°"
            lon_range = f"{df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°"
            description_parts.append(f"Oceanographic data from {lat_range} latitude, {lon_range} longitude")
        
        # Temporal coverage
        if 'date' in df.columns:
            start_date = df['date'].min()
            end_date = df['date'].max()
            description_parts.append(f"spanning from {start_date} to {end_date}")
        
        # Depth coverage
        if 'depth' in df.columns:
            depth_range = f"{df['depth'].min():.1f}m to {df['depth'].max():.1f}m depth"
            description_parts.append(f"covering {depth_range}")
        
        # Data statistics
        description_parts.append(f"comprising {len(df)} measurements")
        
        # Variable coverage
        variables = []
        for var in ['temperature', 'salinity', 'dissolved_oxygen']:
            if var in df.columns:
                valid_count = df[var].notna().sum()
                if valid_count > 0:
                    mean_val = df[var].mean()
                    variables.append(f"{var} (mean: {mean_val:.3f}, {valid_count} measurements)")
        
        if variables:
            description_parts.append(f"including {', '.join(variables)}")
        
        return ". ".join(description_parts) + "."

class ChromaVectorDB:
    """ChromaDB implementation for vector storage"""
    
    def __init__(self, config: VectorDBConfig):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is required but not available")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(path=self.config.db_path)
        
        try:
            self.collection = self.client.get_collection(self.config.collection_name)
            self.logger.info(f"Connected to existing collection: {self.config.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"description": "ARGO oceanographic data embeddings"}
            )
            self.logger.info(f"Created new collection: {self.config.collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], 
                     embeddings: np.ndarray, ids: List[str]):
        """Add documents to the vector database"""
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
            ids=ids
        )
        self.logger.info(f"Added {len(documents)} documents to vector database")
    
    def search(self, query_embedding: np.ndarray, n_results: int = 10) -> List[Dict]:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return [{
            'id': results['ids'][0][i],
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        } for i in range(len(results['ids'][0]))]

class ArgoVectorDB:
    """Main vector database interface for ARGO data"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.embedding_generator = ArgoEmbeddingGenerator(config)
        self.description_generator = ArgoDataDescriptionGenerator()
        
        # Initialize vector database
        if config.db_type == "chroma":
            self.vector_db = ChromaVectorDB(config)
        else:
            raise ValueError(f"Unsupported vector database type: {config.db_type}")
    
    def ingest_dataframe(self, df: pd.DataFrame, batch_size: Optional[int] = None):
        """Ingest a pandas DataFrame into the vector database"""
        if batch_size is None:
            batch_size = self.config.chunk_size
        
        self.logger.info(f"Starting ingestion of {len(df)} records")
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Generate description
            description = self.description_generator.create_profile_description(row)
            documents.append(description)
            
            # Create metadata
            metadata = {
                'float_id': str(row.get('float_id', '')),
                'date': str(row.get('date', '')),
                'latitude': float(row.get('latitude', 0)),
                'longitude': float(row.get('longitude', 0)),
                'depth': float(row.get('depth', 0)),
                'has_temperature': bool(pd.notna(row.get('temperature'))),
                'has_salinity': bool(pd.notna(row.get('salinity'))),
                'has_dissolved_oxygen': bool(pd.notna(row.get('dissolved_oxygen'))),
                'ingestion_time': datetime.now().isoformat()
            }
            metadatas.append(metadata)
            
            # Create unique ID
            id_string = f"{row.get('float_id', '')}-{row.get('date', '')}-{row.get('depth', '')}"
            unique_id = hashlib.md5(id_string.encode()).hexdigest()
            ids.append(unique_id)
            
            # Process in batches
            if len(documents) >= batch_size:
                self._process_batch(documents, metadatas, ids)
                documents, metadatas, ids = [], [], []
        
        # Process remaining documents
        if documents:
            self._process_batch(documents, metadatas, ids)
        
        self.logger.info("Vector database ingestion completed")
    
    def _process_batch(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Process a batch of documents"""
        self.logger.info(f"Processing batch of {len(documents)} documents")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(documents)
        
        # Add to vector database
        self.vector_db.add_documents(documents, metadatas, embeddings, ids)
    
    def search_semantic(self, query: str, n_results: int = 10) -> List[Dict]:
        """Perform semantic search on the vector database"""
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, n_results)
        
        self.logger.info(f"Found {len(results)} results for query: {query}")
        return results
    
    def search_with_filters(self, query: str, filters: Dict[str, Any], 
                           n_results: int = 10) -> List[Dict]:
        """Search with additional metadata filters"""
        # For ChromaDB, filtering can be implemented here
        # This is a simplified version - full implementation would depend on specific filtering needs
        results = self.search_semantic(query, n_results * 2)  # Get more results to filter
        
        # Apply filters
        filtered_results = []
        for result in results:
            metadata = result['metadata']
            match = True
            
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, dict):
                        # Range filter
                        if 'min' in value and metadata[key] < value['min']:
                            match = False
                            break
                        if 'max' in value and metadata[key] > value['max']:
                            match = False
                            break
                    else:
                        # Exact match
                        if metadata[key] != value:
                            match = False
                            break
            
            if match:
                filtered_results.append(result)
                if len(filtered_results) >= n_results:
                    break
        
        return filtered_results[:n_results]

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = VectorDBConfig(
        db_type="chroma",
        embedding_model="sentence-transformers",
        model_name="all-MiniLM-L6-v2"
    )
    
    # Example usage
    vector_db = ArgoVectorDB(config)
    
    # Example DataFrame (would come from your data processing)
    sample_data = pd.DataFrame({
        'float_id': ['5904471', '5904472'],
        'date': ['2023-03-15', '2023-03-16'],
        'latitude': [2.5, 3.0],
        'longitude': [65.0, 66.0],
        'depth': [100.0, 200.0],
        'temperature': [28.5, 25.2],
        'salinity': [34.2, 34.8],
        'dissolved_oxygen': [210.5, 180.3]
    })
    
    print("Sample vector database operations:")
    print("1. Ingesting sample data...")
    # vector_db.ingest_dataframe(sample_data)
    
    print("2. Example semantic search...")
    # results = vector_db.search_semantic("temperature data from tropical ocean", n_results=5)
    # for result in results:
    #     print(f"Distance: {result['distance']:.3f}")
    #     print(f"Document: {result['document'][:100]}...")
    #     print("---")