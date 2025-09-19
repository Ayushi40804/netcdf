"""
RAG (Retrieval Augmented Generation) System for ARGO Oceanographic Data
Combines vector search with SQL query generation for comprehensive data retrieval
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from enum import Enum

# Import our custom modules
from vector_db import ArgoVectorDB, VectorDBConfig
from sql_generator import NaturalLanguageToSQL, SQLGeneratorConfig, LLMProvider

class QueryType(Enum):
    SEMANTIC = "semantic"
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    METADATA = "metadata"

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    # Vector database configuration
    vector_db_config: VectorDBConfig
    
    # SQL generator configuration
    sql_config: SQLGeneratorConfig
    
    # RAG-specific settings
    max_vector_results: int = 20
    max_sql_results: int = 1000
    similarity_threshold: float = 0.7
    hybrid_weight_vector: float = 0.4
    hybrid_weight_sql: float = 0.6
    
    # Query classification
    semantic_keywords: List[str] = None
    analytical_keywords: List[str] = None
    
    def __post_init__(self):
        if self.semantic_keywords is None:
            self.semantic_keywords = [
                "similar", "like", "find", "search", "show me", "what", "describe",
                "characteristics", "properties", "pattern", "trend", "behavior"
            ]
        
        if self.analytical_keywords is None:
            self.analytical_keywords = [
                "average", "mean", "sum", "count", "maximum", "minimum", "total",
                "statistics", "distribution", "correlation", "compare", "difference",
                "group by", "aggregate", "calculate"
            ]

class QueryClassifier:
    """Classifies queries to determine the best retrieval strategy"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def classify_query(self, query: str) -> QueryType:
        """Classify the query type based on content analysis"""
        query_lower = query.lower()
        
        # Count semantic and analytical keywords
        semantic_score = sum(1 for keyword in self.config.semantic_keywords 
                           if keyword in query_lower)
        analytical_score = sum(1 for keyword in self.config.analytical_keywords 
                             if keyword in query_lower)
        
        # Decision logic
        if analytical_score > semantic_score:
            return QueryType.ANALYTICAL
        elif semantic_score > analytical_score:
            return QueryType.SEMANTIC
        else:
            # Check for specific patterns
            if any(pattern in query_lower for pattern in ['avg', 'sum', 'count', 'max', 'min']):
                return QueryType.ANALYTICAL
            elif any(pattern in query_lower for pattern in ['find', 'show', 'what', 'describe']):
                return QueryType.SEMANTIC
            else:
                return QueryType.HYBRID
    
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract potential filters from the query"""
        filters = {}
        query_lower = query.lower()
        
        # Geographic filters
        if "tropical" in query_lower or "equator" in query_lower:
            filters["latitude"] = {"min": -23.5, "max": 23.5}
        elif "northern" in query_lower:
            filters["latitude"] = {"min": 0, "max": 90}
        elif "southern" in query_lower:
            filters["latitude"] = {"min": -90, "max": 0}
        
        # Depth filters
        if "surface" in query_lower or "shallow" in query_lower:
            filters["depth"] = {"min": 0, "max": 100}
        elif "deep" in query_lower:
            filters["depth"] = {"min": 1000, "max": 10000}
        elif "intermediate" in query_lower or "thermocline" in query_lower:
            filters["depth"] = {"min": 100, "max": 1000}
        
        # Variable filters
        if "temperature" in query_lower and "not" not in query_lower:
            filters["has_temperature"] = True
        if "salinity" in query_lower and "not" not in query_lower:
            filters["has_salinity"] = True
        if "oxygen" in query_lower and "not" not in query_lower:
            filters["has_dissolved_oxygen"] = True
        
        return filters

class ResultProcessor:
    """Processes and combines results from different retrieval methods"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_vector_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Process vector search results"""
        processed = {
            "method": "vector_search",
            "total_results": len(results),
            "results": []
        }
        
        for result in results:
            processed_result = {
                "similarity_score": 1 - result.get("distance", 0),
                "content": result.get("document", ""),
                "metadata": result.get("metadata", {}),
                "source_type": "semantic_match"
            }
            processed["results"].append(processed_result)
        
        return processed
    
    def process_sql_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process SQL query results"""
        if not results.get("success", False):
            return {
                "method": "sql_query",
                "error": results.get("error", "Unknown error"),
                "total_results": 0,
                "results": []
            }
        
        data = results.get("data", [])
        
        processed = {
            "method": "sql_query",
            "total_results": len(data),
            "sql_query": results.get("generated_sql", ""),
            "results": []
        }
        
        for row in data:
            processed_result = {
                "data": row,
                "source_type": "analytical_query",
                "relevance_score": 1.0  # SQL results are assumed highly relevant
            }
            processed["results"].append(processed_result)
        
        return processed
    
    def combine_results(self, vector_results: Dict, sql_results: Dict, 
                       query_type: QueryType) -> Dict[str, Any]:
        """Combine results from vector and SQL searches"""
        combined = {
            "query_type": query_type.value,
            "timestamp": datetime.now().isoformat(),
            "vector_results": vector_results,
            "sql_results": sql_results,
            "combined_results": []
        }
        
        # Determine combination strategy based on query type
        if query_type == QueryType.SEMANTIC:
            # Prioritize vector results
            combined["combined_results"] = vector_results.get("results", [])
            # Add relevant SQL results if available
            if sql_results.get("total_results", 0) > 0:
                combined["combined_results"].extend(sql_results["results"][:5])
        
        elif query_type == QueryType.ANALYTICAL:
            # Prioritize SQL results
            combined["combined_results"] = sql_results.get("results", [])
            # Add contextual vector results
            if vector_results.get("total_results", 0) > 0:
                combined["combined_results"].extend(vector_results["results"][:3])
        
        elif query_type == QueryType.HYBRID:
            # Weighted combination
            vector_weight = self.config.hybrid_weight_vector
            sql_weight = self.config.hybrid_weight_sql
            
            # Score and sort results
            all_results = []
            
            for result in vector_results.get("results", []):
                result["combined_score"] = result.get("similarity_score", 0) * vector_weight
                all_results.append(result)
            
            for result in sql_results.get("results", []):
                result["combined_score"] = result.get("relevance_score", 0) * sql_weight
                all_results.append(result)
            
            # Sort by combined score
            all_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            combined["combined_results"] = all_results[:20]
        
        return combined
    
    def generate_summary(self, combined_results: Dict[str, Any], 
                        original_query: str) -> str:
        """Generate a natural language summary of the results"""
        vector_count = combined_results["vector_results"].get("total_results", 0)
        sql_count = combined_results["sql_results"].get("total_results", 0)
        query_type = combined_results["query_type"]
        
        summary_parts = []
        
        # Query interpretation
        summary_parts.append(f"For your query: '{original_query}'")
        summary_parts.append(f"Query type detected: {query_type}")
        
        # Results overview
        if vector_count > 0:
            summary_parts.append(f"Found {vector_count} semantically similar records")
        
        if sql_count > 0:
            sql_query = combined_results["sql_results"].get("sql_query", "")
            if sql_query:
                summary_parts.append(f"Executed analytical query returning {sql_count} records")
        
        # Key insights from results
        if combined_results["combined_results"]:
            total_combined = len(combined_results["combined_results"])
            summary_parts.append(f"Combined {total_combined} most relevant results")
            
            # Analyze data characteristics if SQL results are available
            sql_results = combined_results["sql_results"].get("results", [])
            if sql_results:
                summary_parts.append(self._extract_data_insights(sql_results))
        
        return ". ".join(summary_parts) + "."
    
    def _extract_data_insights(self, sql_results: List[Dict]) -> str:
        """Extract insights from SQL results"""
        if not sql_results:
            return "No analytical data available"
        
        insights = []
        
        # Sample first few results for insight
        sample_size = min(3, len(sql_results))
        sample_data = sql_results[:sample_size]
        
        # Check for common fields and provide insights
        for data in sample_data:
            data_dict = data.get("data", {})
            if "temperature" in data_dict and data_dict["temperature"] is not None:
                insights.append(f"temperature data available (e.g., {data_dict['temperature']:.2f}°C)")
                break
        
        for data in sample_data:
            data_dict = data.get("data", {})
            if "salinity" in data_dict and data_dict["salinity"] is not None:
                insights.append(f"salinity measurements included (e.g., {data_dict['salinity']:.2f} PSU)")
                break
        
        if insights:
            return f"Data includes {', '.join(insights)}"
        else:
            return f"Found {len(sql_results)} data records"

class ArgoRAGSystem:
    """Main RAG system for ARGO oceanographic data"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.vector_db = ArgoVectorDB(config.vector_db_config)
        self.sql_generator = NaturalLanguageToSQL(config.sql_config)
        self.query_classifier = QueryClassifier(config)
        self.result_processor = ResultProcessor(config)
    
    async def query_async(self, query: str) -> Dict[str, Any]:
        """Asynchronous query processing"""
        return await asyncio.get_event_loop().run_in_executor(None, self.query, query)
    
    def query(self, query: str) -> Dict[str, Any]:
        """Main query interface"""
        start_time = datetime.now()
        self.logger.info(f"Processing query: {query}")
        
        try:
            # Classify query
            query_type = self.query_classifier.classify_query(query)
            filters = self.query_classifier.extract_filters(query)
            
            self.logger.info(f"Query classified as: {query_type.value}")
            
            # Execute appropriate retrieval strategies
            vector_results = {}
            sql_results = {}
            
            if query_type in [QueryType.SEMANTIC, QueryType.HYBRID]:
                vector_results = self._execute_vector_search(query, filters)
            
            if query_type in [QueryType.ANALYTICAL, QueryType.HYBRID]:
                sql_results = self._execute_sql_search(query)
            
            # Process and combine results
            processed_vector = self.result_processor.process_vector_results(
                vector_results.get("results", [])
            )
            processed_sql = self.result_processor.process_sql_results(sql_results)
            
            combined_results = self.result_processor.combine_results(
                processed_vector, processed_sql, query_type
            )
            
            # Generate summary
            summary = self.result_processor.generate_summary(combined_results, query)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "query": query,
                "processing_time_seconds": processing_time,
                "summary": summary,
                "results": combined_results,
                "metadata": {
                    "query_type": query_type.value,
                    "filters_applied": filters,
                    "timestamp": start_time.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "timestamp": start_time.isoformat()
            }
    
    def _execute_vector_search(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector search"""
        try:
            if filters:
                results = self.vector_db.search_with_filters(
                    query, filters, n_results=self.config.max_vector_results
                )
            else:
                results = self.vector_db.search_semantic(
                    query, n_results=self.config.max_vector_results
                )
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in results 
                if (1 - r.get("distance", 1)) >= self.config.similarity_threshold
            ]
            
            return {
                "success": True,
                "results": filtered_results,
                "total_before_filtering": len(results),
                "total_after_filtering": len(filtered_results)
            }
            
        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    def _execute_sql_search(self, query: str) -> Dict[str, Any]:
        """Execute SQL search"""
        try:
            return self.sql_generator.query_from_natural_language(query)
        except Exception as e:
            self.logger.error(f"SQL search error: {e}")
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check vector database
        try:
            test_results = self.vector_db.search_semantic("test", n_results=1)
            health["components"]["vector_db"] = {
                "status": "healthy",
                "message": f"Vector search working, {len(test_results)} test results"
            }
        except Exception as e:
            health["components"]["vector_db"] = {
                "status": "error",
                "message": str(e)
            }
        
        # Check SQL generator
        try:
            test_sql = self.sql_generator.generate_sql("test query")
            health["components"]["sql_generator"] = {
                "status": "healthy" if test_sql["success"] else "warning",
                "message": test_sql.get("validation_message", "SQL generation working")
            }
        except Exception as e:
            health["components"]["sql_generator"] = {
                "status": "error",
                "message": str(e)
            }
        
        # Overall health
        component_statuses = [comp["status"] for comp in health["components"].values()]
        if "error" in component_statuses:
            health["overall_status"] = "error"
        elif "warning" in component_statuses:
            health["overall_status"] = "warning"
        else:
            health["overall_status"] = "healthy"
        
        return health

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    vector_config = VectorDBConfig(
        db_type="chroma",
        embedding_model="sentence-transformers",
        model_name="all-MiniLM-L6-v2"
    )
    
    sql_config = SQLGeneratorConfig(
        llm_provider=LLMProvider.OPENAI,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo"
    )
    
    rag_config = RAGConfig(
        vector_db_config=vector_config,
        sql_config=sql_config
    )
    
    # Initialize RAG system
    rag_system = ArgoRAGSystem(rag_config)
    
    # Example queries
    example_queries = [
        "Show me temperature profiles from tropical regions",
        "What's the average salinity at 500 meters depth?",
        "Find data similar to high oxygen measurements",
        "Compare temperature between northern and southern hemispheres",
        "Show salinity distribution in the Indian Ocean"
    ]
    
    print("RAG System Example Queries:")
    for query in example_queries:
        print(f"\\n{'='*50}")
        print(f"Query: {query}")
        print("="*50)
        
        # This would normally execute the full RAG pipeline
        # result = rag_system.query(query)
        # print(f"Query Type: {result['metadata']['query_type']}")
        # print(f"Summary: {result['summary']}")
        # print(f"Processing Time: {result['processing_time_seconds']:.2f}s")
        
        # For demo, just show classification
        query_type = rag_system.query_classifier.classify_query(query)
        filters = rag_system.query_classifier.extract_filters(query)
        print(f"Classified as: {query_type.value}")
        print(f"Filters detected: {filters}")