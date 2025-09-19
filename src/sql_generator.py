"""
Natural Language to SQL Query Generator for ARGO Oceanographic Data
Translates natural language questions into SQL queries using LLM capabilities
"""

import os
import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import mysql.connector
from enum import Enum

# LLM imports with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"

@dataclass
class SQLGeneratorConfig:
    """Configuration for SQL query generation"""
    llm_provider: LLMProvider = LLMProvider.GOOGLE
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"  # Default to Gemini
    max_tokens: int = 1000
    temperature: float = 0.1
    db_schema_file: str = "argo_schema.json"
    
    # Database connection parameters
    db_host: str = "localhost"
    db_user: str = "root"
    db_password: str = "Arman123?"
    db_name: str = "argo_data"

class ArgoSchemaManager:
    """Manages database schema information for query generation"""
    
    def __init__(self, config: SQLGeneratorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.schema_info = self._load_schema_info()
    
    def _load_schema_info(self) -> Dict[str, Any]:
        """Load or generate schema information"""
        if os.path.exists(self.config.db_schema_file):
            with open(self.config.db_schema_file, 'r') as f:
                return json.load(f)
        else:
            return self._generate_schema_info()
    
    def _generate_schema_info(self) -> Dict[str, Any]:
        """Generate schema information from database"""
        schema = {
            "tables": {
                "argo_profiles": {
                    "description": "Main table containing ARGO oceanographic profile data",
                    "columns": {
                        "id": {"type": "BIGINT", "description": "Primary key, auto-increment"},
                        "float_id": {"type": "VARCHAR(50)", "description": "ARGO float identifier"},
                        "date": {"type": "DATETIME", "description": "Date and time of measurement"},
                        "latitude": {"type": "DOUBLE", "description": "Latitude in decimal degrees (-90 to 90)"},
                        "longitude": {"type": "DOUBLE", "description": "Longitude in decimal degrees (-180 to 180)"},
                        "depth": {"type": "DOUBLE", "description": "Depth in meters (positive values)"},
                        "temperature": {"type": "DOUBLE", "description": "Water temperature in Celsius"},
                        "salinity": {"type": "DOUBLE", "description": "Salinity in PSU (Practical Salinity Units)"},
                        "dissolved_oxygen": {"type": "DOUBLE", "description": "Dissolved oxygen in μmol/kg"},
                        "profile_index": {"type": "INT", "description": "Index of profile within float data"},
                        "level_index": {"type": "INT", "description": "Index of level within profile"},
                        "created_at": {"type": "TIMESTAMP", "description": "Record creation timestamp"},
                        "updated_at": {"type": "TIMESTAMP", "description": "Record update timestamp"}
                    },
                    "indexes": [
                        {"name": "idx_date", "columns": ["date"]},
                        {"name": "idx_latlon", "columns": ["latitude", "longitude"]},
                        {"name": "idx_float", "columns": ["float_id"]},
                        {"name": "idx_depth", "columns": ["depth"]},
                        {"name": "idx_location_date", "columns": ["latitude", "longitude", "date"]},
                        {"name": "idx_float_date", "columns": ["float_id", "date"]}
                    ]
                },
                "processing_metadata": {
                    "description": "Metadata about data processing runs",
                    "columns": {
                        "id": {"type": "BIGINT", "description": "Primary key"},
                        "run_id": {"type": "VARCHAR(100)", "description": "Unique identifier for processing run"},
                        "start_time": {"type": "DATETIME", "description": "Processing start time"},
                        "end_time": {"type": "DATETIME", "description": "Processing end time"},
                        "status": {"type": "ENUM", "description": "Status: running, completed, failed"},
                        "profiles_processed": {"type": "INT", "description": "Number of profiles processed"},
                        "files_processed": {"type": "INT", "description": "Number of files processed"},
                        "config_used": {"type": "JSON", "description": "Configuration used for processing"},
                        "error_message": {"type": "TEXT", "description": "Error message if processing failed"}
                    }
                }
            },
            "domain_knowledge": {
                "oceanographic_terms": {
                    "temperature": ["temp", "water temperature", "sea temperature", "ocean temperature"],
                    "salinity": ["salt", "saltiness", "salt content", "psu"],
                    "dissolved_oxygen": ["oxygen", "o2", "dissolved o2", "oxygen content"],
                    "depth": ["pressure", "water depth", "ocean depth", "meters deep"],
                    "location": ["position", "coordinates", "lat", "lon", "latitude", "longitude"],
                    "time": ["date", "time", "when", "period", "temporal"]
                },
                "common_queries": [
                    "temperature profiles",
                    "salinity distribution",
                    "oxygen levels",
                    "seasonal variation",
                    "regional differences",
                    "depth profiles",
                    "time series"
                ],
                "geographic_regions": {
                    "indian_ocean": {"lat_range": [-40, 30], "lon_range": [20, 120]},
                    "tropical": {"lat_range": [-23.5, 23.5]},
                    "equatorial": {"lat_range": [-5, 5]},
                    "northern_hemisphere": {"lat_range": [0, 90]},
                    "southern_hemisphere": {"lat_range": [-90, 0]}
                }
            }
        }
        
        # Save schema for future use
        with open(self.config.db_schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        return schema
    
    def get_schema_context(self) -> str:
        """Get schema information as context for LLM"""
        context = "Database Schema for ARGO Oceanographic Data:\\n\\n"
        
        for table_name, table_info in self.schema_info["tables"].items():
            context += f"Table: {table_name}\\n"
            context += f"Description: {table_info['description']}\\n"
            context += "Columns:\\n"
            
            for col_name, col_info in table_info["columns"].items():
                context += f"  - {col_name} ({col_info['type']}): {col_info['description']}\\n"
            
            context += "\\n"
        
        # Add domain knowledge
        context += "Domain Knowledge:\\n"
        context += "Oceanographic terms and their synonyms:\\n"
        for term, synonyms in self.schema_info["domain_knowledge"]["oceanographic_terms"].items():
            context += f"  - {term}: {', '.join(synonyms)}\\n"
        
        return context

class SQLQueryValidator:
    """Validates and sanitizes SQL queries"""
    
    def __init__(self, config: SQLGeneratorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Allowed SQL keywords and patterns
        self.allowed_keywords = {
            'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 
            'LIMIT', 'OFFSET', 'AS', 'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 
            'LIKE', 'IS', 'NULL', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
            'DISTINCT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN'
        }
        
        # Dangerous keywords to block
        self.dangerous_keywords = {
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'
        }
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate SQL query for safety and correctness"""
        query_upper = query.upper()
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in query_upper:
                return False, f"Dangerous keyword '{keyword}' not allowed"
        
        # Must start with SELECT
        if not query_upper.strip().startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        # Basic syntax validation
        if query.count('(') != query.count(')'):
            return False, "Unmatched parentheses"
        
        # Check for table names (should only query argo_profiles and processing_metadata)
        allowed_tables = {'argo_profiles', 'processing_metadata'}
        
        # Simple table name extraction (could be improved with proper SQL parsing)
        from_match = re.search(r'FROM\\s+([\\w_]+)', query_upper)
        if from_match:
            table_name = from_match.group(1).lower()
            if table_name not in allowed_tables:
                return False, f"Table '{table_name}' not allowed"
        
        return True, "Query is valid"
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize query by removing potentially harmful content"""
        # Remove comments
        query = re.sub(r'--.*?\\n', '', query)
        query = re.sub(r'/\\*.*?\\*/', '', query, flags=re.DOTALL)
        
        # Limit result set size for safety
        if 'LIMIT' not in query.upper():
            query += ' LIMIT 1000'
        
        return query.strip()

class NaturalLanguageToSQL:
    """Main class for converting natural language to SQL queries"""
    
    def __init__(self, config: SQLGeneratorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.schema_manager = ArgoSchemaManager(config)
        self.validator = SQLQueryValidator(config)
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM client"""
        if self.config.llm_provider == LLMProvider.OPENAI:
            if not OPENAI_AVAILABLE or not self.config.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI provider")
            openai.api_key = self.config.openai_api_key
            
        elif self.config.llm_provider == LLMProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE or not self.config.anthropic_api_key:
                raise ValueError("Anthropic API key required for Anthropic provider")
            self.anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            
        elif self.config.llm_provider == LLMProvider.GOOGLE:
            if not GOOGLE_AVAILABLE or not self.config.google_api_key:
                raise ValueError("Google API key required for Google provider")
            genai.configure(api_key=self.config.google_api_key)
            self.google_model = genai.GenerativeModel(self.config.model_name)
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for LLM"""
        schema_context = self.schema_manager.get_schema_context()
        
        prompt = f"""You are an expert SQL query generator for oceanographic data analysis. 
Your task is to convert natural language questions into valid SQL queries for an ARGO oceanographic database.

{schema_context}

Guidelines:
1. Only generate SELECT queries - no INSERT, UPDATE, DELETE, DROP, etc.
2. Always include appropriate WHERE clauses for filtering
3. Use proper SQL syntax and best practices
4. Consider geographic and temporal constraints in the data
5. Use appropriate aggregations when asked for summaries
6. Include LIMIT clauses to prevent excessive results
7. Use proper JOIN syntax when querying multiple tables
8. Consider oceanographic domain knowledge when interpreting terms

Example queries:
- "Show temperature profiles from March 2023" → SELECT * FROM argo_profiles WHERE date BETWEEN '2023-03-01' AND '2023-03-31' AND temperature IS NOT NULL
- "Average salinity by depth" → SELECT depth, AVG(salinity) as avg_salinity FROM argo_profiles WHERE salinity IS NOT NULL GROUP BY depth ORDER BY depth
- "Data from tropical regions" → SELECT * FROM argo_profiles WHERE latitude BETWEEN -23.5 AND 23.5

Respond with only the SQL query, no explanations unless specifically asked."""
        
        return prompt
    
    def _call_llm(self, prompt: str, user_query: str) -> str:
        """Call the LLM with the given prompt"""
        if self.config.llm_provider == LLMProvider.OPENAI:
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content.strip()
            
        elif self.config.llm_provider == LLMProvider.ANTHROPIC:
            response = self.anthropic_client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=prompt,
                messages=[{"role": "user", "content": user_query}]
            )
            return response.content[0].text.strip()
            
        elif self.config.llm_provider == LLMProvider.GOOGLE:
            full_prompt = f"{prompt}\n\nUser Query: {user_query}"
            response = self.google_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            )
            return response.text.strip()
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    def generate_sql(self, natural_language_query: str) -> Dict[str, Any]:
        """Generate SQL query from natural language"""
        try:
            system_prompt = self._create_system_prompt()
            
            # Enhance user query with context
            enhanced_query = f"""
            Convert this natural language question to a SQL query:
            "{natural_language_query}"
            
            Make sure to:
            - Consider the oceanographic context
            - Use appropriate table joins if needed
            - Include reasonable LIMIT clauses
            - Handle NULL values appropriately
            """
            
            # Call LLM
            sql_query = self._call_llm(system_prompt, enhanced_query)
            
            # Clean up the response
            sql_query = self._extract_sql_from_response(sql_query)
            
            # Validate query
            is_valid, validation_message = self.validator.validate_query(sql_query)
            
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Query validation failed: {validation_message}",
                    "sql_query": sql_query,
                    "natural_language_query": natural_language_query
                }
            
            # Sanitize query
            sanitized_query = self.validator.sanitize_query(sql_query)
            
            return {
                "success": True,
                "sql_query": sanitized_query,
                "original_sql": sql_query,
                "natural_language_query": natural_language_query,
                "validation_message": validation_message
            }
            
        except Exception as e:
            self.logger.error(f"Error generating SQL query: {e}")
            return {
                "success": False,
                "error": str(e),
                "natural_language_query": natural_language_query
            }
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response"""
        # Remove markdown code blocks
        response = re.sub(r'```sql\\n?', '', response)
        response = re.sub(r'```\\n?', '', response)
        
        # Find the SQL query (usually starts with SELECT)
        lines = response.split('\\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT') or sql_lines:
                sql_lines.append(line)
                if line.endswith(';'):
                    break
        
        if sql_lines:
            return ' '.join(sql_lines).rstrip(';')
        else:
            return response.strip().rstrip(';')
    
    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query against the database"""
        try:
            conn = mysql.connector.connect(
                host=self.config.db_host,
                user=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name
            )
            
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            return {
                "success": True,
                "data": results,
                "row_count": len(results),
                "sql_query": sql_query
            }
            
        except mysql.connector.Error as e:
            self.logger.error(f"Database error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql_query": sql_query
            }
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()
    
    def query_from_natural_language(self, natural_language_query: str) -> Dict[str, Any]:
        """Complete pipeline: natural language → SQL → results"""
        # Generate SQL
        sql_result = self.generate_sql(natural_language_query)
        
        if not sql_result["success"]:
            return sql_result
        
        # Execute SQL
        execution_result = self.execute_query(sql_result["sql_query"])
        
        # Combine results
        return {
            **execution_result,
            "natural_language_query": natural_language_query,
            "generated_sql": sql_result["sql_query"]
        }

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration (you'll need to set your API key)
    config = SQLGeneratorConfig(
        llm_provider=LLMProvider.GOOGLE,
        google_api_key=os.getenv("GOOGLE_API_KEY"),  # Set your API key
        model_name="gemini-1.5-flash"
    )
    
    # Initialize SQL generator
    sql_generator = NaturalLanguageToSQL(config)
    
    # Example queries
    example_queries = [
        "Show all temperature measurements from March 2023",
        "What's the average salinity at different depths?",
        "Find data from tropical regions with high oxygen levels",
        "Show the temperature profile for float 5904471",
        "What's the temperature variation over time at the equator?"
    ]
    
    print("Example SQL generation:")
    for query in example_queries:
        print(f"\\nNatural Language: {query}")
        result = sql_generator.generate_sql(query)
        if result["success"]:
            print(f"Generated SQL: {result['sql_query']}")
        else:
            print(f"Error: {result['error']}")