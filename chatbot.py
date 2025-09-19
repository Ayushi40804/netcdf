import os
import streamlit as st
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# ------------------------------
# DATABASE & LLM SETUP
# ------------------------------
GOOGLE_API_KEY = st.secrets["api_keys"]["google_api_key"]

# Use SQLite for simplicity (no server required)
DB_PATH = "argo_data.sqlite"

# ------------------------------
# DATABASE CONNECTION
# ------------------------------
try:
    # Create SQLite database connection
    engine = create_engine(f"sqlite:///{DB_PATH}")
    db = SQLDatabase(engine)
    
    # Create a sample table if it doesn't exist
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS argo_profiles (
                profile_id INTEGER PRIMARY KEY,
                float_id TEXT,
                latitude REAL,
                longitude REAL,
                date_time TEXT,
                temperature REAL,
                salinity REAL,
                pressure REAL,
                depth REAL
            )
        """))
        
        # Insert sample data if table is empty
        result = conn.execute(text("SELECT COUNT(*) FROM argo_profiles"))
        count = result.scalar()
        
        if count == 0:
            conn.execute(text("""
                INSERT INTO argo_profiles (float_id, latitude, longitude, date_time, temperature, salinity, pressure, depth)
                VALUES 
                ('5904297', 20.5, 68.5, '2023-01-15', 25.2, 35.1, 10.5, 5.0),
                ('5904297', 20.5, 68.5, '2023-01-15', 24.8, 35.2, 50.2, 25.0),
                ('5904298', 21.0, 69.0, '2023-01-16', 26.1, 34.9, 15.3, 8.0),
                ('5904299', 19.8, 67.2, '2023-01-17', 23.5, 35.4, 80.1, 40.0)
            """))
            conn.commit()
            
except Exception as e:
    st.error(f"Failed to connect to the database. Error: {e}")
    st.stop()

# ------------------------------
# LANGCHAIN LLM + SQL AGENT
# ------------------------------
try:
    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)


    # Create the SQL Agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
except Exception as e:
    st.error(f"Failed to initialize the Language Model. Is your Google API Key correct in secrets.toml? Error: {e}")
    st.stop()

# ------------------------------
# STREAMLIT APP
# ------------------------------
st.title("🌊 FloatChat - ARGO Data Assistant")

user_query = st.text_input("Ask me about ARGO data (e.g., 'What is the average temperature for float 5904297?')")

if user_query:
    with st.spinner('Thinking...'):
        try:
            response = agent_executor.invoke({"input": user_query})
            answer = response.get("output", "Sorry, I couldn't find an answer.")
            st.write("**Answer:**")
            st.markdown(answer)
        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
