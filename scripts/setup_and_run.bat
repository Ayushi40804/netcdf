@echo off
REM ARGO Oceanographic Data System - Windows Setup and Launch Script

echo ==========================================================
echo 🌊 ARGO Oceanographic Data System Setup
echo ==========================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo ✅ Python found

REM Install dependencies
echo.
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed

REM Setup environment files
echo.
echo 🔧 Setting up configuration...

if not exist .env (
    if exist .env.template (
        copy .env.template .env
        echo ✅ Created .env file from template
        echo ⚠️ Please edit .env file with your API keys
    ) else (
        echo ❌ .env.template not found
    )
)

if not exist .streamlit\secrets.toml (
    if exist .streamlit\secrets.toml.template (
        copy .streamlit\secrets.toml.template .streamlit\secrets.toml
        echo ✅ Created Streamlit secrets.toml from template
        echo ⚠️ Please edit .streamlit\secrets.toml with your credentials
    ) else (
        echo ❌ Streamlit secrets template not found
    )
)

echo.
echo ==========================================================
echo 🚀 Setup Complete! Now starting the system...
echo ==========================================================

REM Run the system
python run_system.py

pause