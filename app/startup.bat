@echo off
color 0A
echo ============================================
echo    Phytoplankton Analysis System Startup
echo ============================================
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo Make sure Python is installed and in PATH
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

:: Install/upgrade requirements
echo [INFO] Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)

:: Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo [INFO] Creating template .env file...
    (
        echo # API Keys
        echo GOOGLE_API_KEY=your_google_api_key_here
        echo LANGSMITH_API_KEY=your_langsmith_api_key_here
        echo COHERE_KEY=your_cohere_key_here
        echo GROQ_KEY=your_groq_key_here
        echo.
        echo # NASA API ^(optional - DEMO_KEY has rate limits^)
        echo NASA_API_KEY=DEMO_KEY
    ) > .env
    echo [WARNING] Please edit .env file with your actual API keys
    echo.
)

:: Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "documents\conferences" mkdir documents\conferences
if not exist "documents\reports" mkdir documents\reports
if not exist "documents\research_papers" mkdir documents\research_papers
if not exist "RAG\__pycache__" mkdir RAG\__pycache__

:: Start the FastAPI backend
echo [INFO] Starting FastAPI backend server...
echo [INFO] Backend will run on http://localhost:8000
start /B python api_server.py

:: Wait for backend to start
echo [INFO] Waiting for backend to initialize...
timeout /t 8 /nobreak > nul

:: Check if backend is running (simple check)
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 5 | Out-Null; exit 0 } catch { exit 1 }" > nul 2>&1
if errorlevel 1 (
    echo [WARNING] Backend health check failed, but continuing anyway...
    echo [INFO] If you see errors, check that port 8000 is available
) else (
    echo [SUCCESS] Backend is running successfully
)

echo.
echo ============================================
echo [INFO] Starting Streamlit frontend...
echo [INFO] The application will open in your browser
echo [INFO] Press Ctrl+C to stop the application
echo ============================================
echo.

:: Start Streamlit frontend
streamlit run chatbot_ui.py

:: Cleanup message
echo.
echo [INFO] Application stopped
echo [INFO] Backend processes may still be running
echo [INFO] Check Task Manager if needed to kill python processes
pause