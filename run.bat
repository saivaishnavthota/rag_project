@echo off
echo =========================================
echo Local RAG Server with Qwen 2.5 7B
echo =========================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Run the server
echo Starting server on http://0.0.0.0:8000
python main.py
