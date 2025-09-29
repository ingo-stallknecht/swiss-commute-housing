@echo off
SETLOCAL

REM === Paths ===
SET VENV=.venv
SET PY=%VENV%\Scripts\python.exe
SET PIP=%VENV%\Scripts\pip.exe

IF NOT EXIST %VENV% (
    echo [1/5] Creating virtual environment...
    python -m venv %VENV%
)

echo [2/5] Upgrading pip...
%PY% -m pip install --upgrade pip

echo [3/5] Installing requirements...
%PIP% install -r requirements.txt
%PIP% install -r requirements-dev.txt
%PY% -m pip install -e .

echo [4/5] Building artifacts...
%PY% scripts\make_artifacts.py

echo [5/5] Launching Streamlit app...
%PY% scripts\launch_app.py

ENDLOCAL
pause
