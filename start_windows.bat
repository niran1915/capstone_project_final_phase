@echo off
setlocal
cd /d "%~dp0"
if exist "backend\app\main.py" (
  set "BACKEND_DIR=backend"
) else if exist ".venv\backend\app\main.py" (
  set "BACKEND_DIR=.venv\backend"
) else (
  echo Could not find the backend source folder.
  pause
  exit /b 1
)
if not exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
if not exist "frontend\node_modules\vite\bin\vite.js" (
  echo Frontend packages are not installed. Run setup_windows.bat first.
  pause
  exit /b 1
)
set "HF_HOME=%USERPROFILE%\.cache\huggingface"
start "CrackVision API" /D "%~dp0%BACKEND_DIR%" cmd /k ""%~dp0%BACKEND_DIR%\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "CrackVision UI" /D "%~dp0frontend" cmd /k "npm run dev -- --host 127.0.0.1"
timeout /t 4 /nobreak >nul
start "" http://127.0.0.1:5173
