@echo off
setlocal
cd /d "%~dp0"
if exist "backend\app\main.py" (
  set "BACKEND_DIR=backend"
) else if exist ".venv\backend\app\main.py" (
  set "BACKEND_DIR=.venv\backend"
) else (
  echo ERROR: Could not find the backend source folder.
  pause
  exit /b 1
)
echo ========================================================
echo CrackVision setup - Python backend and React frontend
echo ========================================================
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python launcher ^(py^) was not found. Install Python 3.11 x64 and enable PATH.
  pause
  exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
  echo ERROR: Node.js was not found. Install Node.js 20 or later and reopen this window.
  pause
  exit /b 1
)
if not exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
  echo [1/4] Creating Python virtual environment...
  py -3.11 -m venv "%BACKEND_DIR%\.venv"
  if errorlevel 1 (
    echo Python 3.11 could not be created. Confirm Python 3.11 x64 is installed.
    pause
    exit /b 1
  )
) else (
  echo [1/4] Python virtual environment already exists.
)
echo [2/4] Installing backend dependencies. This downloads PyTorch and Ultralytics and may take several minutes...
"%BACKEND_DIR%\.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%BACKEND_DIR%\.venv\Scripts\python.exe" -m pip install -r "%BACKEND_DIR%\requirements.txt"
if errorlevel 1 goto :failed
echo [3/4] Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 goto :failed
cd ..
echo [4/4] Setup complete.
echo Run start_windows.bat to launch the app.
pause
exit /b 0
:failed
echo.
echo Setup stopped because a dependency could not be installed. Check the error above and README.md.
pause
exit /b 1
