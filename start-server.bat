@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ================================
echo   Math Mistake Notebook
echo   http://localhost:8010
echo ================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python was not found on PATH.
    echo Install Python 3 and try again.
    pause
    exit /b 1
)

python -m uvicorn app:app --port 8010

echo.
echo Server stopped. Press any key to close this window.
pause >nul
