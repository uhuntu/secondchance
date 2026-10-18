@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ================================
echo   seconddraft (温故)
echo   Local   http://localhost:8010
echo   Phone   http://192.168.31.59:8010
echo           (update the IP if your LAN address changes)
echo ================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python was not found on PATH.
    echo Install Python 3 and try again.
    pause
    exit /b 1
)

python -m uvicorn app:app --host 0.0.0.0 --port 8010

echo.
echo Server stopped. Press any key to close this window.
pause >nul
