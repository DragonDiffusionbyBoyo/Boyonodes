@echo off
rem Simple installer for custom nodes
setlocal EnableDelayedExpansion

set "CONFIG_FILE=%~dp0python_path.cfg"

rem Check for saved Python path
if exist "%CONFIG_FILE%" (
    set /p PYTHON_PATH=<"%CONFIG_FILE%"
) else (
    echo No saved Python path found.
    echo Please enter the full path to python.exe:
    set /p PYTHON_PATH="Python Path: "
    echo !PYTHON_PATH!>"%CONFIG_FILE%"
)

rem Verify Python exists
if not exist "!PYTHON_PATH!" (
    echo Python executable not found at: !PYTHON_PATH!
    echo Please check the path and try again.
    del "%CONFIG_FILE%" 2>nul
    pause
    exit /b 1
)

echo ===================================
echo    Installing Custom Nodes
echo ===================================
echo Using Python: !PYTHON_PATH!
echo.

rem Check if requirements.txt exists
if not exist "requirements.txt" (
    echo WARNING: requirements.txt not found in current directory
    echo Continuing with package installations...
    echo.
) else (
    echo Installing requirements from requirements.txt...
    "!PYTHON_PATH!" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements.txt
        pause
        exit /b 1
    )
    echo Requirements installed successfully.
    echo.
)

echo Installing resemble-perth (no dependencies)...
"!PYTHON_PATH!" -m pip install resemble-perth --no-deps
if errorlevel 1 (
    echo ERROR: Failed to install resemble-perth
    pause
    exit /b 1
)
echo resemble-perth installed successfully.
echo.

echo Installing chatterbox-tts (no dependencies)...
"!PYTHON_PATH!" -m pip install chatterbox-tts --no-deps
if errorlevel 1 (
    echo ERROR: Failed to install chatterbox-tts
    pause
    exit /b 1
)
echo chatterbox-tts installed successfully.
echo.

echo ===================================
echo    Installation Complete!
echo ===================================
pause
