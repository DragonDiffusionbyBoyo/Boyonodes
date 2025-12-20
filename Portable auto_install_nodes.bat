@echo off
rem Auto-installing batch file for ComfyUI custom nodes
setlocal EnableDelayedExpansion

echo ===================================
echo    Installing Custom Nodes
echo ===================================

rem Auto-detect Python in ComfyUI portable installation
set "PYTHON_PATH="

rem Method 1: Look for python_embeded (most common ComfyUI portable location)
if exist "..\..\..\python_embeded\python.exe" (
    set "PYTHON_PATH=..\..\..\python_embeded\python.exe"
    echo Found Python: python_embeded
)

rem Method 2: Look for python folder
if "!PYTHON_PATH!"=="" if exist "..\..\..\python\python.exe" (
    set "PYTHON_PATH=..\..\..\python\python.exe"
    echo Found Python: python folder
)

rem Method 3: Look for embedded python with alternative spelling
if "!PYTHON_PATH!"=="" if exist "..\..\..\python_embedded\python.exe" (
    set "PYTHON_PATH=..\..\..\python_embedded\python.exe"
    echo Found Python: python_embedded
)

rem Method 4: Look two levels up (different folder structure)
if "!PYTHON_PATH!"=="" if exist "..\..\python_embeded\python.exe" (
    set "PYTHON_PATH=..\..\python_embeded\python.exe"
    echo Found Python: ../python_embeded
)

rem If still not found, try to find it anywhere starting from ComfyUI root
if "!PYTHON_PATH!"=="" (
    echo Searching for Python executable...
    for /f "delims=" %%i in ('dir "..\..\..\python.exe" /s /b 2^>nul') do (
        set "PYTHON_PATH=%%i"
        echo Found Python: %%i
        goto found_python
    )
)

:found_python
if "!PYTHON_PATH!"=="" (
    echo ERROR: Could not find Python executable in ComfyUI installation
    echo This script should be run from a ComfyUI custom_nodes folder
    echo Please ensure you have a portable ComfyUI installation
    pause
    exit /b 1
)

echo Using Python: !PYTHON_PATH!
echo.

rem Install requirements if file exists
if exist "requirements.txt" (
    echo Installing requirements from requirements.txt...
    "!PYTHON_PATH!" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo WARNING: Some requirements may have failed to install
    ) else (
        echo Requirements installed successfully.
    )
    echo.
)

echo Installing resemble-perth (no dependencies)...
"!PYTHON_PATH!" -m pip install resemble-perth --no-deps
if errorlevel 1 (
    echo WARNING: Failed to install resemble-perth
) else (
    echo resemble-perth installed successfully.
)
echo.

echo Installing chatterbox-tts (no dependencies)...
"!PYTHON_PATH!" -m pip install chatterbox-tts --no-deps
if errorlevel 1 (
    echo WARNING: Failed to install chatterbox-tts
) else (
    echo chatterbox-tts installed successfully.
)
echo.

echo ===================================
echo    Installation Complete!
echo ===================================
echo You can now close this window.
pause
