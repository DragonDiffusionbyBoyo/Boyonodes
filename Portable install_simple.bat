@echo off
rem Minimal installer for custom nodes
setlocal EnableDelayedExpansion

set "CONFIG_FILE=%~dp0python_path.cfg"

if exist "%CONFIG_FILE%" (
    set /p PYTHON_PATH=<"%CONFIG_FILE%"
) else (
    set /p PYTHON_PATH="Python Path: "
    echo !PYTHON_PATH!>"%CONFIG_FILE%"
)

echo Installing packages...
"!PYTHON_PATH!" -m pip install -r requirements.txt
"!PYTHON_PATH!" -m pip install resemble-perth --no-deps
"!PYTHON_PATH!" -m pip install chatterbox-tts --no-deps

echo Done!
pause
