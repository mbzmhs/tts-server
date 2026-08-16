@echo off
rem =======================================================================
rem  TTS Local Server (GPT-SoVITS V4) - Windows launcher
rem
rem  Layout: this folder = tts-server (service files)
rem          engine\     = official GPT-SoVITS package extracted here
rem                        (must contain engine\runtime\python.exe and
rem                         engine\GPT_SoVITS\). Nothing inside engine is
rem                         modified.
rem
rem  Config: edit server_config.json (port / device / default_emotion ...)
rem
rem  Usage:  start.bat                 start (see server_config.json, default 127.0.0.1:9880)
rem          start.bat -p 9881         set port
rem          start.bat --device cpu    force CPU
rem =======================================================================
setlocal
cd /d "%~dp0"

rem locate the Python runtime inside the engine folder (any folder name)
set "PYTHON="
if exist "engine\runtime\python.exe" set "PYTHON=engine\runtime\python.exe"
if not defined PYTHON (
  for /d %%d in (*) do if not defined PYTHON if exist "%%d\runtime\python.exe" set "PYTHON=%%d\runtime\python.exe"
)
if not defined PYTHON set "PYTHON=runtime\python.exe"

if not exist "engine\GPT_SoVITS" (
  echo [TTS] ERROR: engine\GPT_SoVITS not found.
  echo        Put the GPT-SoVITS official package into the engine\ folder.
  echo        See the guide file in this folder for instructions.
  pause
  exit /b 1
)

if not exist "%PYTHON%" (
  echo [TTS] ERROR: Python runtime not found: %PYTHON%
  echo        Make sure the engine\ folder contains engine\runtime\python.exe
  pause
  exit /b 1
)

echo =======================================================================
echo  TTS Local Server (GPT-SoVITS V4)
echo  engine folder: engine
echo  command     : %PYTHON% tts_server.py %*
echo =======================================================================

"%PYTHON%" tts_server.py %*

echo.
echo Server exited.
pause
endlocal