@echo off
rem =======================================================================
rem  TTS Local Server (GPT-SoVITS V4) - Windows launcher
rem
rem  Layout: this folder = tts-server (service files)
rem          engine\     = official GPT-SoVITS package (MUST be named engine)
rem                        (must contain engine\runtime\python.exe and
rem                         engine\GPT_SoVITS\). Nothing inside engine is
rem                         modified.
rem
rem  First run:  if engine\ is missing, this script will guide you to
rem              download the package by GPU model and ask whether to use
rem              the HTTP proxy (http://127.0.0.1:10809).
rem
rem  Config: edit server_config.json (port / device / default_emotion ...)
rem
rem  Usage:  start.bat                 start (see server_config.json, default 127.0.0.1:9880)
rem          start.bat -p 9881         set port
rem          start.bat --device cpu    force CPU
rem =======================================================================
setlocal EnableExtensions
cd /d "%~dp0"

rem =======================================================================
rem  engine check: guide download if engine\GPT_SoVITS is missing
rem =======================================================================
if exist "engine\GPT_SoVITS" goto engine_ok

echo [TTS] Engine not found: engine\GPT_SoVITS
echo        First run needs the official GPT-SoVITS V4 Windows package (~7GB).
echo.
choice /c YN /m "Download the engine now?"
if errorlevel 2 (
  echo [TTS] Download cancelled. Please download the package, extract it as
  echo       engine\ and run this script again.
  echo       See README.md "Install Engine".
  pause
  exit /b 1
)

rem ---- 1) GPU model (select matching CUDA package) ----
echo.
choice /c 12 /m "GPU model [1] Non-50xx (CUDA 12.4) [2] RTX50xx (CUDA 12.8)"
if errorlevel 2 (set "GPUVER=-nvidia50") else (set "GPUVER=")

rem ---- 2) download source ----
echo.
choice /c 12 /m "Download source [1] HuggingFace [2] ModelScope (CN)"
set "SRC=ms"
if not errorlevel 2 set "SRC=hf"

rem ---- 3) HTTP proxy (optional, custom value) ----
echo.
set "CURLPROXY="
set /p "CURLPROXYURL=HTTP proxy (http://host:port) or press Enter for none: "
if not defined CURLPROXYURL set "CURLPROXYURL="
if defined CURLPROXYURL set "CURLPROXY=-x %CURLPROXYURL%"

set "BASENAME=GPT-SoVITS-v4-20250529%GPUVER%"
if "%SRC%"=="hf" set "URL=https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/%BASENAME%.7z?download=true"
if "%SRC%"=="ms" set "URL=https://www.modelscope.cn/models/FlowerCry/gpt-sovits-7z-pacakges/resolve/master/%BASENAME%.7z"

echo.
echo [TTS] Downloading: %BASENAME%.7z
echo       source: %SRC%    proxy: %CURLPROXY%
curl %CURLPROXY% -L --retry 3 --progress-bar -o "%BASENAME%.7z" "%URL%"
if errorlevel 1 (
  echo [TTS] Download failed. Check your network / proxy and retry.
  pause
  exit /b 1
)

echo [TTS] Downloaded. Trying to extract and rename to engine ...
where 7z >nul 2>&1
if not errorlevel 1 (
  7z x -y "%BASENAME%.7z" >nul
  for /d %%d in (*) do (
    if exist "%%d\GPT_SoVITS" if not exist "engine" move "%%d" "engine" >nul
  )
) else (
  echo [TTS] 7-Zip not found. Please extract "%BASENAME%.7z" manually,
  echo       rename the extracted folder to engine, then run this script again.
  pause
  exit /b 1
)

if exist "engine\GPT_SoVITS" (
  echo [TTS] Engine ready.
  del "%BASENAME%.7z" >nul 2>&1
) else (
  echo [TTS] Auto-extract failed. Please extract "%BASENAME%.7z" manually,
  echo       rename the extracted folder to engine, then run this script again.
  pause
  exit /b 1
)

:engine_ok
rem =======================================================================
rem  locate engine Python runtime
rem =======================================================================
set "PYTHON="
if exist "engine\runtime\python.exe" set "PYTHON=engine\runtime\python.exe"
if not defined PYTHON (
  for /d %%d in (*) do if not defined PYTHON if exist "%%d\runtime\python.exe" set "PYTHON=%%d\runtime\python.exe"
)
if not defined PYTHON set "PYTHON=runtime\python.exe"

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