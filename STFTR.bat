@echo off
REM Point OpenUtau Tool2 (Resampler) to this .bat

set "PYTHON_EXE=V:\exp\utau-ml\Scripts\python.exe"
set "SCRIPT_DIR=%~dp0"

"%PYTHON_EXE%" "%~dp0STFTR.py" %*
exit /b %ERRORLEVEL%
