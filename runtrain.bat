@echo off
echo Starting YOLOv8 training...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking if requirements are installed...
pip show ultralytics >nul 2>&1
if %errorlevel% neq 0 (
    echo Required packages are not installed. Running pipinstallreq.bat first...
    call pipinstallreq.bat
)

echo.
echo Running YOLOv8 training script with GPU...
echo.

REM Run the training script with default parameters
python train.py

echo.
echo Training completed!
echo.
pause