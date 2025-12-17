@echo off
setlocal enabledelayedexpansion

echo.
echo rigidRL Build Script
echo =====================
echo.

:: Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Install Python 3.10+ from https://python.org
    exit /b 1
)

:: Check pip
where pip >nul 2>nul
if errorlevel 1 (
    echo ERROR: pip not found in PATH
    exit /b 1
)

:: Display versions
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo Using %%i
for /f "tokens=*" %%i in ('pip --version 2^>^&1') do echo Using pip %%i
echo.

:: Install in editable mode
echo Installing rigidRL...
echo.
pip install -e . --quiet
if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    echo.
    echo Common issues:
    echo   - Missing Visual Studio Build Tools
    echo   - Missing Eigen3 headers
    echo   - Missing SDL2 development files
    echo.
    echo For detailed errors, run: pip install -e . -v
    exit /b 1
)

echo.
echo Build successful
echo.

:: Quick verification
echo Verifying installation...
python -c "import rigidRL; print('rigidRL imported successfully')" 2>nul
if errorlevel 1 (
    echo WARNING: Import verification failed
    exit /b 1
)

echo.
echo Ready. Run examples with: python examples/train_drone_sb3.py --test
echo.
