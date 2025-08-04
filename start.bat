@echo off
setlocal enabledelayedexpansion

:: Smart Advice - Start Script for Windows
:: This script starts both backend and frontend servers simultaneously

echo 🚀 Starting Smart Advice Application...
echo 📖 Reading README.md for setup instructions...

:: Check if README.md exists
if not exist "README.md" (
    echo ❌ README.md not found! Please ensure you're in the correct directory.
    pause
    exit /b 1
)

echo ✅ README.md found - proceeding with startup...

:: Check prerequisites
echo 🔍 Checking prerequisites...

where python >nul 2>nul || where python3 >nul 2>nul
if errorlevel 1 (
    echo ❌ Python 3 is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 18+ first.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo ❌ npm is not installed. Please install npm first.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed

:: Check if backend directory exists
if not exist "backend" (
    echo ❌ Backend directory not found!
    pause
    exit /b 1
)

:: Check if frontend directory exists
if not exist "frontend" (
    echo ❌ Frontend directory not found!
    pause
    exit /b 1
)

:: Start Backend Server
echo 🐍 Starting Backend Server...
cd backend

:: Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv || python3 -m venv venv
)

:: Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

:: Install backend dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo 📥 Installing Python dependencies...
    pip install -r requirements.txt >nul 2>&1
) else (
    echo ⚠️  requirements.txt not found, skipping dependency installation
)

:: Start backend server in background
echo 🚀 Launching backend server on http://localhost:5001...
start /B python app.py > ..\backend.log 2>&1

:: Return to root directory
cd ..

:: Start Frontend Server
echo ⚛️  Starting Frontend Server...
cd frontend

:: Install frontend dependencies
if exist "package.json" (
    echo 📥 Installing Node.js dependencies...
    npm install >nul 2>&1
) else (
    echo ❌ package.json not found in frontend directory!
    pause
    exit /b 1
)

:: Check if .env.local exists, if not create it
if not exist ".env.local" (
    echo ⚙️  Creating .env.local file...
    echo NEXT_PUBLIC_API_URL=http://127.0.0.1:5001 > .env.local
)

:: Start frontend server in background
echo 🚀 Launching frontend server on http://localhost:3000...
start /B npm run dev > ..\frontend.log 2>&1

:: Return to root directory
cd ..

:: Wait a moment for servers to start
timeout /t 3 /nobreak >nul

echo.
echo 🎉 Smart Advice Application Started Successfully!
echo.
echo 📊 Frontend Dashboard: http://localhost:3000
echo 🔧 Backend API: http://localhost:5001
echo.
echo 📝 Logs:
echo    Backend: type backend.log
echo    Frontend: type frontend.log
echo.
echo 🛑 Press Ctrl+C to stop or close this window
echo.

:: Keep command prompt open
pause
