#!/bin/bash

# Smart Advice - Start Script
# This script starts both backend and frontend servers simultaneously

echo "🚀 Starting Smart Advice Application..."
echo "📖 Reading README.md for setup instructions..."

# Check if README.md exists
if [ ! -f "README.md" ]; then
    echo "❌ README.md not found! Please ensure you're in the correct directory."
    exit 1
fi

echo "✅ README.md found - proceeding with startup..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Function to kill processes on script exit
cleanup() {
    echo "🛑 Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "🔴 Backend server stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "🔴 Frontend server stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found!"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found!"
    exit 1
fi

# Start Backend Server
echo "🐍 Starting Backend Server..."
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📥 Installing Python dependencies..."
    pip install -r requirements.txt > /dev/null 2>&1
else
    echo "⚠️  requirements.txt not found, skipping dependency installation"
fi

# Start backend server in background
echo "🚀 Launching backend server on http://localhost:5001..."
python app.py > ../backend.log 2>&1 &
BACKEND_PID=$!

# Return to root directory
cd ..

# Start Frontend Server
echo "⚛️  Starting Frontend Server..."
cd frontend

# Install frontend dependencies
if [ -f "package.json" ]; then
    echo "📥 Installing Node.js dependencies..."
    npm install > /dev/null 2>&1
else
    echo "❌ package.json not found in frontend directory!"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Check if .env.local exists, if not create it
if [ ! -f ".env.local" ]; then
    echo "⚙️  Creating .env.local file..."
    echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:5001" > .env.local
fi

# Start frontend server in background
echo "🚀 Launching frontend server on http://localhost:3000..."
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

# Return to root directory
cd ..

# Wait a moment for servers to start
sleep 3

echo ""
echo "🎉 Smart Advice Application Started Successfully!"
echo ""
echo "📊 Frontend Dashboard: http://localhost:3000"
echo "🔧 Backend API: http://localhost:5001"
echo ""
echo "📝 Logs:"
echo "   Backend: tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "🛑 Press Ctrl+C to stop both servers"
echo ""

# Keep script running and wait for user interrupt
wait
