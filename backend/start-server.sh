#!/bin/bash

echo "🚀 Starting Stock Advisor Backend Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import flask" 2>/dev/null || {
    echo "❌ Flask not found. Installing dependencies..."
    pip install -r requirements.txt
}

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# Start the Flask server
echo "🌟 Starting Flask server on http://127.0.0.1:5001"
echo "   Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
flask run --host=127.0.0.1 --port=5001
