#!/bin/bash
# Face Authentication System - Run Script

echo "Starting Face Authentication System..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if dependencies are installed
python -c "import insightface" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the API server
echo "Starting API server on http://localhost:8000"
python api/main.py
