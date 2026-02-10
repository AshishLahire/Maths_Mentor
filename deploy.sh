#!/bin/bash

# Deployment script for Math Mentor AI
# Bhai, ye script deploy karne mein help karega

echo "🚀 Math Mentor AI Deployment Script"
echo "===================================="

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for Tesseract (OCR)
echo "Checking for Tesseract..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract found"
else
    echo "⚠️ Tesseract not found. Installing..."
    # For Ubuntu/Debian
    # sudo apt-get install tesseract-ocr
    # For macOS
    # brew install tesseract
    echo "Please install Tesseract manually:"
    echo "Ubuntu: sudo apt-get install tesseract-ocr"
    echo "macOS: brew install tesseract"
    echo "Windows: Download from GitHub"
fi

# Initialize knowledge base
echo "Initializing knowledge base..."
python3 -c "
from rag.vector_store import initialize_knowledge_base
initialize_knowledge_base()
print('✅ Knowledge base initialized')
"

# Test the system
echo "Running tests..."
python3 test_app.py

if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    
    echo ""
    echo "🎉 Setup Complete!"
    echo ""
    echo "To run the app locally:"
    echo "  streamlit run app.py"
    echo ""
    echo "For deployment to Streamlit Cloud:"
    echo "1. Push to GitHub"
    echo "2. Go to https://share.streamlit.io"
    echo "3. Connect your repository"
    echo "4. Set main file to app.py"
    echo "5. Deploy!"
else
    echo "❌ Tests failed. Please fix issues before deployment."
    exit 1
fi
