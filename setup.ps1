# AI Chatbot Backend Setup Script for Windows
# Run this script in PowerShell to set up your environment

Write-Host "🎯 AI Chatbot Backend Setup for Windows" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "🐍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Check if pip is installed
Write-Host "📦 Checking pip installation..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ Found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ pip not found. Please ensure pip is installed with Python" -ForegroundColor Red
    exit 1
}

# Create virtual environment (optional but recommended)
Write-Host "🏗️ Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "📥 Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Create .env file
Write-Host "📝 Setting up environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
} else {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ .env file created from template" -ForegroundColor Green
}

# Create directories
Write-Host "📁 Creating necessary directories..." -ForegroundColor Yellow
if (!(Test-Path "documents")) { New-Item -ItemType Directory -Path "documents" }
if (!(Test-Path "vector_store")) { New-Item -ItemType Directory -Path "vector_store" }
Write-Host "✅ Directories created" -ForegroundColor Green

# Test imports
Write-Host "🧪 Testing installation..." -ForegroundColor Yellow
$testScript = @"
try:
    import fastapi
    import uvicorn
    import transformers
    import langchain
    import sentence_transformers
    print("✅ All packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
"@

$testResult = python -c $testScript
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Installation test passed" -ForegroundColor Green
} else {
    Write-Host "❌ Installation test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Add your resume and documents to the 'documents/' folder" -ForegroundColor White
Write-Host "2. Update the sample information in documents/sample_resume.txt" -ForegroundColor White
Write-Host "3. Run the backend:" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor Gray
Write-Host "4. Test the API at: http://localhost:8000/docs" -ForegroundColor White
Write-Host "5. Use the frontend example in frontend_example.md" -ForegroundColor White
Write-Host ""
Write-Host "📚 Check README.md for more detailed instructions." -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the virtual environment in the future, run:" -ForegroundColor Yellow
Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Gray
