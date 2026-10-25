# One-click setup for Windows PowerShell.
# Run from the project folder:   .\setup.ps1
# If scripts are blocked:        Set-ExecutionPolicy -Scope Process Bypass

$ErrorActionPreference = "Stop"

Write-Host "==> Creating virtual environment (.venv)" -ForegroundColor Cyan
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip" -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "==> Installing dependencies (this takes a few minutes)" -ForegroundColor Cyan
pip install -r requirements-dev.txt
pip install -e .

Write-Host "==> Downloading NLTK data for METEOR" -ForegroundColor Cyan
python -m nltk.downloader wordnet omw-1.4

if (-not (Test-Path .env)) { Copy-Item .env.example .env }

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run the app with:  streamlit run app/streamlit_app.py"
Write-Host "Run the tests with: pytest"
