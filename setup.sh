#!/usr/bin/env bash
# One-click setup for Linux / macOS / Git Bash.   Usage: bash setup.sh
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU-only torch on Linux keeps the download small (skip this line if you have a GPU).
if [[ "$(uname)" == "Linux" ]]; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements-dev.txt
pip install -e .
python -m nltk.downloader wordnet omw-1.4
[ -f .env ] || cp .env.example .env

echo
echo "Setup complete."
echo "Run the app with:   streamlit run app/streamlit_app.py"
echo "Run the tests with: pytest"
