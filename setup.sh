#!/bin/bash
set -e
echo "[digital-human-platform] Setting up..."
pip install --upgrade pip
pip install -r requirements.txt
# Pull default LLM
ollama pull mistral 2>/dev/null || echo "Ollama not running — start with: ollama serve"
echo "Setup complete."
