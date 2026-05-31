#!/bin/bash
# Format script - formats code with black and fixes lint issues with ruff

echo "Formatting with black..."
uv run black backend/ main.py

echo "Fixing lint issues with ruff..."
uv run ruff check --fix backend/ main.py

echo "Formatting complete!"