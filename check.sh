#!/bin/bash
# Quality check script - runs all code quality tools

set -e

echo "Running black..."
uv run black --check backend/ main.py

echo "Running ruff..."
uv run ruff check backend/ main.py

echo "All quality checks passed!"