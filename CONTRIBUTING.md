# Contributing

Thanks for your interest in improving **Swiss Housing & Commute Explorer**!

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements-dev.txt
pip install -e .
pytest
