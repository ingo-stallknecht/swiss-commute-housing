# Contributing

Thanks for your interest in improving **Swiss Housing & Commute Explorer**!

## Quick start

Clone the repo and set up a local environment:

```bash
git clone https://github.com/ingo-stallknecht/swiss-commute-housing.git
cd swiss-commute-housing

# Create and activate virtualenv
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests (CI checks, formatting, etc.)
pytest
