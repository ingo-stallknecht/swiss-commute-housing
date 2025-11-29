# Contributing

Thank you for your interest in improving this project.

## Setup

    git clone https://github.com/ingo-stallknecht/swiss-commute-housing.git
    cd swiss-commute-housing

Create and activate a virtual environment.

    python -m venv .venv
    source .venv/Scripts/activate      # Windows (Git Bash)
    # source .venv/bin/activate        # macOS or Linux

Install dependencies.

    pip install -r requirements.txt
    pip install -e .

## Code quality

Run formatting and lint checks.

    pre-commit run --all-files

## Tests

    pytest

## Running the application

    streamlit run app/app.py

## Submitting changes

Create a feature branch, make your changes, ensure tests and checks pass, then open a pull request.
