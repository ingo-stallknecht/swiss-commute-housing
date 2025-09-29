#!/usr/bin/env python
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
APP = os.path.join(ROOT, "app", "app.py")

# Ensure src/ is importable (if app imports your package later)
sys.path.insert(0, os.path.join(ROOT, "src"))

subprocess.run([sys.executable, "-m", "streamlit", "run", APP])
