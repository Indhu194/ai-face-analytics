"""
Vercel Serverless Function entry point for AI Face Analytics FastAPI app.
"""
import sys
import os

# Ensure project root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server import app
