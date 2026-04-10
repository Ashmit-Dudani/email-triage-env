"""
server/app.py - Entry point for multi-mode deployment.
This wraps the main app and exposes a main() function
as required by the OpenEnv spec.
"""

import uvicorn
import sys
import os

# Add parent directory to path so we can import app, models, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main entry point for the server."""
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()