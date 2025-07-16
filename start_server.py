#!/usr/bin/env python3
"""
Simple startup script for the Dekr Strategy Builder Enhanced
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Change to the project directory
    os.chdir('/home/runner/workspace')
    
    # Run the FastAPI application with Uvicorn
    cmd = [
        sys.executable, '-m', 'uvicorn', 
        'app:app', 
        '--host', '0.0.0.0', 
        '--port', '5000', 
        '--reload'
    ]
    
    print("Starting Dekr Strategy Builder Enhanced...")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)