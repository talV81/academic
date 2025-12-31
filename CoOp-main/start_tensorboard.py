#!/usr/bin/env python3
"""
Start TensorBoard server for viewing training logs.
This script handles the TensorBoard startup and provides the URL.
"""
import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def start_tensorboard(logdir, port=6006, host="0.0.0.0"):
    """Start TensorBoard server."""
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(logdir),
        "--port", str(port),
        "--host", host
    ]
    
    print("=" * 70)
    print("Starting TensorBoard...")
    print("=" * 70)
    print(f"Log directory: {logdir}")
    print(f"Port: {port}")
    print(f"Host: {host}")
    print()
    print("TensorBoard will be available at:")
    print(f"  http://localhost:{port}")
    print()
    print("Press Ctrl+C to stop TensorBoard")
    print("=" * 70)
    print()
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nTensorBoard stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start TensorBoard for viewing training logs")
    parser.add_argument("--logdir", "-d", type=str, 
                       default="output/",
                       help="Directory containing TensorBoard logs (default: output/)")
    parser.add_argument("--port", "-p", type=int, default=6006,
                       help="Port to run TensorBoard on (default: 6006)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    logdir = Path(args.logdir).resolve()
    if not logdir.exists():
        print(f"Error: Directory {logdir} does not exist")
        sys.exit(1)
    
    start_tensorboard(logdir, args.port, args.host)




