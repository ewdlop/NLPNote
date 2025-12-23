#!/usr/bin/env python3
"""
Universal Planetary Notebook Launcher
全行星通用笔记本启动器

Quick launcher script for the Universal Planetary Notebook.
"""

import os
import sys
import argparse
import webbrowser
import subprocess
from pathlib import Path

def main():
    print("🪐 Universal Planetary Notebook Launcher")
    print("全行星通用笔记本启动器")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="Launch Universal Planetary Notebook")
    parser.add_argument("--port", type=int, default=8888, help="Server port (default: 8888)")
    parser.add_argument("--demo", action="store_true", help="Launch with demo content")
    parser.add_argument("--interface", action="store_true", help="Open interactive web interface")
    parser.add_argument("--file", type=str, help="Load specific notebook file")
    
    args = parser.parse_args()
    
    # Check if we're in the correct directory
    if not os.path.exists("universal_notebook.py"):
        print("❌ Error: universal_notebook.py not found!")
        print("Please run this script from the NLPNote repository root directory.")
        sys.exit(1)
    
    if args.interface:
        # Open the standalone web interface
        interface_path = Path("notebook_interface.html").absolute()
        if interface_path.exists():
            print(f"🌐 Opening interactive web interface...")
            print(f"📂 File: {interface_path}")
            webbrowser.open(f"file://{interface_path}")
        else:
            print("❌ Error: notebook_interface.html not found!")
            sys.exit(1)
    else:
        # Launch the Python server
        cmd = ["python3", "universal_notebook.py", "--port", str(args.port)]
        
        if args.demo:
            cmd.append("--demo")
            print("🎭 Launching with demo content...")
        
        if args.file:
            cmd.extend(["--file", args.file])
            print(f"📖 Loading notebook file: {args.file}")
        
        print(f"🚀 Starting server on port {args.port}...")
        print("💡 Tip: Press Ctrl+C to stop the server")
        print("🌍 The notebook will open automatically in your browser")
        print()
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

if __name__ == "__main__":
    main()