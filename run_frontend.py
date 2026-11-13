import os
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_APP = ROOT_DIR / "frontend" / "app.py"

if __name__ == "__main__":
    venv_streamlit = ROOT_DIR / ".venv" / "Scripts" / "streamlit.exe"

    cmd = [
        str(venv_streamlit) if venv_streamlit.exists() else sys.executable,
        "run" if venv_streamlit.exists() else "-m",
    ]
    if not venv_streamlit.exists():
        cmd.extend(["streamlit", "run"])

    cmd.extend([
        str(FRONTEND_APP),
        "--browser.gatherUsageStats", "false",
        "--server.port", "8501"
    ])

    print(f"Starting DocuMind Streamlit UI...")
    print(f"Command: {' '.join(cmd)}")
    print(f"UI will be available at: http://localhost:8501")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDocuMind Frontend stopped.")
