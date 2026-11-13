import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

if __name__ == "__main__":
    venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable

    print(f"Starting DocuMind Backend using {python_cmd}...")
    try:
        subprocess.run([
            python_cmd, "-m", "uvicorn", "app.main:app",
            "--app-dir", str(BACKEND_DIR),
            "--host", "127.0.0.1",
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\nDocuMind Backend stopped.")

