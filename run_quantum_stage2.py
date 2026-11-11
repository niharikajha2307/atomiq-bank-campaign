"""
Helper launcher: sets cwd then runs quantum_stage2.py main().
Run this from anywhere with:
  python run_quantum_stage2.py
"""
import os, sys, subprocess

WORKSPACE = r"C:\Users\Rajiv Kumar Jha\Downloads\QML"
PYTHON    = os.path.join(WORKSPACE, ".venv", "Scripts", "python.exe")
SCRIPT    = os.path.join(WORKSPACE, "src", "quantum_stage2.py")

result = subprocess.run(
    [PYTHON, SCRIPT],
    cwd=WORKSPACE,
    capture_output=False,
)
sys.exit(result.returncode)
