import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_name: str) -> None:
    script_path = PROJECT_ROOT / "scripts" / script_name
    print(f"Running {script_name}...")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    run_script("run_prepare.py")
    run_script("run_train_basic.py")
    run_script("run_train_deep.py")
    run_script("run_evaluate.py")
    print("Full pipeline finished.")


if __name__ == "__main__":
    main()

