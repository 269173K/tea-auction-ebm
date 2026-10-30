"""Reproduce every result in the report, in order: dataset -> training/evaluation ->
explainability -> front-end bundle -> standalone page. Takes about two minutes."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
steps = ["src/build_dataset.py", "src/train_evaluate.py", "src/explain.py", "src/export_bundle.py", "app/build_standalone.py"]
for s in steps:
    print(f"\n=== {s}")
    subprocess.run([sys.executable, str(ROOT / s)], check=True, cwd=ROOT)
print("\nDone. Figures in outputs/figures, tables in outputs/tables. Start the app with: python app/app.py")
