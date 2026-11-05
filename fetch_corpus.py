"""Copy docs from any git repo into data/corpus. Example (a docs-heavy project of your choice):
python scripts/fetch_corpus.py https://github.com/<org>/<repo> --subdir docs"""
import argparse, shutil, subprocess, tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--subdir", default=".")
ap.add_argument("--out", default="data/corpus")
a = ap.parse_args()
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["git", "clone", "--depth", "1", a.repo, tmp], check=True)
    n = 0
    for f in (Path(tmp) / a.subdir).rglob("*"):
        if f.suffix.lower() in {".md", ".rst", ".txt"}:
            dst = Path(a.out) / f.relative_to(Path(tmp) / a.subdir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst); n += 1
print(f"copied {n} files to {a.out}")
