#!/usr/bin/env python3
"""Build a flat Overleaf bundle: all PNGs at root, paths rewritten."""
import re, shutil, subprocess, sys
from pathlib import Path

OUT = Path("overleaf_flat")

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    src = Path("paper.latex").read_text()

    for path in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', src):
        p = Path(path)
        if not p.exists():
            print(f"MISSING: {path}", file=sys.stderr); return 1
        shutil.copy(p, OUT / p.name)
        src = src.replace("{" + path + "}", "{" + p.name + "}")
        print(f"  {path} -> {p.name}")

    (OUT / "paper.tex").write_text(src)
    subprocess.run(["zip", "-qr", "../overleaf_flat.zip", "."], cwd=OUT, check=True)
    print(f"\nwrote overleaf_flat.zip ({Path('overleaf_flat.zip').stat().st_size//1024} KB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
