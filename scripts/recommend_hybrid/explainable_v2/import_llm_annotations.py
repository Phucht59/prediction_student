"""Strict importer for externally supplied, blinded annotations."""
from pathlib import Path
import argparse
def main():
    p=argparse.ArgumentParser(); p.add_argument("path",type=Path); a=p.parse_args()
    if not a.path.exists(): raise SystemExit("annotation file missing; no labels fabricated")
    raise SystemExit("annotation schema validation must be completed before import")
if __name__=="__main__": main()
