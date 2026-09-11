#!/usr/bin/env python3
"""
Group the flat numucc bee/ zips into bee/chunk_NN/subchunk_MM/, the SAME layout
as the nuecc sample (see that sample's bee/chunk_manifest.txt):

  chunk assignment : seeded random  (seed 20260830 -- the sample's own date,
                     analogous to nuecc's seed 20260829)
  subchunk         : by SORTED zip name within the chunk, sliced into 10s
  100 zips / chunk, 10 / subchunk; the final chunk / subchunk carries the remainder

MOVES the files (mv), then writes bee/chunk_manifest.txt. Re-runnable: a zip
already sitting in its computed target is left alone.

  python3 chunk_numucc_sample.py --dry-run     # report only
  python3 chunk_numucc_sample.py               # do it
"""
import argparse
import random
import re
import sys
from pathlib import Path

BEE = Path("/Volumes/My Passport/Research_Life/Experiment/SBND/"
           "Wirecell_Reconstruction/Samples/"
           "img-clus-match-tag-pr-mc-1000file-sync-2026-08-30/bee")
SEED = 20260830
CHUNK_SIZE = 100
SUBCHUNK_SIZE = 10
ZIP_RE = re.compile(r'^bee_r\d+_s\d+_e\d+\.zip$')


def plan():
    """[(zip_name, chunk_name, subchunk_name), ...] over every bee_*.zip that is
    either in the flat bee root or already in a chunk_NN/subchunk_MM/ dir."""
    flat = sorted(p.name for p in BEE.iterdir()
                  if p.is_file() and ZIP_RE.match(p.name))
    already = []
    for chunk_dir in BEE.glob("chunk_*"):
        if not chunk_dir.is_dir():
            continue
        for sub_dir in chunk_dir.glob("subchunk_*"):
            for z in sub_dir.glob("*.zip"):
                if ZIP_RE.match(z.name):
                    already.append(z.name)

    all_names = sorted(set(flat) | set(already))
    if not all_names:
        sys.exit(f"no bee_*.zip found under {BEE}")

    # Seeded shuffle -> chunk assignment. Deterministic given SEED and the full
    # sorted name list, so re-running reproduces the same layout.
    shuffled = list(all_names)
    random.Random(SEED).shuffle(shuffled)
    chunk_of = {}
    for i, name in enumerate(shuffled):
        chunk_of[name] = f"chunk_{i // CHUNK_SIZE:02d}"

    # subchunk: sort the chunk's members by name, slice into SUBCHUNK_SIZE.
    by_chunk = {}
    for name, ch in chunk_of.items():
        by_chunk.setdefault(ch, []).append(name)
    rows = []
    for ch in sorted(by_chunk):
        for j, name in enumerate(sorted(by_chunk[ch])):
            rows.append((name, ch, f"subchunk_{j // SUBCHUNK_SIZE:02d}"))
    return rows, len(flat), len(already)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows, n_flat, n_already = plan()
    n_chunks = len({r[1] for r in rows})
    last_chunk = max(r[1] for r in rows)
    last_chunk_n = sum(1 for r in rows if r[1] == last_chunk)
    print(f"{len(rows)} zips  ({n_flat} still flat, {n_already} already in chunk dirs)")
    print(f"-> {n_chunks} chunks (chunk_00 .. {last_chunk}, last has {last_chunk_n})")
    if args.dry_run:
        from collections import Counter
        sub_counts = Counter((r[1], r[2]) for r in rows)
        odd = {k: v for k, v in sub_counts.items() if v != SUBCHUNK_SIZE}
        print(f"subchunks not exactly {SUBCHUNK_SIZE}: {sorted(odd.items())}")
        for name, ch, sub in rows[:5]:
            print(f"  {name}  ->  {ch}/{sub}")
        return

    moved = skipped = 0
    for name, ch, sub in rows:
        dest_dir = BEE / ch / sub
        dest = dest_dir / name
        src = BEE / name
        if dest.exists():
            skipped += 1
            continue
        if not src.exists():
            sys.exit(f"MISSING: {name} is neither flat nor at {dest}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        moved += 1
        if moved % 500 == 0:
            print(f"  moved {moved} ...", flush=True)
    print(f"moved {moved}, skipped {skipped} (already placed)")

    manifest = BEE / "chunk_manifest.txt"
    with open(manifest, "w") as f:
        f.write(f"# numucc sample -- {len(rows)} zips in bee/chunk_NN/subchunk_MM/ "
                f"({CHUNK_SIZE}/chunk, {SUBCHUNK_SIZE}/subchunk; final chunk/subchunk carries the remainder)\n")
        f.write(f"# chunk assignment: seeded random (seed {SEED}); subchunk: by sorted zip name\n")
        f.write("# chunk\tsubchunk\tzip\n")
        for name, ch, sub in rows:
            f.write(f"{ch}\t{sub}\t{name}\n")
    print(f"wrote {manifest}")

    leftover = [p.name for p in BEE.iterdir() if p.is_file() and ZIP_RE.match(p.name)]
    if leftover:
        print(f"WARNING: {len(leftover)} zip(s) still flat in bee/: {leftover[:5]}")
    else:
        print("bee/ root is clean -- only chunk_* dirs and chunk_manifest.txt remain")


if __name__ == "__main__":
    main()
