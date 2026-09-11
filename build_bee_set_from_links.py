"""
Build ONE BEE upload from a selection spread across several chunks.

A BEE url is .../set/<set id>/event/<n>/, and the set id identifies one upload --
so events from ten different chunk uploads cannot share a link no matter how the
urls are written. The only way to get a single link for a selection is to upload
the selection as its own set, which is what this builds.

INPUT: any number of bee_links_*.txt files written by draw_tagger_impact (or any
file whose figure names carry chunk<N>_event<M>). The (chunk, event) pairs are
taken from the FIGURE NAMES rather than the urls, because a figure name survives
the renumbering below and a url does not.

OUTPUT: <out>/data/<n>/<n>-<rest>.json for n = 0, 1, 2, ... -- the layout
bee-upload-with-truth.sh expects -- plus event_map.txt, and a zip of the lot.

EVENTS ARE RENUMBERED. BEE indexes events by position in the set, so the new set
cannot keep chunk4's event 96 at 96 while chunk8's event 96 is also present.
event_map.txt maps every new number back to its chunk, original event and the
figures that selected it; without it the uploaded set is unattributable.

FILES ARE HARD-LINKED where the filesystem allows, so a 1.6 GB selection costs no
extra disk until the zip is written. A copy is used when linking fails (different
filesystem), which is slower but produces the same tree.

Run:  python3 build_bee_set_from_links.py <links.txt> [<links.txt> ...] --out <dir>
"""

import argparse
import os
import re
import shutil
import subprocess
import zipfile
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_PARENT = REPO / 'Haiwang_files_charge_light_matching_Tagger_Included_MCP2025C_FallProd_100files'

# chunk<N>_event<M> anywhere in a figure name.
_FIGURE = re.compile(r'chunk(\d+)_event(\d+)')


def selections_from_links(paths):
    """
    OrderedDict {(chunk, event): [figure names]} over every links file given.

    Ordered by chunk then event so the new numbering is predictable, and keyed so
    an event selected by two figures -- one neutrino matched by two reco clusters
    -- is uploaded once and carries both figure names in the map.
    """
    found = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            figure = line.split()[0]
            match = _FIGURE.search(figure)
            if not match:
                continue
            key = (f"chunk{match.group(1)}", int(match.group(2)))
            found.setdefault(key, []).append(figure)
    return OrderedDict(sorted(found.items(), key=lambda kv: (kv[0][0], kv[0][1])))


def link_or_copy(src, dst):
    """Hard-link src to dst, falling back to a copy across filesystems."""
    try:
        os.link(src, dst)
        return 'linked'
    except OSError:
        shutil.copy2(src, dst)
        return 'copied'


def build(selections, parent_dir, out_dir):
    """
    Assemble data/<n>/ for each selected event. Returns the rows for event_map.txt.
    """
    data_dir = Path(out_dir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    rows, n_linked, n_copied = [], 0, 0
    for new_event, ((chunk, event), figures) in enumerate(selections.items()):
        source = Path(parent_dir) / chunk / 'data' / str(event)
        if not source.is_dir():
            print(f"  WARNING: {source} missing -- skipped")
            continue
        target = data_dir / str(new_event)
        target.mkdir(exist_ok=True)
        for src in sorted(source.iterdir()):
            if not src.is_file() or src.suffix != '.json':
                continue
            # <event>-<rest>.json -> <new event>-<rest>.json; the prefix must
            # match the directory or BEE will not find the file.
            rest = src.name[len(f"{event}-"):] if src.name.startswith(f"{event}-") else src.name
            how = link_or_copy(src, target / f"{new_event}-{rest}")
            n_linked += how == 'linked'
            n_copied += how == 'copied'
        rows.append((new_event, chunk, event, figures))
    print(f"  {len(rows)} event(s) assembled, {n_linked} file(s) linked, {n_copied} copied")
    return rows


def write_event_map(rows, out_dir, links_files):
    path = Path(out_dir) / 'event_map.txt'
    lines = ["=" * 88,
             "BEE SET -- what each event in this upload actually is",
             "=" * 88,
             "",
             "Events were RENUMBERED: BEE indexes them by position in the set, so the",
             "original chunk event numbers could not be kept (two chunks can both hold an",
             "event 96). Read this file to get from a BEE event number back to the data.",
             "",
             "Built from:"]
    lines += [f"    {Path(p).name}" for p in links_files]
    lines += ["", f"{len(rows)} event(s).", "",
              "-" * 88,
              f"  {'bee event':>9s}  {'chunk':<8s}{'orig event':>11s}  figures",
              "-" * 88]
    for new_event, chunk, event, figures in rows:
        lines.append(f"  {new_event:>9d}  {chunk:<8s}{event:>11d}  {figures[0]}")
        for extra in figures[1:]:
            lines.append(f"  {'':>9s}  {'':<8s}{'':>11s}  {extra}")
    lines.append("=" * 88)
    path.write_text("\n".join(lines) + "\n")
    return path


# The BEE server rejects a POST above roughly a gigabyte with 413. 827 MB has
# uploaded successfully and 2.10 GB has not, so warn from 0.9 GB -- the caller
# still gets the zip, it just has to be split before it will upload.
UPLOAD_SIZE_WARN_GB = 0.9


def make_zip(out_dir, zip_path):
    """
    Zip data/ and event_map.txt, the shape bee-upload-with-truth.sh uploads.

    The file list is taken ONCE and then every entry must still be there when it
    is written: a file that disappears mid-zip is an ERROR, not something to skip.
    This is not hypothetical -- renaming the run directory while a 2 GB set was
    being zipped made an earlier version silently drop 3067 of 7719 files and
    produce a structurally valid archive with no event_map.txt, which would have
    uploaded as a plausible-looking BEE set missing 40% of its events. The count
    is checked against the archive afterwards for the same reason.
    """
    out_dir = Path(out_dir)
    files = [p for p in sorted(out_dir.rglob('*'))
             if p.is_file() and p.name != Path(zip_path).name]
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            if not path.is_file():
                raise RuntimeError(
                    f"{path} vanished while zipping -- was the run directory renamed "
                    f"or cleaned mid-build? Nothing was uploaded; re-run the build.")
            archive.write(path, path.relative_to(out_dir))

    written = zipfile.ZipFile(zip_path).namelist()
    if len(written) != len(files):
        raise RuntimeError(f"zip holds {len(written)} entries, expected {len(files)}")
    if 'event_map.txt' not in written:
        raise RuntimeError("zip has no event_map.txt -- the set would be unattributable")
    return Path(zip_path)


_SET_URL_RE = re.compile(r'https?://\S+/set/[0-9a-f-]+/event/')


def upload_bee_zip(zip_path, upload_script=None, cwd=None):
    """
    Upload one BEE zip via upload-to-bee.sh and return the set URL it prints
    (its last stdout line, of the form .../set/<id>/event/list/).

    Returns None -- and prints why -- if the upload script is missing, the process
    fails or times out, or the last line is not a set URL. An upload that did not
    happen must never be mistaken for one that did.

    cwd defaults to the zip's directory: upload-to-bee.sh writes and deletes a
    cookies.txt in the working directory, so it must run somewhere writable.
    """
    zip_path = Path(zip_path)
    upload_script = Path(upload_script) if upload_script else REPO / 'upload-to-bee.sh'
    if not upload_script.exists():
        print(f"  BEE upload skipped: {upload_script} not found")
        return None
    cwd = Path(cwd) if cwd is not None else zip_path.parent
    try:
        proc = subprocess.run(
            ['bash', str(upload_script), str(zip_path.resolve())],
            cwd=str(cwd), capture_output=True, text=True, timeout=1800,
            env={**os.environ, 'BROWSER': 'echo'})
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  BEE upload failed to run: {exc}")
        return None
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    url = lines[-1] if lines else ''
    if not _SET_URL_RE.match(url):
        print(f"  BEE upload did not return a set URL (rc={proc.returncode}); "
              f"last line: {url!r}")
        if proc.stderr.strip():
            print(f"  stderr: {proc.stderr.strip()[:400]}")
        return None
    return url


def build_and_upload_set(selections, parent_dir, out_dir, source_label,
                         upload=True, upload_script=None):
    """
    build() + write_event_map() + make_zip() for a {(chunk, event): [figures]}
    selection, then (upload=True) upload it as ONE BEE set.

    Returns (set_url or None, event_map_path or None, rows) where rows is
    build()'s output -- [(bee_event_number, chunk, orig_event, figures), ...] --
    so a caller can map (chunk, orig_event) to <set>/event/<bee_event_number>/.
    On a successful upload the URL is also written into event_map.txt as a
    'BEE SET URL:' line under the header (see feedback_bee_sets_per_run). Returns
    (None, None, []) if the selection assembled no events.
    """
    out_dir = Path(out_dir)
    rows = build(selections, parent_dir, out_dir)
    if not rows:
        return None, None, []
    map_path = write_event_map(rows, out_dir, [source_label])
    zip_path = out_dir.with_suffix('.zip')
    make_zip(out_dir, zip_path)
    url = upload_bee_zip(zip_path, upload_script) if upload else None
    if url:
        header = "BEE SET -- what each event in this upload actually is"
        map_path.write_text(map_path.read_text().replace(
            header, f"{header}\nBEE SET URL: {url}", 1))
    return url, map_path, rows


def build_population_bee_set(entries, parent_dir, out_dir, label,
                             key_fn=None, upload=True):
    """
    ONE BEE set for a whole saved-view population, and entry['bee_url'] set on
    every entry in place to its per-event url inside that new set.

    The nuecc sample has no per-chunk BEE sets, so a population's figures cannot
    otherwise carry a link. This builds a set from the population's own events
    (via build_and_upload_set), uploads it, and back-fills the links so the
    normal bee_links writers produce real urls.

    entries: list of dicts, each with a 'path' (a drawn figure) and a way to name
    its staged event. key_fn(entry) -> (staged_dir_name, event_int); the default
    reads entry['chunk'] and int(entry['event']).

    Returns the set url (or None -- empty population, nothing assembled, or the
    upload failed; entry['bee_url'] is then left untouched).
    """
    if not entries:
        return None
    if key_fn is None:
        def key_fn(entry):
            return (entry['chunk'], int(entry['event']))

    selections = OrderedDict()
    for entry in entries:
        try:
            key = key_fn(entry)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        selections.setdefault(key, []).append(Path(entry.get('path') or '').name or str(key))
    if not selections:
        return None
    selections = OrderedDict(sorted(selections.items(),
                                    key=lambda kv: (str(kv[0][0]), kv[0][1])))

    url, _map_path, rows = build_and_upload_set(selections, parent_dir, out_dir,
                                                label, upload=upload)
    if not url:
        return None
    # url ends '.../set/<id>/event/list/'; the per-event url swaps 'list' for n.
    event_base = url.rstrip('/').rsplit('/', 1)[0]
    per_event = {(chunk, event): f"{event_base}/{n}/" for n, chunk, event, _figs in rows}
    for entry in entries:
        try:
            entry['bee_url'] = per_event.get(key_fn(entry), url)
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('links', nargs='+', help='bee_links_*.txt file(s)')
    parser.add_argument('--parent-dir', default=str(DEFAULT_PARENT),
                        help='the production tree holding chunk<N>/data/<event>/')
    parser.add_argument('--out', required=True, help='directory to build the set in')
    parser.add_argument('--no-zip', action='store_true', help='build the tree only')
    args = parser.parse_args()

    selections = selections_from_links(args.links)
    print(f"{len(selections)} distinct (chunk, event) selection(s) from "
          f"{len(args.links)} links file(s)")
    if not selections:
        return 1

    rows = build(selections, args.parent_dir, args.out)
    map_path = write_event_map(rows, args.out, args.links)
    print(f"  event map: {map_path}")

    if not args.no_zip:
        zip_path = Path(args.out).with_suffix('.zip')
        print("  zipping (this reads every file, so it takes a while)...")
        make_zip(args.out, zip_path)
        size_gb = zip_path.stat().st_size / 1e9
        print(f"  zip: {zip_path}  ({size_gb:.2f} GB)")
        if size_gb > UPLOAD_SIZE_WARN_GB:
            print(f"  WARNING: {size_gb:.2f} GB is likely over the BEE POST limit -- "
                  f"upload-to-bee.sh will return 413 and print the HTML error where the\n"
                  f"           set id belongs. Split the selection into smaller sets first.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
