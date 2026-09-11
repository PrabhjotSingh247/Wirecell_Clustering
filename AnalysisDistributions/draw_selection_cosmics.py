"""
SELECTED COSMIC RECO CLUSTERS -- driven by

    AnalysisDistributions/Draw_Selection_Cosmics.ipynb                 all of them
    AnalysisDistributions/Draw_Cosmics_Survived_After_CosmicTagger.ipynb   tagger MISSED
    AnalysisDistributions/Draw_Cosmics_Removed_After_CosmicTagger.ipynb        tagger CAUGHT

Reco clusters that survived the beam-window cut and matched NO true neutrino:
the background the selection is left with. The three notebooks split that
population by what the cosmic tagger cut did to it -- all, survived, removed --
using the same code and layout, only a different directory and (for the tagged
set) a tagged_by_cluster filter passed by the notebook.

PER-CLUSTER VIEWS, in XZ, YZ and XY. Reco points only -- a cosmic candidate is by
definition a reco cluster with no true neutrino to put above it, so these are
single-row figures rather than the two-row layout the pair populations use.

    category        == 'cosmic'
    true energy     >  MIN_COSMIC_VIEW_ENERGY_MEV   (100 MeV)

The energy is that of the true cluster the reco cluster overlaps MOST (by
purity), summed over all its points -- not the reco cluster's own charge, and not
just the overlapping part. A cosmic reco cluster is by construction a fragment,
so its own energy says how much was reconstructed, while the true cluster's says
how much was there to reconstruct, which is the thing worth looking at. A
candidate overlapping no true cluster at all has no true energy and is not drawn.

The cut exists because the cosmic category is mostly small fragments -- the
smallest in one chunk was 2 MeV -- and a three-panel view of a handful of points
shows nothing.

NO SAMPLING. Every qualifying candidate is drawn. These views used to be a
reservoir sample of 10, drawn by draw_saved_clusters.py, because they shared a
directory with the completeness-purity grid and the count had to be bounded. The
energy gate turns out to bound it already: the full sample holds 15 qualifying
candidates, so a random 10 of 15 threw away a third of a population small enough
to look at whole. Every one is now drawn, as for the contaminated pairs.

NO DISTRIBUTIONS. Only the per-cluster views. The charge and reco-energy
histograms of the selected cosmics (draw_selection_performance.
draw_cosmic_distributions) are not drawn here and, since they also left
SignalBackground_Distributions.ipynb, are not drawn anywhere -- restore the call
in either notebook if they are wanted again.

WHY A SEPARATE MODULE. These outputs used to come from
SignalBackground_Distributions.ipynb, split across two files. That notebook draws
DISTRIBUTIONS; a picture of one cosmic cluster is not one, and the pictures cost
event-loop time there for output nobody reads in that context. They now sit with
the other per-cluster views, in the layout Contamination_Clusters established.
"""

from pathlib import Path

import numpy as np

from draw_saved_clusters import _draw_panels, _RECO_STYLE
from draw_contamination_clusters import (
    _id_text, bee_event_url, load_bee_links, split_event_key,
)


# A cosmic candidate is only worth a picture above this TRUE deposited energy, in
# MeV. See the module docstring for which true energy this is and why.
MIN_COSMIC_VIEW_ENERGY_MEV = 100.0

COSMIC_DIR_NAME = 'selection_cosmics'

# The same population drawn AFTER the cosmic tagger cut, by
# Draw_Cosmics_Survived_After_CosmicTagger.ipynb: cosmic candidates the taggers
# did NOT catch. Same code, same layout, different directory -- what differs is
# which clusters reach it, and that is decided by the notebook.
COSMICS_SURVIVED_DIR_NAME = 'CosmicsSurvived'

# The complement of the surviving set: cosmic candidates the tagger cut REMOVED,
# drawn by Draw_Cosmics_Removed_After_CosmicTagger.ipynb. That notebook runs the
# tagger without applying the cut and passes tagged_by_cluster to
# save_event_cosmic_views, so only the removed cosmics reach a figure.
COSMICS_REMOVED_DIR_NAME = 'CosmicsRemoved'

# The sentinel EvaluatePurity marks a reco cluster that matched no true cluster.
_PURITY_UNMATCHED_TRUE_ID = 8888


def cosmic_true_energy_mev(record, purity_results, clusters_true):
    """
    (energy, true cluster id) of the true cluster this reco cluster overlaps most,
    or (None, None) if it overlaps none.

    Chosen by purity because that is the overlap measured from the RECO side --
    "how much of this reco cluster is that true cluster" -- which is the right
    question for a fragment. Completeness would favour whichever true cluster is
    smallest.
    """
    best = None
    for entry in purity_results or []:
        if entry.get('reco_cluster_id') != record['reco_cluster_id']:
            continue
        if entry.get('true_cluster_id') == _PURITY_UNMATCHED_TRUE_ID or (entry.get('purity') or 0) <= 0:
            continue
        if best is None or entry['purity'] > best['purity']:
            best = entry
    if best is None:
        return None, None
    true_points = clusters_true.get(best['true_cluster_id'])
    if true_points is None:
        return None, None
    return float(np.asarray(true_points)[:, 5].sum()), best['true_cluster_id']


def draw_cosmic_views(record, clusters_reco, output_root, event_key,
                      true_energy=None, true_cluster_id=None, bee_url=None,
                      dir_name=COSMIC_DIR_NAME,
                      title="Cosmic candidate reco cluster",
                      tagged_detail=None):
    """
    One cosmic candidate in XZ, YZ and XY -- reco points only.

    tagged_detail, when given, is this cluster's entry from tag_reco_clusters --
    it adds a line saying which tagger flagged it, or that the tag reached it by
    propagation across the beam window.

    Returns the path written, or None when the event's point clouds no longer
    hold the cluster.
    """
    reco_points = clusters_reco.get(record['reco_cluster_id'])
    if reco_points is None:
        return None

    chunk, event = split_event_key(event_key)
    legend_lines = [
        f"event {event_key}",
        "cosmic candidate (no true neutrino match)",
        f"reco id {record['reco_cluster_id']:.3f}",
        f"reco E {(record.get('reco_energy_mev') or 0):.0f} MeV",
        f"true E {true_energy:.0f} MeV" if true_energy is not None else "true E n/a",
        (f"of true id {true_cluster_id:.0f} (largest overlap)"
         if true_cluster_id is not None else "no true overlap"),
    ]
    if tagged_detail is not None:
        legend_lines.append(
            "REMOVED: tagged by " + ", ".join(tagged_detail['taggers'])
            if tagged_detail.get('tagged_directly') else
            "REMOVED: tagged by propagation across the beam window")

    # Same convention as the pair populations -- every number LABELLED -- minus
    # the true id, which names the cluster the ENERGY came from rather than a
    # match, and would read as one in a filename.
    name = (f"reco_cosmic_cluster_{chunk}_event{event}"
            f"_recoID{_id_text(record['reco_cluster_id'])}.png")
    footer_note = ('bee-display', bee_url) if bee_url else None
    return _draw_panels(
        title,
        [(reco_points, _RECO_STYLE)],
        # One sub-directory per chunk, as for the contaminated pairs: the chunk is
        # already in every filename, so this adds no information -- it makes the
        # directory browsable instead of one flat listing of the whole sample.
        Path(output_root) / dir_name / (chunk or 'unknown_chunk') / name,
        legend_lines,
        footer_note=footer_note)


def save_event_cosmic_views(selection_records, clusters_true, clusters_reco,
                            output_root, event_key, purity_results=None,
                            bee_links=None,
                            min_true_energy=MIN_COSMIC_VIEW_ENERGY_MEV,
                            dir_name=COSMIC_DIR_NAME,
                            title="Cosmic candidate reco cluster",
                            tagged_by_cluster=None):
    """
    Every qualifying cosmic candidate in ONE event.

    Called from inside the event loop because that is the only place the point
    clouds exist -- they are far too large to carry to job level.

    purity_results for this event is required to identify the overlapping true
    cluster; without it no candidate has a true energy and none is drawn.

    tagged_by_cluster, when given, is tag_reco_clusters' output for this event --
    the population is then restricted to the cosmics the tagger cut REMOVED, and
    n_cosmics_seen counts only those. When None every cosmic candidate is in play.

    Returns (entries, n_cosmics_seen): one entry per figure written, and how many
    cosmic candidates the event held in total. The second number is what makes an
    empty directory readable -- a job with 900 cosmics and no figures means the
    energy gate is doing the work, not that the selection is clean.
    """
    chunk, event = split_event_key(event_key)
    bee_url = bee_event_url((bee_links or {}).get(chunk), event) if event else None

    written, n_seen = [], 0
    for record in selection_records or []:
        if record.get('category') != 'cosmic':
            continue
        tagged_detail = None
        if tagged_by_cluster is not None:
            tagged_detail = tagged_by_cluster.get(record.get('reco_cluster_id'))
            if tagged_detail is None:
                continue
        n_seen += 1
        true_energy, true_cluster_id = cosmic_true_energy_mev(
            record, purity_results, clusters_true)
        # The energy gate is the "all cosmic candidates" case's tool for cutting
        # the fragment pile-up down to something worth looking at. The tagged set
        # (tagged_by_cluster given) is small and already pinned to what the cut
        # removed, so every one of them is drawn -- even those overlapping no
        # surviving true cluster, which still show the cosmic the tagger caught.
        if tagged_by_cluster is None and (true_energy is None or true_energy <= min_true_energy):
            continue
        path = draw_cosmic_views(record, clusters_reco, output_root, event_key,
                                 true_energy=true_energy, true_cluster_id=true_cluster_id,
                                 bee_url=bee_url, dir_name=dir_name, title=title,
                                 tagged_detail=tagged_detail)
        if path is None:
            continue
        written.append({
            'path':            path,
            'event_key':       event_key,
            'chunk':           chunk,
            'event':           event,
            'reco_cluster_id': record.get('reco_cluster_id'),
            'true_cluster_id': true_cluster_id,
            'true_energy_mev': true_energy,
            'reco_energy_mev': record.get('reco_energy_mev'),
            'total_charge':    record.get('total_charge'),
            'tagged_directly': (tagged_detail or {}).get('tagged_directly'),
            'taggers':         (tagged_detail or {}).get('taggers', []),
            'bee_url':         bee_url,
        })
    return written, n_seen


def _relative_path(entry):
    """
    'chunk3/reco_cosmic_cluster_chunk3_event57_recoID12.png' -- the path as it
    should appear in the index, relative to the directory the index sits in.
    """
    path = Path(entry['path'])
    return f"{path.parent.name}/{path.name}"


def write_cosmic_index(entries, output_root, n_cosmics_seen=None,
                       min_true_energy=MIN_COSMIC_VIEW_ENERGY_MEV,
                       filename='selection_cosmics.txt',
                       dir_name=COSMIC_DIR_NAME, preamble=None,
                       headline='SELECTED COSMIC RECO CLUSTERS -- index',
                       bee_set_url=None):
    """
    The index: one row per figure, largest true energy first -- the biggest
    cosmics are the ones a selection most needs to explain.

    Also writes bee_links.txt beside it -- filename and URL only, nothing else --
    because that is the file a reader can actually click a link out of. The URL is
    printed on each figure too, but a PNG cannot hold a working link.
    """
    output_root = Path(output_root) / dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries or [], key=lambda e: -(e['true_energy_mev'] or 0))

    lines = []
    lines.append("=" * 104)
    lines.append(headline)
    lines.append("=" * 104)
    lines.append("")
    if preamble:
        lines.extend(preamble)
    else:
        lines.append("EVERY reco cluster -- no sampling -- that survived the beam-window cut,")
        lines.append("matched no true neutrino, and overlaps a true cluster depositing more than")
        lines.append(f"{min_true_energy:.0f} MeV.")
    lines.append("")
    lines.append("The energy tested is the TRUE cluster's, not the reco cluster's own charge:")
    lines.append("a cosmic reco cluster is a fragment, so its own energy says how much was")
    lines.append("reconstructed while the true cluster's says how much was there. E reco can")
    lines.append("therefore exceed E true, because a reco cosmic cluster may span several")
    lines.append("true ones and only the largest overlap is reported here.")
    lines.append("")
    lines.append("Reco points only -- a cosmic candidate has no true neutrino cluster to draw")
    lines.append("beside it, so these are single-row figures.")
    lines.append("")
    if n_cosmics_seen is None:
        lines.append(f"{len(entries)} figure(s).")
    else:
        lines.append(f"{len(entries)} figure(s), from {n_cosmics_seen} selected cosmic "
                     f"cluster(s) in the job.")
    lines.append("")
    lines.append("-" * 104)
    lines.append(f"  {'event':<14s}{'true E':>10s}{'reco E':>10s}{'charge':>14s}"
                 f"{'reco id':>11s}{'true id':>10s}  file")
    lines.append("-" * 104)
    for entry in entries:
        lines.append(
            f"  {str(entry['event_key']):<14s}{(entry['true_energy_mev'] or 0):>10.0f}"
            f"{(entry['reco_energy_mev'] or 0):>10.0f}{(entry['total_charge'] or 0):>14.0f}"
            f"{_id_text(entry['reco_cluster_id']):>11s}"
            f"{_id_text(entry['true_cluster_id']):>10s}  {_relative_path(entry)}")

    index_path = output_root / filename
    index_path.write_text("\n".join(lines) + "\n")

    link_lines = ["# BEE event display, one per figure. The same URL is printed on the",
                  "# figure itself, where it cannot be clicked -- PNG has no hyperlinks."]
    if bee_set_url:
        link_lines.append(f"# BEE SET (whole population, one upload): {bee_set_url}")
    link_lines.append("")
    for entry in entries:
        if entry.get('bee_url'):
            link_lines.append(f"{_relative_path(entry)}  {entry['bee_url']}")
    (output_root / 'bee_links.txt').write_text("\n".join(link_lines) + "\n")
    return index_path
