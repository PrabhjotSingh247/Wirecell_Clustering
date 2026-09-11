"""
Standalone diagnostic script (not part of the main notebook pipeline), and the
true-side mirror of investigate_extra_reco_clusters.py: that one explains why
Evaluation_ChargeLightMatching_AfterBeamWindowCut.ipynb reports MORE selected
reco clusters than true neutrinos (88 vs 72), this one explains why FEWER true
neutrinos than that find a reco match at all -- job-wide only 61 of the 72 true
neutrinos form a 1-to-1 pair, so 11 are unaccounted for.

Every true neutrino cluster is categorized via
metadata.categorize_unmatched_true_neutrinos() into:
  - matched                     : found its MatchTrueToReco1to1 reco partner (not a failure)
  - wrong_charge_light_matching : the neutrino WAS imaged (img-global reco overlaps the
                                   sed-sce truth well) AND a clustering reco cluster of it
                                   exists, but charge-light matching set its drift (X)
                                   coordinate wrong -- no flash bridged, a flash outside
                                   the window, or a flash so wrong the cluster has no 3D
                                   overlap left but still lines up in YZ. Merges the old
                                   reco_no_flash_match, no_reco_overlap_x_shift and the
                                   wrong-flash rows of reco_outside_beam_window
  - removed_by_cosmic_tagger    : reconstructed and in time, but the cosmic tagger cut it
  - reco_outside_beam_window    : defensive residual -- clustering reconstructed something
                                   imaging did not, flash outside the window. Almost always empty
  - broken_or_sparse_reco       : reco charge sits on the neutrino, split too sparse to
                                   clear the completeness neighbor threshold
  - no_reco_overlap             : no clustering reco cluster for this neutrino at all -- no
                                   3D overlap AND no YZ alignment. img_ovl / img_pur say
                                   whether imaging had it (lost at the clustering stage) or
                                   not (never reconstructed anywhere)

The diagnosis works by re-running the overlap test against the FULL
pre-beam-window-cut reco set AND against the img-global (pre charge-light-
matching) reco, so each failure is attributed to the stage that actually
dropped it -- see metadata.categorize_unmatched_true_neutrinos' IMG-LEVEL
CROSS-CHECK. ALL output lives at JOB level, under one directory --
in_volume_neutrinos/ -- rather than also being duplicated per event or per file:
a per-event/per-file copy of the same aggregated charts was exactly the
chunk/subchunk-cluttered "head directory" tree that used to sit beside it and
has been removed. Event level still does its own pass over every event (it has
to, to find which events have an unmatched neutrino and draw their spatial
plot), it just no longer writes a breakdown chart/pies/info.txt of its own.

Writes job-level unmatched_true_neutrino_info.txt tables
(writeinformation.write_unmatched_true_neutrino_info), a job-level bar chart
(DrawRecoTrueClusters.DrawUnmatchedTrueNeutrinoBreakdown -- true neutrinos vs.
selected reco vs. pairs on top, matched vs. not matched in the middle, reasons on
the bottom), plus per-event XZ/YZ/XY spatial plots
(DrawRecoTrueClusters.DrawUnmatchedTrueNeutrinos, cluster IDs in the legend) for
every event with at least one unmatched true neutrino.

SPLIT BY POPULATION. Every one of those outputs (pies, breakdown chart, info.txt,
flash-time plot) is written once per population (see POPULATIONS below) --
IN-VOLUME, BY INTERACTION CHANNEL ONLY:

  in_volume_neutrinos/numu_CC/    vertex in volume, numu CC
  in_volume_neutrinos/nue_CC/     vertex in volume, nue CC
  in_volume_neutrinos/NC/         vertex in volume, NC

No 'all' and no plain in/out-of-volume population any more: a failure means
something different for an out-of-volume interaction (it only ever deposits the
part of itself that leaked into the active volume) than for one fully inside the
detector, so mixing them hides that, and the channel split is what the per-event
spatial plots (below) are organised under. Every population here is in-volume, so
the top directory itself says so (in_volume_neutrinos) rather than repeating an
'in_volume' segment under it on every path.

Only the TRUE side is split. The reco set is never cut, so a category assignment
is identical in every copy (it IS the same row), and the selected-reco bar in the
top panel stays the whole beam-window reco population everywhere, since that is
what all of these neutrinos were matched against. There is deliberately no
reco-side version of this split: with no vertex reconstruction, a reco cluster has
no volume and no channel of its own.

PER-EVENT SPATIAL PLOTS live one level deeper than the channel's own
pies/info.txt: channel, then FAILURE CATEGORY, then chunk and event -- so
browsing one category under one channel shows only the events that actually
belong to it, never a bare chunk/subchunk directory sitting directly under the
channel:

  in_volume_neutrinos/<channel>/<category>/<chunk>/event_<NNN>/
      unmatched_true_neutrinos_event<N>_Combined.png
      bee_link.txt        <- this ONE event's url, inside this (channel, category) BEE set

Only events with >=1 unmatched neutrino of that channel AND that category get a
directory -- a matched neutrino needs no picture, so nothing is drawn or saved
for it. An event whose neutrinos fail two different ways is filed once under
each category (same plot, so nothing new to look at, just findable from either).

BEE SETS -- see write_unmatched_bee_sets. Per CHANNEL:
  in_volume_neutrinos/<channel>/<category>/bee_link.txt
      one set per failure category under that channel (removed_by_cosmic_tagger,
      reco_outside_beam_window, ...) -- every event of that channel with that
      category, in one upload.
  in_volume_neutrinos/<channel>/bee_link_all_categories.txt
      one set of every unmatched event under that channel, any category.
There is no set spanning multiple channels -- a link always answers "this channel,
this category" or "this channel, everything".

Run directly: python investigate_unmatched_true_neutrinos.py
Output: multi_file_plots_charge_light_matching/unmatched_true_neutrino_investigation/
        <SAMPLE_NAME>/{timestamp}/   (SAMPLE_NAME = NuECC_Sample / NuMuCC_Sample / ...)
"""
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from readfiles import read_charge_light_files_for_event, flatten_mc_tree
from selections import (
    tag_reco_clusters,
    GroupClustersByID, build_true_points_charge_light,
    reassign_cluster_ID_true_charge_light, reassign_cluster_ID_reco,
    apply_energy_cutoff, apply_true_pointwise_energy_cutoff, apply_wire_readout_sensitive_yz_plane_cut_true,
    Fiducial_X_MIN, Fiducial_X_MAX, Fiducial_Y_MIN,
    Fiducial_Y_MAX, Fiducial_Z_MIN, Fiducial_Z_MAX,
    apply_wire_readout_sensitive_yz_plane_cut_reco,
    apply_deadarea_cut_true_charge_light,
)
from completeness_purity_estimate import EvaluateCompleteness, EvaluatePurity
from clusterpairmatching import MatchTrueToReco1to1
from metadata import (
    build_cluster_flash_metadata, build_img_cluster_flash_metadata,
    categorize_unmatched_true_neutrinos, NEUTRINO_CLUSTER_ID_BASE,
    # The vertex-volume split: the vertex records give each interaction its
    # in/out-of-volume flag, build_neutrino_volume_map turns those into a
    # {(event, true cluster id) -> 'in'|'out'} lookup, and filter_records_by_label
    # selects one population's rows. The neutrino rows here carry the same
    # (event, true_cluster_id) key, so the same filter serves them unchanged.
    build_neutrino_vertex_records, build_neutrino_volume_map, build_neutrino_channel_map,
    filter_records_by_label, restrict_label_map,
)
from writeinformation import write_unmatched_true_neutrino_info
from DrawRecoTrueClusters import (DrawUnmatchedTrueNeutrinos, DrawUnmatchedTrueNeutrinoBreakdown,
                                  DrawUnmatchedTrueNeutrinoPies,
                                  DrawUnmatchedSelectionEfficiency)
from DrawRecoTrueFlashes import (BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US,
                                  draw_unmatched_neutrino_flash_times)
from build_bee_set_from_links import build_population_bee_set

# ============================================================================
# CONFIG -- same selection/beam-window-cut settings as
# Evaluation_ChargeLightMatching_AfterBeamWindowCut.ipynb (cells 4 and 6) and as
# investigate_extra_reco_clusters.py, so counts here are directly comparable to
# both that notebook's job summary and the extra-reco investigation.
# ============================================================================
# nuecc: point at the staging tree (chunk_NN__subchunk_MM/data/<k>/ dirs staged
# by readfiles.stage_nuecc_chunks -- run SignalBackground_Distributions or a
# Draw_* CosmicTagger notebook first to stage). Set back to the Haiwang path for
# the tagger sample.
#
# SAMPLE_NAME names the per-sample output subdirectory under OUTPUT_DIR, so a
# later run over a different production lands beside this one rather than on top
# of it. Change PARENT_DIR and SAMPLE_NAME together:
#   nuecc  -> "img-clus-match-tag-pr-nuecc-1000file-2026-08-29/staging"      + "NuECC_Sample"
#   numucc -> "img-clus-match-tag-pr-mc-1000file-sync-2026-08-30/staging"    + "NuMuCC_Sample"
# The staging tree is built by readfiles.stage_nuecc_chunks (run a
# SignalBackground / Draw_* CosmicTagger notebook, or call it directly, on the
# source chunk first -- both samples share the one-event-per-zip layout).
PARENT_DIR  = Path("/Volumes/My Passport/Research_Life/Experiment/SBND/"
                   "Wirecell_Reconstruction/Samples/"
                   "img-clus-match-tag-pr-mc-1000file-sync-2026-08-30/staging")
SAMPLE_NAME = "NuMuCC_Sample"
# "first 10 chunks" -- chunk_00 through chunk_09, every subchunk of each (10
# subchunks/chunk, 10 events/subchunk => 100 files, 1000 events), combined into
# one run -- same scope as every other "first 10 chunks" job in this project.
FIRST_10_CHUNKS = [f"chunk_{c:02d}__subchunk_{s:02d}" for c in range(10) for s in range(10)]
FIRST_15_CHUNKS = [f"chunk_{c:02d}__subchunk_{s:02d}" for c in range(15) for s in range(10)]
CHUNK_00        = [f"chunk_00__subchunk_{s:02d}" for s in range(10)]   # one whole chunk (100 files)

TARGET_FILE = FIRST_15_CHUNKS   # "all" for every file subdirectory with a data/ folder,
                                # one name to test on, or a list of names (CHUNK_00 / FIRST_10_CHUNKS / FIRST_15_CHUNKS)
EVENT_LOW   = None    # None = auto-detect from each file's data/ (all events present)
EVENT_HIGH  = None    # exclusive; None = auto-detect
OUTPUT_DIR  = Path("multi_file_plots_charge_light_matching/unmatched_true_neutrino_investigation") / SAMPLE_NAME
APA_LABEL   = "Combined"

radius_completeness        = 2
radius_purity_xz         = 2
radius_purity_yz         = 5
radius_purity_xy         = 5
min_recopoints_threshold = 5
min_cluster_energy       = 100
min_true_point_energy    = 0.02   # MeV per true POINT -- see selections.py
x_min, x_max = Fiducial_X_MIN, Fiducial_X_MAX
y_min, y_max = Fiducial_Y_MIN, Fiducial_Y_MAX
z_min, z_max = Fiducial_Z_MIN, Fiducial_Z_MAX

# Apply the cosmic tagger cut to the beam-window reco set, so a neutrino the
# tagger removed is attributed to the TAGGER rather than counted as reconstructed.
# Without this the script measures the pipeline as it was before the tagger
# existed. Set False to get that older picture back.
APPLY_COSMIC_TAGGER_CUT = True

b_draw_event_level_plots = True   # per-event XZ/YZ/XY plots for events with >=1 unmatched true neutrino

# The populations every output is written for. 'all' keeps its outputs where they
# have always been (directly in the level's directory); the rest go in
# subdirectories, with the population named in the plot titles. Filenames are
# identical in all of them, so the same plot can be diffed between populations.
#
# Two axes, composed: WHERE the interaction happened (vertex in / out of the
# wire-readout sensitive box) and WHAT came out of it (numu CC / nue CC / NC, from
# metadata.classify_neutrino_interaction). The channel breakdown is done for the
# IN-VOLUME neutrinos: those are the ones fully inside the detector, so a failure
# there is a reconstruction statement about that channel rather than a statement
# about how much of the interaction happened to leak in.
#
# 'volume'/'channel' are the labels a row must carry to belong; None means that
# axis is not applied.
POPULATIONS = [
    # IN-VOLUME, BY INTERACTION CHANNEL ONLY -- see the module docstring. 'subdir'
    # is now flat (just the channel name): it is both the pie/info.txt directory
    # AND the top directory the per-event spatial plots nest under (each
    # category, then chunk/event, under it), so a reader who wants "every nue CC
    # failure" opens exactly one directory for both.
    {'key': 'in_numu_CC', 'volume': 'in',  'channel': 'numu_CC',
     'subdir': Path("numu_CC"), 'label': 'vertex in volume, numu CC'},
    {'key': 'in_nue_CC',  'volume': 'in',  'channel': 'nue_CC',
     'subdir': Path("nue_CC"), 'label': 'vertex in volume, nue CC'},
    {'key': 'in_NC',      'volume': 'in',  'channel': 'NC',
     'subdir': Path("NC"), 'label': 'vertex in volume, NC'},
]


def population_rows(neutrino_rows, population, volume_map, channel_map):
    """
    The rows belonging to one population. Nothing is recategorized -- the rows
    were built once against the full reco set, this only selects which of them a
    given directory shows.
    """
    if population['channel'] is not None:
        # Channel labels restricted to this volume, so the two axes compose.
        composed = restrict_label_map(channel_map, volume_map, population['volume'])
        return filter_records_by_label(neutrino_rows, composed, population['channel'])
    if population['volume'] is not None:
        return filter_records_by_label(neutrino_rows, volume_map, population['volume'])
    return list(neutrino_rows)


def find_input_files():
    """Same discovery rule as investigate_extra_reco_clusters.py's find_input_files --
    kept local since this script is standalone by design. TARGET_FILE may be
    "all", one file name, or a list of file names (e.g. FIRST_10_CHUNKS)."""
    if TARGET_FILE == "all":
        return [d.name for d in sorted(PARENT_DIR.iterdir())
                if d.is_dir() and (d / "data").is_dir()]
    if isinstance(TARGET_FILE, (list, tuple)):
        return list(TARGET_FILE)
    return [TARGET_FILE]


def find_events(file_name):
    """Same as investigate_extra_reco_clusters.py's find_events -- event numbering
    is per-file, so a fixed range can't serve every file."""
    if EVENT_LOW is not None and EVENT_HIGH is not None:
        return list(range(EVENT_LOW, EVENT_HIGH))

    data_dir = PARENT_DIR / file_name / "data"
    if not data_dir.is_dir():
        return []
    events = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            events.append(int(item.name))
    return sorted(events)


def group_reco_with_provenance(predicted_points):
    """
    reassign_cluster_ID_reco() + GroupClustersByID() in one pass, additionally
    returning where each grouped cluster came from.

    Needed because this investigation has to compare the PRE- and POST-beam-window-cut
    reco sets cluster by cluster, and reassign_cluster_ID_reco replaces the raw
    real_cluster_id with the cluster's rounded mean X -- which discards exactly
    the ID the beam-window cut and the flash records are keyed on. The grouping
    itself is unchanged from the notebook's path: the beam-window cut removes
    WHOLE clusters (it filters on real_cluster_id), so a surviving cluster
    contains the same points, and therefore gets the same mean-X ID, whether the
    cut is applied before or after this grouping.

    Args:
        predicted_points: Nx5 array [x, y, z, real_cluster_id, q], already
            YZ-plane-cut

    Returns:
        (clusters, provenance) -- clusters is {mean_x_id: [point, ...]} exactly as
        GroupClustersByID would return it, provenance is {mean_x_id: [real_cluster_id, ...]}
        (more than one only in the rare case of two raw clusters whose mean X
        agrees to 3 decimals, which the notebook's path would also merge).
    """
    by_real_id = {}
    for point in predicted_points:
        by_real_id.setdefault(point[3], []).append(point)

    clusters, provenance = {}, {}
    for real_id, points in by_real_id.items():
        points = np.array(points)
        new_id = round(float(np.mean(points[:, 0])), 3)
        points[:, 3] = new_id
        clusters.setdefault(new_id, []).extend(list(points))
        provenance.setdefault(new_id, []).append(real_id)
    return clusters, provenance


def render_level_outputs(neutrino_rows, volume_map, channel_map, n_selected_reco, level_dir,
                         level_name, filename_prefix, file_name=None,
                         clusters_true=None, clusters_reco_all=None, event=None,
                         draw=True, always_write_breakdown=True,
                         event_plot_root=None, spatial_plot_entries=None, spatial_only=False):
    """
    Every output of one level (event, file or job), written once per vertex-volume
    population: all true neutrinos, then the in-volume and out-of-volume subsets.

    Nothing is recategorized -- the rows were built once against the full reco set
    and are only filtered here, so a neutrino's category is the same in whichever
    copy it appears in. n_selected_reco likewise stays the FULL beam-window reco
    count in every copy: that is the population all of these neutrinos were
    matched against, and scaling it per subset would invent a number the matching
    never used.

    Parameters:
    - neutrino_rows: categorize_unmatched_true_neutrinos() rows for this level
    - volume_map / channel_map: build_neutrino_volume_map() and
      build_neutrino_channel_map() output covering those rows
    - n_selected_reco: beam-window reco clusters at this level (top-panel bar)
    - level_dir: the level's output directory; 'all' writes here, the other two
      into subdirectories of it
    - level_name, filename_prefix, file_name: drawer conventions, unchanged
    - clusters_true / clusters_reco_all / event: event level only -- when given,
      the per-event XZ/YZ/XY spatial plot is drawn too
    - draw: False writes the text tables and skips the plots
    - always_write_breakdown: the breakdown chart is drawn even when nothing is
      unmatched (event level does this: "all matched" is a result worth seeing);
      the info table and spatial/flash plots still need >=1 unmatched row
    - event_plot_root: when given (the in-volume-neutrinos summary dir, event
      level only), the per-event spatial plot is written to
      event_plot_root/<channel>/<category>/<file_name>/event_<NNN>/ instead of
      pop_dir -- see the module docstring's PER-EVENT SPATIAL PLOTS section.
    - spatial_plot_entries: a list this function APPENDS to (in place) with one
      dict per spatial plot actually drawn -- {'chunk', 'event', 'channel',
      'category', 'plot_dir'} -- so the caller can write a per-event BEE link
      into plot_dir once the BEE sets exist (built after every event is
      processed).
    - spatial_only: True skips the breakdown chart, pies, efficiency curves,
      info.txt and flash-time plot entirely -- only the per-event spatial plot
      (and its spatial_plot_entries record) is produced. Event level uses this:
      those per-population summaries are only wanted once, aggregated, at job
      level -- a per-event copy under the file's own directory tree is exactly
      the "chunk/subchunk in the head directory" clutter that was removed.

    Returns {population key: number of unmatched rows in that population}.
    """
    unmatched_by_population = {}

    for population in POPULATIONS:
        pop_key = population['key']
        pop_rows = population_rows(neutrino_rows, population, volume_map, channel_map)
        unmatched_by_population[pop_key] = sum(1 for r in pop_rows if r['category'] != 'matched')

        # An empty subset means this level has no neutrino of that kind -- skip it
        # rather than create a directory of empty plots. The 'all' population is
        # never skipped: its directory is the level's own.
        if population['subdir'] is not None and not pop_rows:
            continue

        # level_dir is None for the spatial-only event-level call (no per-event
        # directory tree of its own any more): pop_dir is then never actually
        # used (event_plot_root always wins below), so it stays None too rather
        # than erroring on None / population['subdir'].
        if level_dir is None:
            pop_dir = None
        elif population['subdir'] is None:
            pop_dir = level_dir
        else:
            pop_dir = level_dir / population['subdir']
        pop_level_name = level_name if not population['label'] else f"{level_name} ({population['label']})"

        if not spatial_only:
            if draw and (always_write_breakdown or unmatched_by_population[pop_key] > 0):
                DrawUnmatchedTrueNeutrinoBreakdown(pop_rows, n_selected_reco, pop_dir, APA_LABEL,
                                                    pop_level_name, filename_prefix, file_name=file_name)
                # The same two splits as the bar chart's lower panels, as pies, in
                # their own files -- see DrawUnmatchedTrueNeutrinoPies.
                DrawUnmatchedTrueNeutrinoPies(pop_rows, pop_dir, APA_LABEL,
                                              pop_level_name, filename_prefix, file_name=file_name)
                # Efficiency curves at JOB level only: a 200 MeV bin holds one or two
                # interactions in a single event, so per-event and per-file copies
                # would be noise with error bands wider than the axis.
                if level_name.lower().startswith('job'):
                    DrawUnmatchedSelectionEfficiency(pop_rows, pop_dir, APA_LABEL,
                                                     pop_level_name, filename_prefix,
                                                     file_name=file_name)

            if unmatched_by_population[pop_key] > 0:
                write_unmatched_true_neutrino_info(pop_rows, pop_dir)
                if draw:
                    draw_unmatched_neutrino_flash_times(pop_rows, pop_dir, APA_LABEL,
                                                         pop_level_name, filename_prefix, file_name=file_name)

        if (draw and unmatched_by_population[pop_key] > 0
                and clusters_true is not None and event is not None):
            # event_plot_root redirects the spatial plot under the job's
            # summary tree, one directory per unmatched category the event's
            # neutrinos fall into: <event_plot_root>/<channel>/<category>/
            # <chunk>/event_<NNN>/. Usually one category; an event whose
            # neutrinos fail two different ways gets the same plot filed under
            # each, so browsing either category shows it.
            categories_present = sorted({r['category'] for r in pop_rows
                                         if r['category'] != 'matched'})
            for category in categories_present:
                if event_plot_root is not None and population['subdir'] is not None:
                    plot_dir = (Path(event_plot_root) / population['subdir']
                               / category / file_name / f"event_{event:03d}")
                else:
                    plot_dir = pop_dir
                # Full cluster dicts on purpose: the drawer indexes into
                # them by the ids on the rows it was given.
                DrawUnmatchedTrueNeutrinos(clusters_true, pop_rows, event, APA_LABEL, plot_dir,
                                            file_name=file_name, clusters_reco_all=clusters_reco_all)
                if spatial_plot_entries is not None:
                    spatial_plot_entries.append({
                        'chunk': file_name, 'event': event,
                        'channel': population['subdir'].name, 'category': category,
                        'plot_dir': plot_dir,
                    })

    return unmatched_by_population


BUILD_BEE_SET = True   # build + upload the unmatched BEE sets at job level


def write_unmatched_bee_sets(spatial_plot_entries, in_volume_dir):
    """
    BEE sets for the in-volume unmatched population, built from
    spatial_plot_entries (one dict per {chunk, event, channel, category, plot_dir}
    that render_level_outputs actually drew a spatial plot for) rather than from
    job_rows directly, so the sets line up exactly with the directory tree those
    plots were written into: in_volume_neutrinos/<channel>/<category>/<chunk>/
    event_<NNN>/ -- see the module docstring's BEE SETS section.

    Builds, per CHANNEL (numu_CC / nue_CC / NC):
      - one BEE set per unmatched CATEGORY under that channel (every event of
        that channel with that category, in one upload) -> bee_link.txt inside
        in_volume_neutrinos/<channel>/<category>/
      - one BEE set of every unmatched event under that channel, any category
        -> bee_link_all_categories.txt inside in_volume_neutrinos/<channel>/
    Then, into every event directory in spatial_plot_entries, a bee_link.txt with
    that one event's url inside its (channel, category) set -- the directory
    already names the category unambiguously, so this is a single link, not a
    per-category list.

    Uses build_bee_set_from_links.build_population_bee_set() (same helper the
    CosmicTagger notebooks use for their single-population links -- see
    feedback_single_bee_link_per_population) rather than the old subprocess+regex
    path, whose chunk(\\d+)_event(\\d+) regex cannot parse this sample's
    chunk_00__subchunk_00 directory names.

    Set BUILD_BEE_SET = False to skip every upload -- nothing is built and no
    bee_link.txt files are written anywhere in that case.
    """
    in_volume_dir = Path(in_volume_dir)
    if not spatial_plot_entries:
        print("\nNo unmatched in-volume true neutrinos -- no BEE sets built")
        return
    print(f"\nUnmatched BEE sets: {len(spatial_plot_entries)} spatial plot(s), by channel/category:")
    if not BUILD_BEE_SET:
        print("  BUILD_BEE_SET is False -- skipping every upload")
        return

    by_channel_category = {}   # (channel, category) -> [entry, ...]
    by_channel = {}            # channel -> [entry, ...] (categories mixed)
    for e in spatial_plot_entries:
        by_channel_category.setdefault((e['channel'], e['category']), []).append(e)
        by_channel.setdefault(e['channel'], []).append(e)

    def _bee_entries(entries, label):
        return [{'chunk': e['chunk'], 'event': int(e['event']), 'path': f"{e['chunk']}_event{e['event']}_{label}.png"}
                for e in entries]

    # One set per (channel, category) -> bee_link.txt in that category's own directory.
    per_event_by_cc = {}   # (channel, category) -> {(chunk, evt): url}
    for (channel, category), entries in sorted(by_channel_category.items()):
        bee_entries = _bee_entries(entries, category)
        n_events = len({(e['chunk'], e['event']) for e in bee_entries})
        print(f"  {channel}/{category}: {n_events} event(s) ...", flush=True)
        cat_dir = in_volume_dir / channel / category
        url = build_population_bee_set(bee_entries, PARENT_DIR, cat_dir / "bee_set", f"{channel}/{category}")
        if not url:
            print("    BEE set build/upload failed -- skipped")
            continue
        per_event_by_cc[(channel, category)] = {(e['chunk'], e['event']): e['bee_url'] for e in bee_entries}
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / 'bee_link.txt').write_text(
            f"BEE SET URL: {url}\n\n{n_events} unmatched event(s), category '{category}', channel {channel}.\n")
        print(f"    {url}")

    # One set per channel, every category -> bee_link_all_categories.txt in the channel's directory.
    for channel, entries in sorted(by_channel.items()):
        seen, dedup = set(), []
        for e in entries:
            key = (e['chunk'], int(e['event']))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(e)
        bee_entries = _bee_entries(dedup, "unmatched")
        print(f"  {channel} (all categories): {len(bee_entries)} event(s) ...", flush=True)
        chan_dir = in_volume_dir / channel
        url = build_population_bee_set(bee_entries, PARENT_DIR, chan_dir / "bee_set_all_categories",
                                       f"{channel} all categories")
        if not url:
            print("    BEE set build/upload failed -- skipped")
            continue
        chan_dir.mkdir(parents=True, exist_ok=True)
        (chan_dir / 'bee_link_all_categories.txt').write_text(
            f"BEE SET URL: {url}\n\n{len(bee_entries)} unmatched event(s), every category, channel {channel}.\n")
        print(f"    {url}")

    # Per-event bee_link.txt, inside the directory render_level_outputs already
    # drew that event's spatial plot into -- one link, from its own (channel,
    # category) set, since the directory already pins down which category this is.
    n_written = 0
    for entry in spatial_plot_entries:
        key = (entry['channel'], entry['category'])
        url = per_event_by_cc.get(key, {}).get((entry['chunk'], int(entry['event'])))
        if not url:
            continue
        plot_dir = Path(entry['plot_dir'])
        plot_dir.mkdir(parents=True, exist_ok=True)
        (plot_dir / 'bee_link.txt').write_text(f"{entry['category']}: {url}\n")
        n_written += 1
    print(f"  bee_link.txt written into {n_written} event directory(ies)")


def process_event(input_dir, file_name, evt):
    """
    Run one event through the same selection + beam-window-cut pipeline as
    Evaluation_ChargeLightMatching_AfterBeamWindowCut.ipynb's cell 6, but keep
    the PRE-cut reco set alongside the post-cut one, then categorize every true
    neutrino cluster.

    Returns (clusters_true, clusters_reco, clusters_reco_all, neutrino_rows,
    vertex_records) or None if the event's files are missing.
    """
    result = read_charge_light_files_for_event(input_dir, evt)
    if result is None:
        return None

    event_key = f"{file_name}_{evt}"

    # --- True side (clustering-level: sed-smear, paired with clustering-global) ---
    x_true, y_true, z_true, id_true, q_true, real_id_true, e_true, nu_idx_true = result['true_clustering']
    true_points = build_true_points_charge_light(x_true, y_true, z_true, real_id_true, q_true,
                                                  energy=e_true, nu_idx=nu_idx_true)
    true_points = reassign_cluster_ID_true_charge_light(true_points)
    # POINT-wise first, so the cluster total the cluster cut tests is the
    # total of the points that survive. 0.01 MeV -- see selections.py.
    true_points = apply_true_pointwise_energy_cutoff(true_points, min_true_point_energy)
    true_points = apply_energy_cutoff(true_points, min_cluster_energy)
    true_points = apply_wire_readout_sensitive_yz_plane_cut_true(true_points)
    true_points = apply_deadarea_cut_true_charge_light(true_points, output_dir=None, event=evt, file_name=file_name)
    clusters_true = GroupClustersByID(true_points) if len(true_points) else {}

    # --- IMG-LEVEL true/reco (pre charge-light-matching): sed-sce truth +
    # img-global reco. Keyed by the same 99990+nu_idx, put through the same
    # true-side cut chain as clusters_true so a neutrino present in one is
    # present in the other. metadata.categorize_unmatched_true_neutrinos uses
    # these to tell a charge-light X-shift failure from a reconstruction gap. ---
    xi_t, yi_t, zi_t, _idi_t, qi_t, ridi_t, ei_t, nui_t = result['true']
    img_true_points = build_true_points_charge_light(xi_t, yi_t, zi_t, ridi_t, qi_t,
                                                     energy=ei_t, nu_idx=nui_t)
    img_true_points = reassign_cluster_ID_true_charge_light(img_true_points)
    img_true_points = apply_true_pointwise_energy_cutoff(img_true_points, min_true_point_energy)
    img_true_points = apply_energy_cutoff(img_true_points, min_cluster_energy)
    img_true_points = apply_wire_readout_sensitive_yz_plane_cut_true(img_true_points)
    img_true_points = apply_deadarea_cut_true_charge_light(img_true_points, output_dir=None,
                                                          event=evt, file_name=file_name)
    clusters_img_true = GroupClustersByID(img_true_points) if len(img_true_points) else {}

    xi_r, yi_r, zi_r, _idi_r, qi_r, ridi_r = result['reco']
    img_reco_points = np.column_stack((xi_r, yi_r, zi_r, ridi_r, qi_r))
    img_reco_points = apply_wire_readout_sensitive_yz_plane_cut_reco(img_reco_points)
    clusters_img_reco = GroupClustersByID(img_reco_points) if len(img_reco_points) else {}

    # --- Reco side: flash association first, so the beam-window cut can be
    # applied as a SELECTION over the full set rather than as a filter that
    # throws the rejected clusters away -- those rejects are the evidence this
    # investigation needs.
    cluster_flash_records = build_cluster_flash_metadata(result['op'], file_name, evt, APA_LABEL, event_key=event_key)
    img_cluster_flash_records = build_img_cluster_flash_metadata(
        result['reco'], result['clustering'], cluster_flash_records, file_name, evt, APA_LABEL, event_key=event_key)
    clu_beam_window_ids = {float(r['clustering_cluster_id']) for r in img_cluster_flash_records
                            if BEAM_WINDOW_MIN_US <= r['flash_time'] <= BEAM_WINDOW_MAX_US}
    flash_times_by_real_id = {}
    flash_indices_by_real = {}
    for r in img_cluster_flash_records:
        flash_times_by_real_id.setdefault(float(r['clustering_cluster_id']), []).append(r['flash_time'])
        flash_indices_by_real.setdefault(float(r['clustering_cluster_id']), set()).add(r['flash_index'])

    x_clu, y_clu, z_clu, id_clu, q_clu, real_id_clu = result['clustering']
    predicted_points = np.column_stack((x_clu, y_clu, z_clu, real_id_clu, q_clu))
    predicted_points = apply_wire_readout_sensitive_yz_plane_cut_reco(predicted_points)

    if len(predicted_points) == 0:
        clusters_reco_all, reco_provenance, clusters_reco = {}, {}, {}
    else:
        clusters_reco_all, reco_provenance = group_reco_with_provenance(predicted_points)
        clusters_reco = {cid: points for cid, points in clusters_reco_all.items()
                          if any(rid in clu_beam_window_ids for rid in reco_provenance[cid])}

    # --- COSMIC TAGGER CUT, on the beam-window survivors ---
    # The reco ids here are group_reco_with_provenance's, one per avg-X group,
    # NOT clustering-global's coarse cluster_id -- so flash-mates are separate
    # clusters and the per-flash tag has to be told which of them share a flash,
    # or it would only ever remove the directly-tagged one. The group key is the
    # set of IN-WINDOW flash indices a cluster's real ids carry; clusters with no
    # in-window flash are left out of the map entirely, so each is its own group
    # rather than all of them sharing a "no flash" bucket.
    # Both initialised here: an event with no in-beam clusters skips the branch
    # below entirely, and the row loop reads them unconditionally.
    tagger_removed_ids = set()
    tagger_names_by_cluster = {}
    if APPLY_COSMIC_TAGGER_CUT and clusters_reco:
        in_window_index_by_real = {}
        for r in img_cluster_flash_records:
            if BEAM_WINDOW_MIN_US <= r['flash_time'] <= BEAM_WINDOW_MAX_US:
                in_window_index_by_real.setdefault(
                    float(r['clustering_cluster_id']), set()).add(r['flash_index'])
        flash_group_by_cluster = {}
        for cid in clusters_reco:
            indices = set()
            for rid in reco_provenance.get(cid, []):
                indices |= in_window_index_by_real.get(rid, set())
            if indices:
                flash_group_by_cluster[cid] = frozenset(indices)
        tagged = tag_reco_clusters(result.get('taggers'), clusters_reco,
                                   flash_group_by_cluster=flash_group_by_cluster)
        tagger_removed_ids = set(tagged)
        # WHICH tagger, per cluster -- the figures name it, and "stm" vs "tgm"
        # is the difference between a stopping muon and a through-going one.
        # Propagated entries carry the names of whatever was tagged in their
        # flash group, flagged so the plot can say it was not tagged directly.
        tagger_names_by_cluster = {cid: (e.get('taggers') or [], e.get('tagged_directly', False))
                                   for cid, e in tagged.items()}
        clusters_reco = {cid: pts for cid, pts in clusters_reco.items()
                         if cid not in tagger_removed_ids}

    # FLASH MATES. All reco activity on one flash is one bundle -- that is the
    # premise the whole beam-window selection rests on -- so a cluster's
    # flash-mates are part of the same activity and belong in the same picture.
    # Built over the PRE-cut set, since the clusters of interest here are exactly
    # the ones a cut removed.
    flash_indices_by_reco = {}
    for cid, reals in reco_provenance.items():
        idx = set()
        for rid in reals:
            idx |= flash_indices_by_real.get(float(rid), set())
        if idx:
            flash_indices_by_reco[cid] = idx
    flash_mates_by_reco = {}
    for cid, idx in flash_indices_by_reco.items():
        mates = sorted(other for other, other_idx in flash_indices_by_reco.items()
                       if other != cid and (idx & other_idx))
        if mates:
            flash_mates_by_reco[cid] = mates

    completeness_results = EvaluateCompleteness(clusters_true, clusters_reco, event_key, radius_completeness, min_recopoints_threshold)
    purity_results     = EvaluatePurity(clusters_true, clusters_reco, event_key, radius_purity_xz, radius_purity_yz, radius_purity_xy)
    matched_pairs      = MatchTrueToReco1to1(completeness_results, purity_results)

    neutrino_rows = categorize_unmatched_true_neutrinos(
        clusters_true, clusters_reco, clusters_reco_all, reco_provenance,
        clu_beam_window_ids, flash_times_by_real_id, matched_pairs,
        file_name, evt, apa=APA_LABEL, event_key=event_key,
        radius_completeness=radius_completeness, min_recopoints_threshold=min_recopoints_threshold,
        tagger_removed_ids=tagger_removed_ids,
        radius_purity_xz=radius_purity_xz, radius_purity_yz=radius_purity_yz,
        radius_purity_xy=radius_purity_xy,
        clusters_img_true=clusters_img_true, clusters_img_reco=clusters_img_reco)

    # --- Interaction vertices (mc.json), for the in/out-of-volume split ---
    # Same builder and same bounds as the evaluation notebook, so "in volume"
    # means exactly what it means there. The flag is copied onto each neutrino row
    # as well, so unmatched_true_neutrino_info.txt can show it per interaction
    # without the reader having to cross-reference another table.
    vertex_records = build_neutrino_vertex_records(
        flatten_mc_tree(result['mc']), clusters_true, file_name, evt, event_key,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)
    volume_by_cluster  = {r['cluster_id']: r['vertex_in_volume'] for r in vertex_records}
    channel_by_cluster = {r['cluster_id']: r['interaction_channel'] for r in vertex_records}
    # The interaction VERTEX, for the truth panels. mc.json's root start_xyz, in
    # the same cm frame as the true points.
    vertex_xyz_by_cluster = {}
    for r in vertex_records:
        vx, vy, vz = r.get('vertex_x'), r.get('vertex_y'), r.get('vertex_z')
        if None not in (vx, vy, vz):
            vertex_xyz_by_cluster[r['cluster_id']] = (vx, vy, vz)
    for row in neutrino_rows:
        row['vertex_in_volume']    = volume_by_cluster.get(row['true_cluster_id'])
        row['interaction_channel'] = channel_by_cluster.get(row['true_cluster_id'])
        row['vertex_xyz']          = vertex_xyz_by_cluster.get(row['true_cluster_id'])

        # WHICH tagger removed this neutrino's cluster, and whether it was tagged
        # on its own points or inherited the tag from a flash-mate. Only set for
        # the tagger category, where it is the answer to "why".
        evidence_cid = row.get('best_strict_reco_cluster_id')
        names, direct = tagger_names_by_cluster.get(evidence_cid, ([], None))
        row['tagger_names']    = list(names)
        row['tagger_direct']   = direct
        # Other reco clusters on the SAME FLASH as the evidence cluster: the rest
        # of the bundled in-beam activity, typically a coincident cosmic.
        row['flash_mate_reco_ids'] = list(flash_mates_by_reco.get(evidence_cid, []))
        # wrong_charge_light_matching's evidence cluster is the YZ-aligned one
        # when there is no 3D overlap to point at (drift-shifted off the truth).
        if (row.get('category') == 'wrong_charge_light_matching'
                and not row.get('best_strict_reco_cluster_id')):
            yz_cid = row.get('yz_best_reco_cluster_id')
            row['flash_mate_reco_ids'] = list(flash_mates_by_reco.get(yz_cid, []))

    return clusters_true, clusters_reco, clusters_reco_all, neutrino_rows, vertex_records


def main():
    start_time = time.time()
    start_stamp = datetime.now()

    input_files = find_input_files()
    if not input_files:
        print(f"No input files found in {PARENT_DIR}")
        return

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    in_volume_dir = output_dir / "in_volume_neutrinos"

    job_rows = []
    job_vertex_records = []
    job_selected_reco = 0
    events_processed = 0
    events_with_unmatched = 0
    # One entry per per-event spatial plot actually drawn (any channel, any
    # chunk) -- render_level_outputs appends to this in place. Used after the
    # whole job to write each event's bee_link.txt once the BEE sets exist.
    spatial_plot_entries = []

    for file_name in input_files:
        events = find_events(file_name)
        print(f"{file_name}: {len(events)} event(s) to process", flush=True)

        for evt in events:
            processed = process_event(PARENT_DIR / file_name, file_name, evt)
            if processed is None:
                continue
            clusters_true, clusters_reco, clusters_reco_all, neutrino_rows, vertex_records = processed

            job_rows.extend(neutrino_rows)
            job_vertex_records.extend(vertex_records)
            job_selected_reco  += len(clusters_reco)
            events_processed += 1

            n_unmatched = sum(1 for r in neutrino_rows if r['category'] != 'matched')
            event_volume_map = build_neutrino_volume_map(vertex_records)
            n_in  = sum(1 for r in neutrino_rows
                        if event_volume_map.get((r['event'], r['true_cluster_id'])) == 'in')
            print(f"  {file_name}_{evt}: {len(neutrino_rows)} true neutrino(s) "
                  f"({n_in} in volume, {len(neutrino_rows) - n_in} out), "
                  f"{len(clusters_reco)}/{len(clusters_reco_all)} reco in beam window, "
                  f"{n_unmatched} unmatched", flush=True)

            if n_unmatched > 0:
                events_with_unmatched += 1

            # spatial_only: no per-event/per-file directory tree of its own any
            # more (see the module docstring) -- this call exists only to find
            # which events have an unmatched neutrino and draw its spatial plot,
            # under in_volume_dir, via event_plot_root.
            render_level_outputs(
                neutrino_rows, event_volume_map, build_neutrino_channel_map(vertex_records),
                len(clusters_reco),
                None, "Event Level", file_name,
                file_name=file_name,
                clusters_true=clusters_true, clusters_reco_all=clusters_reco_all, event=evt,
                draw=b_draw_event_level_plots, spatial_only=True,
                event_plot_root=in_volume_dir, spatial_plot_entries=spatial_plot_entries)

    if job_rows or job_selected_reco:
        job_volume_map = build_neutrino_volume_map(job_vertex_records)
        render_level_outputs(job_rows, job_volume_map, build_neutrino_channel_map(job_vertex_records),
                             job_selected_reco, in_volume_dir, "Job Level", "alljobs")
        write_unmatched_bee_sets(spatial_plot_entries, in_volume_dir)

    categories = ['matched', 'wrong_charge_light_matching', 'reco_outside_beam_window',
                  'broken_or_sparse_reco', 'no_reco_overlap', 'unexplained']
    if APPLY_COSMIC_TAGGER_CUT:
        categories.insert(2, 'removed_by_cosmic_tagger')
    job_volume_map  = build_neutrino_volume_map(job_vertex_records)
    job_channel_map = build_neutrino_channel_map(job_vertex_records)
    rows_by_population = {p['key']: population_rows(job_rows, p, job_volume_map, job_channel_map)
                          for p in POPULATIONS}

    print(f"\n{'='*70}")
    print(f"Events processed: {events_processed} across {len(input_files)} file(s)")
    print(f"Events with >=1 unmatched true neutrino: {events_with_unmatched}")
    print(f"Total selected reco clusters (beam window, post cuts): {job_selected_reco}")
    print()
    # Same numbers as before, now with the two vertex-volume columns beside the
    # total. 'in' + 'out' can fall short of 'all' by any interaction with no
    # volume flag (no vertex in mc.json) -- none in the current dataset.
    columns = [(p['key'], p['label'] or 'all') for p in POPULATIONS]
    print(f"{'category':<26}" + "".join(f"{name:>30}" for _, name in columns))
    print(f"{'true neutrino clusters':<26}"
          + "".join(f"{len(rows_by_population[key]):>30}" for key, _ in columns))
    for cat in categories:
        print(f"  {cat + ':':<24}"
              + "".join(f"{sum(1 for r in rows_by_population[key] if r['category'] == cat):>30}"
                        for key, _ in columns))
    print(f"\nOutput written to: {output_dir}")
    for population in POPULATIONS:
        where = "in_volume_neutrinos/" if population['subdir'] is None \
            else f"in_volume_neutrinos/{population['subdir']}/"
        print(f"  {(population['label'] or 'all true neutrinos'):<28}: {where}")

    # How long the job took, for the overnight/multi-chunk runs where that isn't
    # otherwise visible anywhere -- see feedback_periodic_job_status.
    end_stamp = datetime.now()
    elapsed_s = time.time() - start_time
    hours, rem = divmod(int(elapsed_s), 3600)
    minutes, seconds = divmod(rem, 60)
    duration = (f"{hours}h {minutes}m {seconds}s" if hours
                else f"{minutes}m {seconds}s")
    summary_lines = [
        f"Started:  {start_stamp:%Y-%m-%d %H:%M:%S}",
        f"Finished: {end_stamp:%Y-%m-%d %H:%M:%S}",
        f"Duration: {duration}  ({elapsed_s:.1f} s)",
        "",
        f"Input file(s): {len(input_files)}  ({', '.join(input_files) if len(input_files) <= 10 else input_files[0] + ' ... ' + input_files[-1]})",
        f"Events processed: {events_processed}",
        f"Events with >=1 unmatched true neutrino: {events_with_unmatched}",
        f"Total true neutrino clusters: {len(job_rows)}",
        f"Total selected reco clusters (beam window, post cuts): {job_selected_reco}",
    ]
    (output_dir / 'summary.txt').write_text("\n".join(summary_lines) + "\n")
    print(f"\nJob duration: {duration}")
    print(f"  {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
