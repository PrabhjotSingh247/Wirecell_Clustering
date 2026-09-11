"""
SELECTION PERFORMANCE -- driven by
AnalysisDistributions/SignalBackground_Distributions.ipynb.

Reconstruction performance judged in RECO SPACE: every selected reco cluster gets
a category, using truth only to assign it. Where draw_signal_background.py stacks
TRUE clusters, everything here counts RECO clusters, so the totals are what a
selection would actually return.

THE CATEGORISATION, reco-centric

    for each reco cluster surviving the beam-window cut:
        matches = true NEUTRINO clusters with purity >= MIN_MATCH_PURITY
        no matches                      -> cosmic                        [terminal]
        else pair with argmax(purity)
            paired true vertex NOT in volume -> out_of_volume            [terminal]
            paired true vertex in volume:
                completeness > 0.8 AND purity > 0.8 -> high_signal_<channel>
                otherwise                           -> contaminated

RECO-CENTRIC, and by PURITY. This is not clusterpairmatching.MatchTrueToReco1to1,
which runs the other way (each TRUE cluster takes its best reco, chosen by
completeness). That function is left alone -- the evaluation notebooks depend on
it. Here the question is "what is this reco cluster mostly made of", which is
purity, asked once per reco cluster, because the population being categorised is
the reco one.

A true cluster may therefore be claimed by more than one reco cluster. That is
allowed and is not deduplicated: each reco cluster gets exactly one label, which
is what a reco-space stack needs. It does have a consequence for efficiency --
see build_selection_efficiency.

WHY A PURITY FLOOR. EvaluatePurity records any overlap above zero, so a reco
cluster sharing 0.5% of its charge with a neutrino would be "matched" and would
leave the cosmic category on the strength of a few points. MIN_MATCH_PURITY makes
that boundary a deliberate number rather than an accident of the KDTree radius.

THE CATEGORIES PARTITION THE SELECTED RECO SET. Every selected cluster lands in
exactly one, so their counts must sum to the number of clusters that survived the
beam-window cut. draw_reco_selection_stack draws that total as an outline over
the stack AND checks it, because a stack that silently loses clusters looks
exactly like a stack that does not.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import beta
from matplotlib.ticker import MaxNLocator, MultipleLocator
from pathlib import Path

from draw_variables import NEUTRINO_CLUSTER_ID_BASE
from draw_signal_background import (
    set_fitted_title, RECO_WORK_FUNCTION_EV,
    RECO_RECOMBINATION_FACTOR, RECO_MEV_PER_CHARGE,
    ENERGY_AXIS_MAX_MEV, ENERGY_AXIS_TICK_MEV, ENERGY_BIN_WIDTH_MEV,
    energy_bin_edges, count_overflow, reco_cluster_energy_mev,
    RECO_WORK_FUNCTION_EV, RECO_RECOMBINATION_FACTOR,
    _AXIS_LABEL_FONTSIZE, _TITLE_FONTSIZE, _TICK_LABEL_FONTSIZE, _LEGEND_FONTSIZE,
    _Y_HEADROOM_LINEAR,
)

# A reco cluster must own at least this fraction of its charge in common with a
# true neutrino cluster before the overlap counts as a match. Below it, the
# cluster stays 'cosmic'. See the module docstring.
MIN_MATCH_PURITY = 0.05

# Both must be EXCEEDED for a pair to be high signal.
HIGH_SIGNAL_THRESHOLD = 0.8

# A matched pair whose completeness AND purity are both below this is treated as a
# COSMIC, not as a contaminated signal candidate.
#
# The reasoning: a reco cluster that holds under 10% of a neutrino's energy and is
# itself under 10% neutrino has essentially nothing to do with that neutrino. The
# match exists because the two touch somewhere, not because the cluster
# reconstructs the interaction. Calling it contaminated signal would put it in a
# band that is supposed to mean "we found this neutrino badly" when the honest
# statement is "we did not find it".
#
# BOTH must be below: a cluster that is 5% pure but 90% complete has genuinely
# captured the neutrino inside a larger cluster, and one that is 95% pure but 3%
# complete is a real fragment of it. Either is a reconstruction failure worth
# keeping in the signal accounting; neither is a cosmic.
COSMIC_MAX_COMPLETENESS_PURITY = 0.10

# Where the reco-energy and efficiency axes STOP, in MeV. Only a display limit:
# the histograms are still binned over the full ENERGY_AXIS_MAX_MEV range and the
# tables and ROOT files still hold every entry, so nothing is lost -- the axis just
# stops showing a long empty tail. The count beyond it is printed when a figure is
# drawn, so "not shown" never becomes "not there".
PLOT_X_MAX_MEV = 3000.0

# A true interaction joins the efficiency DENOMINATOR only if its cluster
# deposited more than this, in MeV.
#
# 0.0 -- the notebook's cluster-level min_cluster_energy (Apply_energy_cutoff) is
# OFF: only the per-POINT floor (min_true_point_energy, 0.02 MeV) still applies,
# so a true cluster's energy here can be any positive number, not just >= 100.
# This USED to match a 100 MeV min_cluster_energy, for the reason a value here
# still has to: an interaction depositing less than what the notebook's cluster
# cut allows to survive could never have been found by any selection, and
# leaving it in the denominator would charge the selection for a loss that is
# the cut's, not its own. With that cut off, nothing is excluded on those
# grounds any more.
#
# Measured on the PRE-CUT cluster sum -- the same quantity the efficiency is binned
# in -- so the denominator's membership and its x position are the one number.
MIN_TRUE_ENERGY_MEV = 0.0

# Where the "average efficiency below / above" split in efficiency.txt is drawn,
# in MeV of true (pre-cut) deposited energy. Reported as two integrated ratios
# alongside the all-energy one.
EFFICIENCY_SPLIT_MEV = 500.0

_PURITY_UNMATCHED_TRUE_ID = 8888   # EvaluatePurity's "no true cluster" sentinel

CHANNELS = ('numu_CC', 'nue_CC', 'NC')

# Efficiency is produced at each of these thresholds on completeness AND purity.
# The stack's categories are fixed at HIGH_SIGNAL_THRESHOLD, but efficiency is
# rebuilt from each pair's own metrics (see build_selection_efficiency), so
# loosening the definition costs nothing and needs no re-run.
#
# The point of the set is to separate two questions that a single threshold
# conflates: how much signal the selection finds at all (good+bad) against how
# much it reconstructs to a given standard. Reading the four curves together shows
# where between 50% and 80% the reconstruction actually loses events.
# Each entry is either a scalar (same bar for completeness and purity) or a
# (completeness, purity) pair. Every stack and efficiency figure is produced at
# each of them from ONE categorisation -- see category_at_threshold -- so adding a
# definition costs figure-drawing time and no re-analysis.
#
# A purity bar of 0.0 means "irrespective of purity": completeness alone decides,
# subject only to the MIN_MATCH_PURITY floor that has to be cleared for a pair to
# exist at all. (0.6, 0.0) is "anything above 60% completeness is signal".
EFFICIENCY_THRESHOLDS = (0.8, 0.7, 0.5, (0.6, 0.8), (0.6, 0.6), (0.6, 0.0))


# ============================================================================
# OUTPUT LAYOUT
# ============================================================================
# One place that decides where a plot goes, so the tree stays navigable as the
# number of figures grows and no caller invents its own convention:
#
#   selection_efficiency/<width>MeV/<channel>/   efficiency curves, per interaction
#   selection_reco/<width>MeV/             reco-space composition stacks
#   signal_background_true/<width>MeV/     the true-cluster stacks
#   selection_completeness_vs_purity/      the pair scatter (no binning)
#   signal_neutrino_multiplicity/          the multiplicity bar chart (no binning)
#   Saved_Clusters/event_<n>/              per-cluster XZ/YZ/XY views

def plot_directory(root, kind, bin_width=None, channel=None, *variants):
    """
    The directory a plot of this kind belongs in, created if missing.

    Trailing *variants are free-form levels appended in order -- the efficiency
    plots use two, a binning (see EFFICIENCY_BINNINGS) then an uncertainty style
    (see UNCERTAINTY_DIRS), giving
    selection_efficiency/<width>MeV/<channel>/<binning>/<style>/.
    """
    path = Path(root) / kind
    if bin_width is not None:
        path = path / f"{bin_width:.0f}MeV"
    if channel is not None:
        path = path / channel
    for variant in variants:
        if variant is not None:
            path = path / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# MATCHING AND CATEGORISATION
# ============================================================================

def _is_neutrino_true_id(true_cluster_id):
    return (true_cluster_id is not None
            and true_cluster_id != _PURITY_UNMATCHED_TRUE_ID
            and true_cluster_id >= NEUTRINO_CLUSTER_ID_BASE)


def match_reco_to_true_neutrino(purity_results, completeness_results,
                                min_match_purity=MIN_MATCH_PURITY):
    """
    One pair per reco cluster: the true NEUTRINO cluster it is most pure against,
    provided that purity reaches min_match_purity.

    Returns {(event, reco_cluster_id): {'true_cluster_id', 'purity', 'completeness'}}
    for the reco clusters that matched; a reco cluster absent from the mapping
    matched no neutrino and is a cosmic candidate.

    Completeness comes from EvaluateCompleteness' row for the SAME (true, reco)
    pair -- not recomputed here, so the number is the one the rest of the pipeline
    would give. A pair with purity but no completeness row (possible: the two use
    different radii and different point-count thresholds) gets completeness 0.0
    rather than being dropped, since the reco cluster is genuinely matched and
    must still receive a label.
    """
    completeness_by_pair = {
        (r['event'], r['true_cluster_id'], r['reco_cluster_id']): r['completeness_energy_weighted']
        for r in (completeness_results or [])
    }

    best = {}
    for row in purity_results or []:
        true_id = row.get('true_cluster_id')
        if not _is_neutrino_true_id(true_id):
            continue
        purity = row.get('purity')
        if purity is None or purity < min_match_purity:
            continue
        key = (row['event'], row['reco_cluster_id'])
        if key not in best or purity > best[key]['purity']:
            best[key] = {
                'true_cluster_id': true_id,
                'purity':          float(purity),
                'completeness':    float(completeness_by_pair.get(
                                       (row['event'], true_id, row['reco_cluster_id']), 0.0)),
            }
    return best


def categorize_reco_clusters(reco_records, purity_results, completeness_results,
                             true_records, min_match_purity=MIN_MATCH_PURITY,
                             threshold=HIGH_SIGNAL_THRESHOLD):
    """
    One record per SELECTED reco cluster, carrying its category and everything
    the plots need.

    Parameters:
    - reco_records: build_reco_cluster_variable_records output for the reco
        selection being judged (the beam-window one)
    - purity_results / completeness_results: EvaluatePurity / EvaluateCompleteness
        for the SAME reco selection -- pairing against a different reco population
        would label clusters by overlaps the plots never show
    - true_records: build_true_cluster_variable_records output with
        attach_interaction_channel applied, i.e. carrying 'interaction_channel'
        and 'vertex_in_volume'

    Returns a list of dicts: identity, reco_energy_mev, category, channel,
    pair_true_cluster_id, pair_purity, pair_completeness.

    Categories: cosmic, out_of_volume, high_signal_numu_CC, high_signal_nue_CC,
    high_signal_NC, contaminated. Every input record gets exactly one.

    'cosmic' covers two cases: a reco cluster that matched no true neutrino at all,
    and one whose match is below COSMIC_MAX_COMPLETENESS_PURITY on both axes. Only
    the second has a completeness and a purity, and 'cosmic_by_low_overlap'
    distinguishes them.

    A pair whose true cluster has vertex_in_volume None (no mc.json vertex to
    test) is counted as out_of_volume, NOT as in-volume: an unknown vertex has not
    been shown to be inside, and the alternative -- a seventh 'unknown' band --
    would split a category on a bookkeeping gap rather than on physics. The count
    is returned by the caller-visible warning below so it cannot grow unnoticed.
    """
    completeness_min, purity_min = threshold_pair(threshold)
    pairs = match_reco_to_true_neutrino(purity_results, completeness_results, min_match_purity)
    true_by_id = {(r['event'], r['true_cluster_id']): r for r in (true_records or [])}

    n_unknown_vertex = 0
    n_unknown_channel = 0
    records = []
    for reco in reco_records or []:
        key = (reco['event'], reco['reco_cluster_id'])
        pair = pairs.get(key)

        record = {
            'file_name':            reco.get('file_name'),
            'event':                reco['event'],
            'event_num':            reco.get('event_num'),
            'apa':                  reco.get('apa'),
            'reco_cluster_id':      reco['reco_cluster_id'],
            'total_charge':         reco.get('total_charge'),
            'reco_energy_mev':      reco_cluster_energy_mev(reco.get('total_charge') or 0.0),
            'pair_true_cluster_id': None,
            'pair_purity':          None,
            'pair_completeness':    None,
            'channel':              None,
            'category':             'cosmic',
            # True only for the ones demoted by the low-overlap rule; an unmatched
            # cluster has no completeness or purity at all, and the two are drawn
            # differently on the scatter.
            'cosmic_by_low_overlap': False,
        }

        if pair is not None:
            true_record = true_by_id.get((reco['event'], pair['true_cluster_id'])) or {}
            in_volume = true_record.get('vertex_in_volume')
            channel   = true_record.get('interaction_channel')
            record.update({
                'pair_true_cluster_id': pair['true_cluster_id'],
                'pair_purity':          pair['purity'],
                'pair_completeness':    pair['completeness'],
                'channel':              channel,
            })
            # Tested BEFORE the vertex volume: a cluster this weakly related to its
            # neutrino is a cosmic wherever that neutrino's vertex happens to be, so
            # the rule applies to out-of-volume matches too. It is threshold
            # independent, so a record marked cosmic here stays cosmic at every
            # high-signal threshold (see category_at_threshold).
            if (pair['completeness'] < COSMIC_MAX_COMPLETENESS_PURITY
                    and pair['purity'] < COSMIC_MAX_COMPLETENESS_PURITY):
                record['category'] = 'cosmic'
                record['cosmic_by_low_overlap'] = True
            elif in_volume is not True:
                if in_volume is None:
                    n_unknown_vertex += 1
                record['category'] = 'out_of_volume'
            elif (pair['completeness'] > completeness_min and pair['purity'] > purity_min
                  and channel in CHANNELS):
                record['category'] = f'high_signal_{channel}'
            else:
                if channel not in CHANNELS:
                    n_unknown_channel += 1
                record['category'] = 'contaminated'

        records.append(record)

    if n_unknown_vertex:
        print(f"    NOTE: {n_unknown_vertex} reco cluster(s) paired to a true neutrino with no "
              f"vertex in mc.json -- counted as out_of_volume")
    if n_unknown_channel:
        print(f"    NOTE: {n_unknown_channel} in-volume pair(s) have no interaction channel -- "
              f"counted as contaminated even if well reconstructed")
    return records


# ============================================================================
# THE STACK'S BANDS
# ============================================================================
# Declaration order only; order_selection_components_by_size sorts smallest-first
# at draw time, as in draw_signal_background. 'legend_rank' pins the three signal
# channels to the top of the legend in a fixed order, so they do not move between
# runs as their counts change.
# 'signal' marks the three bands the analysis is trying to select. They are drawn
# SOLID and sit on TOP of the stack. Everything else is drawn UNFILLED -- hatch
# and outline in its own colour, no solid face -- and sits below.
#
# Two conventions doing one job: the signal is what the eye should find first, and
# on top it has a free upper edge to read against instead of being sandwiched.
# Leaving the rest unfilled keeps three large background bands from dominating the
# picture by sheer area, and the distinction survives greyscale printing and
# colour-blind readers, which a colour-only scheme does not.
SELECTION_COMPONENTS = [
    {'key': 'high_signal_numu_CC', 'label': r'Candidate $\nu_\mu$ CC in-volume, Signal',
     'color': 'tab:blue',   'legend_rank': 0, 'signal': True,  'hatch': None, 'pinned': True},
    {'key': 'high_signal_NC',      'label': 'Candidate NC in-volume, Signal',
     'color': 'tab:green',  'legend_rank': 1, 'signal': True,  'hatch': None, 'pinned': True},
    {'key': 'high_signal_nue_CC',  'label': r'Candidate $\nu_e$ CC in-volume, Signal',
     'color': 'tab:orange', 'legend_rank': 2, 'signal': True,  'hatch': None, 'pinned': True},
    {'key': 'contaminated',        'label': 'Contamination+Incomplete',
     'color': 'tab:cyan',   'legend_rank': 3, 'signal': False, 'hatch': '//', 'pinned': True},
    {'key': 'out_of_volume',       'label': 'Out-of-volume neutrinos',
     'color': 'tab:purple', 'legend_rank': 4, 'signal': False, 'hatch': 'xx'},
    # Cosmics are the one background drawn FILLED, in light grey: they are the
    # largest band and the least interesting, and a hatch over that much area
    # competes with the signal for attention. A flat pale fill reads as
    # background without demanding any.
    {'key': 'cosmic',              'label': 'Cosmic',
     'color': 'lightgray', 'legend_rank': 5, 'signal': False, 'hatch': None,
     'solid_fill': True},
]

_SELECTED_TOTAL_STYLE = dict(color='black', linestyle='-', linewidth=2.0)


def threshold_pair(threshold):
    """
    (completeness threshold, purity threshold) from either form.

    A scalar means the same bar on both metrics -- the historic behaviour, and
    still what HIGH_SIGNAL_THRESHOLD is. A 2-tuple sets them SEPARATELY, which is
    what asymmetric definitions like "completeness > 60%, purity > 80%" need: the
    two metrics fail for different reasons (incomplete vs dirty) and there is no
    reason the bar has to be the same height for both.
    """
    if isinstance(threshold, (tuple, list)):
        return float(threshold[0]), float(threshold[1])
    return float(threshold), float(threshold)


def threshold_tag(threshold):
    """Filename fragment: '_thr80' when symmetric, '_c60p80' when not,
    '_c60pany' when purity is unconstrained."""
    completeness, purity = threshold_pair(threshold)
    if completeness == purity:
        return f"_thr{completeness:.0%}".replace('%', '')
    if purity == 0.0:
        return f"_c{completeness:.0%}pany".replace('%', '')
    return f"_c{completeness:.0%}p{purity:.0%}".replace('%', '')


def threshold_label(threshold):
    """Human-readable form for titles and legends."""
    completeness, purity = threshold_pair(threshold)
    if completeness == purity:
        return f"completeness & purity > {completeness:.0%}"
    if purity == 0.0:
        # Purity unconstrained -- the absence of a purity clause says so, and the
        # signal-definition subdirectory the figure lands in spells it out.
        return f"completeness > {completeness:.0%}"
    return f"completeness > {completeness:.0%}, purity > {purity:.0%}"


def threshold_dirname(threshold):
    """
    Subdirectory name for one signal definition, so the efficiency figures are
    grouped by the threshold they were drawn at rather than all in one flat
    directory. Filenames still carry threshold_tag; this is the folder above them.
    """
    completeness, purity = threshold_pair(threshold)
    if completeness == purity:
        return f"completeness_and_purity_gt_{completeness * 100:.0f}pc"
    if purity == 0.0:
        return f"completeness_gt_{completeness * 100:.0f}pc"
    return (f"completeness_gt_{completeness * 100:.0f}pc"
            f"_purity_gt_{purity * 100:.0f}pc")


def category_at_threshold(record, threshold=HIGH_SIGNAL_THRESHOLD):
    """
    The category this record would have at a different high-signal threshold.

    Only the in-volume split moves: cosmic and out_of_volume are decided by the
    match and the vertex, neither of which a quality threshold touches. Derived
    from the pair's own metrics rather than from the stored category, so one
    categorisation serves every threshold and the stack can be redrawn at 70% or
    50% without re-running the job.

    At threshold == HIGH_SIGNAL_THRESHOLD this returns the stored category.
    """
    if record['category'] in ('cosmic', 'out_of_volume'):
        return record['category']
    channel = record.get('channel')
    completeness_min, purity_min = threshold_pair(threshold)
    if (channel in CHANNELS
            and (record.get('pair_completeness') or 0) > completeness_min
            and (record.get('pair_purity') or 0) > purity_min):
        return f'high_signal_{channel}'
    return 'contaminated'


def order_components_for_stack(components, by_key):
    """
    Drawing order, bottom first: the free background bands (smallest first), then
    the PINNED bands on top, REVERSED so that the declared order reads top-down.

    The declaration order of SELECTION_COMPONENTS is therefore the single place
    that fixes both orderings: first declared is first in the legend AND highest
    in the stack. Reversing here rather than declaring them backwards keeps the
    list readable in the order the plots present it.

    Pinned by rule rather than by size, so the bands the analysis cares about keep
    their position between runs and between thresholds -- at a loose threshold the
    signal grows and the contaminated band shrinks, and a size-ordered stack would
    reshuffle itself and stop being comparable to the tighter one. Contamination
    is pinned too, directly under the three signal channels: it is in-volume
    signal that missed the quality cut, so it belongs with them rather than
    floating among the backgrounds by size.
    """
    pinned = [c for c in components if c.get('pinned')]
    free = sorted((c for c in components if not c.get('pinned')),
                  key=lambda c: len(by_key.get(c['key'], [])))
    return free + pinned[::-1]


def split_by_category(categorized_records, components=None, threshold=HIGH_SIGNAL_THRESHOLD):
    """{component key: [records]} for the stack, plus a check that it partitions."""
    components = components if components is not None else SELECTION_COMPONENTS
    by_key = {c['key']: [] for c in components}
    unclaimed = []
    for record in categorized_records or []:
        category = category_at_threshold(record, threshold)
        if category in by_key:
            by_key[category].append(record)
        else:
            unclaimed.append(record)
    return by_key, unclaimed


def order_selection_components_by_size(components, by_key):
    """Smallest first -- the drawing order, as in draw_signal_background."""
    return sorted(components, key=lambda c: len(by_key.get(c['key'], [])))


def order_selection_components_for_legend(components, by_key):
    """Ranked components first, then biggest-first."""
    def sort_key(component):
        rank = component.get('legend_rank')
        n = len(by_key.get(component['key'], []))
        return (0, rank, 0) if rank is not None else (1, 0, -n)
    return sorted(components, key=sort_key)


# ============================================================================
# PLOT 1 -- RECO-SPACE STACK
# ============================================================================

def draw_reco_selection_stack(categorized_records, output_dir, level_name, filename_prefix, apa,
                              components=None, bin_width=ENERGY_BIN_WIDTH_MEV,
                              reco_cuts_label='AfterBeamWindowCut', y_scale='linear',
                              threshold=HIGH_SIGNAL_THRESHOLD, fill_style='mixed',
                              filename='selection_reco_energy_stack'):
    """
    Stacked histogram of SELECTED RECO clusters by category, x = reco cluster
    energy from its charge, y = number of reco clusters.

    The outline over the stack is every selected reco cluster, drawn from the same
    records. It is a closure check made visible: the stack's bands partition the
    selection, so the outline must sit exactly on the top of the stack in every
    bin. Any daylight between them means a cluster was lost or double counted.
    The check is also made in code -- see the assertion below -- because a
    one-cluster discrepancy is invisible by eye and fatal to the interpretation.

    Returns (by_key, all_energies).
    """
    components = components if components is not None else SELECTION_COMPONENTS
    # One sub-directory per signal threshold, matching the selection_efficiency
    # tree (see threshold_dirname): the caller draws this at every entry in
    # EFFICIENCY_THRESHOLDS, and a flat directory made the six versions hard to
    # tell apart. Filenames still carry threshold_tag; this is the folder above.
    output_dir = Path(output_dir) / threshold_dirname(threshold)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_key, unclaimed = split_by_category(categorized_records, components, threshold)
    if unclaimed:
        raise ValueError(
            f"{len(unclaimed)} reco cluster(s) carry a category no component claims "
            f"({sorted({r['category'] for r in unclaimed})}). The stack would not be the "
            f"selection; fix the component list or the categoriser rather than drawing this.")

    n_stacked = sum(len(v) for v in by_key.values())
    n_total   = len(categorized_records or [])
    assert n_stacked == n_total, (
        f"stack holds {n_stacked} reco clusters but {n_total} were selected -- "
        f"the categories do not partition the selection")

    drawing_order = order_components_for_stack(components, by_key)
    values_by_key = {key: [r['reco_energy_mev'] for r in records] for key, records in by_key.items()}
    all_energies  = [r['reco_energy_mev'] for r in (categorized_records or [])]

    edges = energy_bin_edges(bin_width)
    n_overflow = count_overflow(all_energies)
    if n_overflow:
        print(f"    NOTE: {n_overflow} selected reco cluster(s) above {ENERGY_AXIS_MAX_MEV:.0f} MeV "
              f"are off the fixed axis and not drawn")

    fig, ax = plt.subplots(figsize=(10, 7))
    _, _, patch_groups = ax.hist(
        [values_by_key[c['key']] for c in drawing_order],
        bins=edges, stacked=True, histtype='stepfilled',
        color=[c['color'] for c in drawing_order],
        edgecolor='black', linewidth=1.0,
    )
    # hist takes one hatch and one facecolor for the whole call, so the per-band
    # styling is applied to the returned artists instead. Non-signal bands lose
    # their solid face entirely and keep only the hatch, drawn in the band's own
    # colour so the legend key still identifies it.
    for component, group in zip(drawing_order, patch_groups):
        # 'solid' draws every band as a plain filled colour, for the version where
        # the composition is read by area alone; 'mixed' keeps the hatched
        # backgrounds. Same data either way -- only the styling differs.
        if fill_style == 'solid' or component.get('signal') or component.get('solid_fill'):
            continue
        for patch in (group if hasattr(group, '__iter__') else [group]):
            patch.set_facecolor('none')
            patch.set_edgecolor(component['color'])
            patch.set_linewidth(1.4)
            if component.get('hatch'):
                patch.set_hatch(component['hatch'])

    # The closure outline, drawn last so nothing paints over it.
    total_counts, _ = np.histogram(all_energies, bins=edges)
    total_handle = ax.step(edges, np.append(total_counts, total_counts[-1]), where='post',
                           **_SELECTED_TOTAL_STYLE)[0]

    # Per-bin closure, not just the totals: the stack must equal the outline
    # everywhere, which is the statement the plot is making.
    stacked_counts = np.sum([np.histogram(values_by_key[c['key']], bins=edges)[0]
                             for c in drawing_order], axis=0)
    assert np.array_equal(stacked_counts, total_counts), (
        "stack and selected-total disagree bin by bin")

    ax.set_xlabel('Reco Cluster Energy (MeV)', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Number of Reco Clusters', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    set_fitted_title(ax, f'Selection Composition -- reco: {reco_cuts_label} '
                 f'-- Signal: {threshold_label(threshold)}',
                 _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(edges[0], PLOT_X_MAX_MEV)
    ax.xaxis.set_major_locator(MultipleLocator(ENERGY_AXIS_TICK_MEV))
    n_beyond = sum(1 for e in all_energies if e > PLOT_X_MAX_MEV)
    if n_beyond:
        print(f"    NOTE: {n_beyond} selected reco cluster(s) above {PLOT_X_MAX_MEV:.0f} MeV are "
              f"binned and tabulated but beyond the drawn axis")

    ax.set_yscale(y_scale)
    # The y range follows the DRAWN range, not the full one: a tall bin out at
    # 4 GeV would otherwise set a scale for a region the reader cannot see.
    visible = total_counts[edges[:-1] < PLOT_X_MAX_MEV]
    tallest = max(int(visible.max()) if len(visible) else 0, 1)
    if y_scale == 'log':
        ax.set_ylim(0.5, tallest * 25.0)
    else:
        ax.set_ylim(0, tallest * (1 + _Y_HEADROOM_LINEAR))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    handle_by_key = {c['key']: (group[0] if len(group) else None)
                     for c, group in zip(drawing_order, patch_groups)}
    keyed = [(handle_by_key[c['key']], f"{c['label']} ({len(by_key[c['key']])})")
             for c in order_selection_components_for_legend(components, by_key)
             if handle_by_key.get(c['key']) is not None]
    keyed.append((total_handle, f"All selected reco ({n_total})"))
    ax.legend([h for h, _ in keyed], [l for _, l in keyed],
              fontsize=_LEGEND_FONTSIZE, loc='upper right', framealpha=0.9)

    scale_suffix = "_logy" if y_scale == 'log' else "_liny"
    tag = threshold_tag(threshold)
    fill_tag = "" if fill_style == 'mixed' else f"_{fill_style}"
    path = output_dir / (f"{filename}_{reco_cuts_label}_{bin_width:.0f}MeV"
                         f"{tag}{fill_tag}{scale_suffix}_{filename_prefix}_{apa}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return by_key, all_energies


# ============================================================================
# PLOT 2 -- COMPLETENESS vs PURITY
# ============================================================================

# The cosmic candidates have no pair, so no purity and no completeness. They are
# drawn in an off-scale box below and left of the physical square, the same
# placeholder idiom completeness_purity_draw.py uses, rather than at a literal
# (0, 0) that would sit on the axes among real measurements.
_COSMIC_BOX_LO, _COSMIC_BOX_HI = -0.12, -0.02


def draw_completeness_vs_purity(categorized_records, output_dir, level_name, filename_prefix, apa,
                                threshold=HIGH_SIGNAL_THRESHOLD,
                                reco_cuts_label='AfterBeamWindowCut',
                                filename='selection_completeness_vs_purity',
                                multi_neutrino_events=None, multi_neutrino_only=False,
                                channel_only=None):
    """
    Scatter of pair completeness (y) against pair purity (x), one point per
    IN-VOLUME reco-true pair, coloured by interaction channel, with the cosmic
    candidates in their own off-scale box and a dashed box marking the high-signal
    corner.

    Out-of-volume pairs are deliberately absent: they are a rejection category and
    their completeness/purity say nothing about how well the signal was
    reconstructed.

    multi_neutrino_events, when given, is the set of event keys holding MORE THAN
    ONE signal neutrino ("multi neutrino" in the legend). Pairs from those events
    are drawn RED instead of their
    channel colour, so it is visible whether the multi-neutrino population sits
    anywhere in particular on the plane. They keep their channel MARKER, so the
    channel is still readable, and the channel counts in the legend exclude them
    to keep the totals consistent. The filename gains '_multinu'.

    channel_only, when given ('numu_CC', 'nue_CC' or 'NC'), keeps ONLY that
    channel's pairs and drops the cosmics. The combined plot is dominated by numu
    CC -- 409 of the 553 in-volume pairs on the full sample -- so the handful of
    nue CC and the NC population are invisible under it. Filename gains
    '_<channel>'. The axes and limits are unchanged, so a per-channel plot can be
    laid over the combined one.

    multi_neutrino_only additionally DROPS everything else -- single-neutrino
    pairs and cosmics -- leaving just the red population on the same axes, so the
    handful of points can be read without the bulk on top of them. Filename
    '_multinu_only'. The axes and limits are identical to the other two versions
    so the three can be compared directly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_volume = [r for r in categorized_records or []
                 if r['category'] == 'contaminated' or r['category'].startswith('high_signal_')]
    # Two kinds of cosmic, drawn differently because only one of them has
    # coordinates: the low-overlap ones sit at their real (purity, completeness),
    # in the bottom-left corner where the rule put them, while the unmatched ones
    # have no such values at all and go in the off-scale box.
    demoted   = [r for r in categorized_records or []
                 if r['category'] == 'cosmic' and r.get('cosmic_by_low_overlap')]
    cosmics   = [r for r in categorized_records or []
                 if r['category'] == 'cosmic' and not r.get('cosmic_by_low_overlap')]

    multi = ({r['event'] for r in in_volume if r['event'] in multi_neutrino_events}
             if multi_neutrino_events else set())
    if channel_only:
        # Cosmics have no channel, so a per-channel plot cannot show them --
        # dropping them is what keeps the plot about that channel.
        in_volume = [r for r in in_volume if r['channel'] == channel_only]
        demoted, cosmics = [], []
        multi &= {r['event'] for r in in_volume}
    if multi_neutrino_only:
        # Cosmics are not attributable to a neutrino, so a multi-neutrino-only
        # plot cannot show them; dropping them keeps the plot honest rather than
        # implying they belong to the red population.
        in_volume = [r for r in in_volume if r['event'] in multi]
        demoted, cosmics = [], []

    fig, ax = plt.subplots(figsize=(10, 8))
    channel_style = {'numu_CC': ('tab:blue',   'o', r'$\nu_\mu$ CC'),
                     'nue_CC':  ('tab:orange', 's', r'$\nu_e$ CC'),
                     'NC':      ('tab:green',  '^', 'NC')}
    for channel, (color, marker, label) in channel_style.items():
        selected = [r for r in in_volume
                    if r['channel'] == channel and r['event'] not in multi]
        if not selected:
            continue
        ax.scatter([r['pair_purity'] for r in selected],
                   [r['pair_completeness'] for r in selected],
                   c=color, marker=marker, s=60, alpha=0.7, edgecolors='black', linewidth=0.4,
                   label=f'{label} ({len(selected)})')

    if multi:
        # Red, keeping each pair's channel marker so both facts stay readable.
        # One legend entry per channel present, rather than one lumped entry:
        # otherwise the channel counts above and this count double-count pairs.
        for channel, (_color, marker, label) in channel_style.items():
            selected = [r for r in in_volume
                        if r['channel'] == channel and r['event'] in multi]
            if not selected:
                continue
            ax.scatter([r['pair_purity'] for r in selected],
                       [r['pair_completeness'] for r in selected],
                       c='red', marker=marker, s=60, alpha=0.85,
                       edgecolors='black', linewidth=0.4, zorder=5,
                       label=f'{label}, from multi neutrino event ({len(selected)})')

    if demoted:
        ax.scatter([r['pair_purity'] for r in demoted],
                   [r['pair_completeness'] for r in demoted],
                   c='tab:gray', marker='v', s=45, alpha=0.7, edgecolors='black', linewidth=0.3,
                   label=f'Cosmic, both < {COSMIC_MAX_COMPLETENESS_PURITY:.0%} ({len(demoted)})')
        # The corner the rule claims, so the boundary is visible rather than
        # implied by where the triangles happen to stop.
        ax.add_patch(plt.Rectangle((0, 0), COSMIC_MAX_COMPLETENESS_PURITY,
                                   COSMIC_MAX_COMPLETENESS_PURITY,
                                   fill=False, edgecolor='tab:gray', linestyle='--', linewidth=1.6))

    if cosmics:
        rng = np.random.default_rng(42)
        x = rng.uniform(_COSMIC_BOX_LO, _COSMIC_BOX_HI, len(cosmics))
        y = rng.uniform(_COSMIC_BOX_LO, _COSMIC_BOX_HI, len(cosmics))
        ax.scatter(x, y, c='tab:gray', marker='x', s=45, alpha=0.7, linewidth=1.4,
                   label=f'Cosmic, no true match ({len(cosmics)})')
        ax.add_patch(plt.Rectangle((_COSMIC_BOX_LO - 0.01, _COSMIC_BOX_LO - 0.01),
                                   (_COSMIC_BOX_HI - _COSMIC_BOX_LO) + 0.02,
                                   (_COSMIC_BOX_HI - _COSMIC_BOX_LO) + 0.02,
                                   fill=False, edgecolor='gray', linestyle='--', linewidth=1.4))

    # No box is drawn at the high-signal corner: the thresholds are a downstream
    # choice (efficiency is produced at several -- see EFFICIENCY_THRESHOLDS), and
    # marking one of them here would make this plot look like it endorses that one.
    # The distribution is the point; where a cut goes is a separate question.

    ax.set_xlabel('Purity', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Completeness', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    channel_label = {'numu_CC': r'$\nu_\mu$ CC', 'nue_CC': r'$\nu_e$ CC',
                     'NC': 'NC'}.get(channel_only, channel_only)
    population = ('in-volume, MULTI NEUTRINO events only' if multi_neutrino_only
                  else f'in-volume {channel_label}' if channel_only
                  else 'in-volume')
    set_fitted_title(ax, f'Reco-True Pairs, {population} -- reco: {reco_cuts_label}',
                     _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(_COSMIC_BOX_LO - 0.04, 1.06)
    ax.set_ylim(_COSMIC_BOX_LO - 0.04, 1.06)
    ax.legend(fontsize=_LEGEND_FONTSIZE, loc='lower right', framealpha=0.9)

    suffix = ('_multinu_only' if multi_neutrino_only
              else '_multinu' if multi_neutrino_events else '')
    if channel_only:
        suffix += f'_{channel_only}'
    path = output_dir / f"{filename}{suffix}_{reco_cuts_label}_{filename_prefix}_{apa}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return in_volume, cosmics


# ROOT's kTemperatureMap is a dark-blue -> pale -> dark-red diverging ramp;
# matplotlib's 'coolwarm' is the same idea and the closest thing in its built-in
# set. Named here rather than inline so the scatter and the colz cannot drift onto
# different palettes.
_COLZ_CMAP = 'coolwarm'


def draw_completeness_vs_purity_colz(categorized_records, output_dir, level_name,
                                     filename_prefix, apa, bins=20,
                                     reco_cuts_label='AfterBeamWindowCut',
                                     filename='selection_completeness_vs_purity_colz',
                                     include_cosmics=True, log_scale=False):
    """
    The same in-volume pairs as the scatter, as a 2D histogram.

    The scatter shows every pair and where the outliers are; this shows where the
    density is, which the scatter hides once points overlap -- at these statistics
    the high-completeness/high-purity corner is a solid block of markers in the
    scatter and its internal structure is invisible. The two are the same data and
    are drawn from the same selection, so they must agree on the totals.

    Two versions, chosen by include_cosmics:

    - True: cosmic candidates get an off-scale band below and left of the physical
      square. They are a large part of what a selection returns, and a picture of
      the selection that omits them understates it. They will usually be the
      hottest cell on the map, which is itself the point.
    - False: axes start at (0, 0) and only the in-volume pairs are filled -- the
      same distribution read without the off-scale band pulling the eye out of the
      square, and with the same limits as the scatter, so the two can be compared
      cell for marker.

    The caller draws both; they differ in filename by the '_with_cosmics' suffix.

    log_scale picks the COLOUR scale, not the axes. It now defaults to False
    (linear -- the honest reading of a density, twice the colour is twice the
    clusters) and the linear version carries NO filename suffix; the log-z
    version was dropped from the standard job output. Pass log_scale=True to get
    it back, which adds a '_logz' suffix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_volume = [r for r in categorized_records or []
                 if r['category'] == 'contaminated' or r['category'].startswith('high_signal_')]
    cosmics = [r for r in categorized_records or [] if r['category'] == 'cosmic']
    if not in_volume:
        return None
    purity = [r['pair_purity'] for r in in_volume]
    completeness = [r['pair_completeness'] for r in in_volume]

    # The cosmics have no completeness and no purity, so they go in one wide
    # off-scale bin below and left of the physical square -- the same placeholder
    # idiom the scatter uses, and the reason this plot's axes start below zero.
    # Kept as ONE bin rather than spread over several: nothing inside it is a
    # coordinate, and giving it width would suggest a distribution it does not have.
    box_mid = (_COSMIC_BOX_LO + _COSMIC_BOX_HI) / 2
    if include_cosmics:
        for record in cosmics:
            if record.get('cosmic_by_low_overlap'):
                purity.append(record['pair_purity'])        # these DO have coordinates
                completeness.append(record['pair_completeness'])
            else:
                purity.append(box_mid)
                completeness.append(box_mid)

    square = np.linspace(0.0, 1.0, bins + 1)
    edges = (np.concatenate(([_COSMIC_BOX_LO], square)) if include_cosmics else square)
    fig, ax = plt.subplots(figsize=(10, 8))
    # Empty bins left blank (cmin=1) either way. The colour scale is LINEAR by
    # default now (log_scale=False). The log-z version -- which kept the pair
    # distribution readable when the single cosmic cell holds every unmatched
    # cluster and would otherwise take the whole colour range -- was dropped from
    # the standard job output; pass log_scale=True to bring it back.
    counts, _, _, mesh = ax.hist2d(purity, completeness, bins=[edges, edges],
                                   cmap=_COLZ_CMAP,
                                   norm=(LogNorm() if log_scale else None), cmin=1)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label('Clusters (log scale)' if log_scale else 'Clusters',
                       fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    colorbar.ax.tick_params(labelsize=_TICK_LABEL_FONTSIZE)

    ax.set_xlabel('Purity', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Completeness', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    counted = (f'in-volume ({len(in_volume)}) + cosmics ({len(cosmics)})'
               if include_cosmics else f'in-volume ({len(in_volume)})')
    set_fitted_title(ax, f'Reco-True Pairs, {counted} -- reco: {reco_cuts_label}',
                     _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.set_xlim(edges[0], 1)
    ax.set_ylim(edges[0], 1)
    if include_cosmics:
        # Mark where the physical square starts, so the off-scale band is not read
        # as part of the distribution.
        ax.axvline(0.0, color='black', linewidth=1.2, linestyle='--')
        ax.axhline(0.0, color='black', linewidth=1.2, linestyle='--')
        ax.text(_COSMIC_BOX_LO / 2, _COSMIC_BOX_LO / 2, 'cosmic\n(no match)',
                ha='center', va='center',
                fontsize=_LEGEND_FONTSIZE - 2, fontweight='bold')

    suffix = ('_with_cosmics' if include_cosmics else '') + ('_logz' if log_scale else '')
    path = output_dir / f"{filename}{suffix}_{reco_cuts_label}_{filename_prefix}_{apa}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return path


def place_equation_above_legend(fig, ax, legend, equation, top=0.98):
    """
    Put `equation` in a box directly above `legend`, inside the axes, moving the
    legend DOWN if the box would otherwise run off the top.

    A legend at 'upper right' already sits against the top of the axes, so there
    is no room above it: writing there put the equation over the title. Rather
    than shrink the text or drop it below the legend -- where it reads as a
    footnote to the entries rather than a statement about the axis -- this takes
    the room it needs from the legend's position, which nothing else depends on.

    Two passes, because both boxes are sized in pixels by the renderer and their
    height in axes coordinates is not known until they are drawn.
    """
    fig.canvas.draw()
    box = legend.get_window_extent().transformed(ax.transAxes.inverted())
    text = ax.text(box.x1, box.y1 + 0.03, equation, transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=_LEGEND_FONTSIZE,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray'))

    fig.canvas.draw()
    text_box = text.get_window_extent().transformed(ax.transAxes.inverted())
    overflow = text_box.y1 - top
    if overflow > 0:
        legend.set_bbox_to_anchor((box.x1, box.y1 - overflow), transform=ax.transAxes)
        fig.canvas.draw()
        moved = legend.get_window_extent().transformed(ax.transAxes.inverted())
        text.set_position((moved.x1, moved.y1 + 0.03))
    return text


def draw_cosmic_distributions(categorized_records, output_dir, level_name, filename_prefix, apa,
                              reco_cuts_label='AfterBeamWindowCut',
                              filename='selection_cosmic'):
    """
    Charge and reco energy of the SELECTED COSMIC clusters, one figure each.

    Their own plot because the stack cannot show them: on a shared axis the
    cosmics pile into the first bin or two and their shape is invisible. Measured
    on the full sample, 86% of them sit below 100 MeV with a median near 50 MeV,
    an order of magnitude softer than any other category -- so a dedicated axis is
    the only way to see the distribution at all.

    The energy figure carries the conversion it was made with, printed on the
    axes: a reader should not have to look up how charge became MeV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cosmics = [r for r in categorized_records or [] if r['category'] == 'cosmic']
    if not cosmics:
        return []

    charges  = np.array([r.get('total_charge') or 0.0 for r in cosmics], dtype=float)
    energies = np.array([r.get('reco_energy_mev') or 0.0 for r in cosmics], dtype=float)

    paths = []
    for values, xlabel, tag, equation in (
            (charges, 'Reco Cluster Charge (ADC units)', 'charge', None),
            (energies, 'Reco Cluster Energy (MeV)', 'energy',
             rf"$E = W\,Q/R$   with  $W$ = {RECO_WORK_FUNCTION_EV} eV,  $R$ = {RECO_RECOMBINATION_FACTOR}"
             "\n" rf"i.e. $E$ [MeV] = {RECO_MEV_PER_CHARGE:.3e} $\times\ Q$")):
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        # 99th percentile, not the max: a single huge cluster would otherwise put
        # every other cosmic in the first bin.
        top = np.percentile(finite, 99) if len(finite) > 10 else finite.max()
        bins = np.linspace(0, max(top, 1e-9), 61)
        ax.hist(np.clip(finite, 0, bins[-1]), bins=bins, color='lightgray',
                edgecolor='black', linewidth=1.0,
                label=f'Cosmic ({len(finite)})')
        ax.set_xlabel(xlabel, fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_ylabel('Number of Reco Clusters', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
        set_fitted_title(ax, f'Selected cosmic clusters -- reco: {reco_cuts_label}',
                         _TITLE_FONTSIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        legend = ax.legend(fontsize=_LEGEND_FONTSIZE, loc='upper right', framealpha=0.9)
        if equation:
            place_equation_above_legend(fig, ax, legend, equation)
        median = np.median(finite)
        ax.axvline(median, color='tab:red', linestyle='--', linewidth=1.4)
        ax.text(median, ax.get_ylim()[1] * 0.96, f' median {median:,.0f}',
                color='tab:red', fontsize=_LEGEND_FONTSIZE - 1, va='top')
        path = output_dir / f"{filename}_{tag}_{reco_cuts_label}_{filename_prefix}_{apa}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        paths.append(path)
    return paths


# Quality cuts the energy-reconstruction plots are made at. Only well matched
# pairs are useful for a calibration: on a poorly matched pair the reco cluster
# and the true cluster are not the same object, so the two energies are not two
# measurements of one quantity and the correlation means nothing.
ENERGY_RECO_QUALITY = (0.90, 0.80)


def draw_energy_reconstruction(categorized_records, true_records, output_dir, level_name,
                               filename_prefix, apa, bin_width=ENERGY_BIN_WIDTH_MEV,
                               quality=0.90, reco_cuts_label='AfterBeamWindowCut',
                               filename='energy_reconstruction'):
    """
    TRUE deposited energy (y) against RECO energy (x) for well matched pairs, 2D.

    Restricted to pairs above `quality` in BOTH completeness and purity: below
    that the reco cluster is not the same object as the true cluster and the
    correlation is between two different things.

    The true energy is the paired true cluster's summed point energy AFTER the
    cuts -- the same quantity the truth stacks are filled with, not the pre-cut
    sum the efficiency is binned in. The reco energy is charge times the
    conversion printed on the plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_energy_by_id = {(r['event'], r['true_cluster_id']): r.get('total_energy')
                         for r in (true_records or [])}
    pairs = []
    for record in categorized_records or []:
        if record['category'] != 'contaminated' and not record['category'].startswith('high_signal_'):
            continue
        if (record.get('pair_completeness') or 0) <= quality or (record.get('pair_purity') or 0) <= quality:
            continue
        true_energy = true_energy_by_id.get((record['event'], record['pair_true_cluster_id']))
        if not true_energy:
            continue
        pairs.append((record['reco_energy_mev'], true_energy))
    if not pairs:
        print(f"    NOTE: no pairs above {quality:.0%} completeness and purity, "
              f"no energy_reconstruction plot at {bin_width:.0f} MeV")
        return None

    reco = np.array([p[0] for p in pairs], dtype=float)
    true = np.array([p[1] for p in pairs], dtype=float)
    edges = np.arange(0, PLOT_X_MAX_MEV + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    counts, _, _, mesh = ax.hist2d(np.clip(reco, 0, edges[-1]), np.clip(true, 0, edges[-1]),
                                   bins=[edges, edges], cmap=_COLZ_CMAP, cmin=1)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label('Reco-true pairs', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    colorbar.ax.tick_params(labelsize=_TICK_LABEL_FONTSIZE)
    # y = x: where a perfectly calibrated reco energy would sit.
    ax.plot([0, edges[-1]], [0, edges[-1]], color='black', linestyle=':', linewidth=1.6,
            label='true = reco')

    ax.set_xlabel('Reco Cluster Energy (MeV)', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('True Deposited Cluster Energy (MeV)', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    set_fitted_title(ax, f'Energy reconstruction, pairs above {quality:.0%} completeness '
                         f'and purity -- reco: {reco_cuts_label}',
                     _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.set_xlim(0, edges[-1]); ax.set_ylim(0, edges[-1])
    ax.grid(True, linestyle='--', alpha=0.3)
    legend = ax.legend(fontsize=_LEGEND_FONTSIZE, loc='lower right', framealpha=0.9,
                       title=f'{len(pairs)} pairs')

    equation = (rf"$E = W\,Q/R$   with  $W$ = {RECO_WORK_FUNCTION_EV} eV,  "
                rf"$R$ = {RECO_RECOMBINATION_FACTOR}" "\n"
                rf"i.e. $E$ [MeV] = {RECO_MEV_PER_CHARGE:.3e} $\times\ Q$")
    place_equation_above_legend(fig, ax, legend, equation)

    path = output_dir / (f"{filename}_q{quality:.0%}_{reco_cuts_label}_{bin_width:.0f}MeV"
                         f"_{filename_prefix}_{apa}.png").__str__().replace('%', '')
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return Path(path)


# ============================================================================
# PLOT 3 -- SELECTION EFFICIENCY
# ============================================================================

# How many SIGNAL interactions an event must hold to belong to each class. The
# count is over every channel, not one: an event with a numu CC and an NC in it is
# a two-neutrino event for both of them, because what matters is how many
# neutrinos the reconstruction had to tell apart.
MULTIPLICITY_CLASSES = {
    'all':    lambda n: True,
    'single': lambda n: n == 1,
    'multi':  lambda n: n > 1,
}


def count_signal_interactions_per_event(vertex_records, min_true_energy=MIN_TRUE_ENERGY_MEV):
    """
    {event: number of in-volume signal interactions depositing above threshold} --
    the same population the efficiency denominator is drawn from, so an event's
    multiplicity is counted in exactly the terms the efficiency is measured in.
    """
    counts = {}
    for vertex in vertex_records or []:
        if (vertex.get('vertex_in_volume') is True
                and vertex.get('interaction_channel')
                and (vertex.get('precut_energy_MeV') or 0) > min_true_energy):
            counts[vertex['event']] = counts.get(vertex['event'], 0) + 1
    return counts


def build_selection_efficiency(categorized_records, vertex_records, channel,
                               bin_width=ENERGY_BIN_WIDTH_MEV, threshold=HIGH_SIGNAL_THRESHOLD,
                               min_true_energy=MIN_TRUE_ENERGY_MEV,
                               multiplicity_class='all'):
    """
    Numerators and denominator for one channel's efficiency, binned in TRUE
    deposited energy.

    DENOMINATOR: every true interaction of this channel with its vertex in volume
    AND more than min_true_energy of deposited energy -- including the ones that
    produced no selected reco cluster at all, which are exactly the inefficiency
    being measured. Binned by precut_energy_MeV: the summed energy of the true
    cluster's points BEFORE the fiducial and energy cuts.

    Why pre-cut and not the post-cut sum: the post-cut sum is None for precisely
    the interactions whose cluster did not survive, so using it would drop them
    from the denominator -- 42% of the in-volume NC interactions in the sample this
    was built on -- and inflate the efficiency by deleting its own failures.

    NUMERATORS, both binned by the same pre-cut energy of the pair's true cluster:
      high signal : true clusters with a high_signal_<channel> reco
      good+bad    : true clusters with any in-volume pair of this channel

    COUNTED AS DISTINCT TRUE CLUSTERS, not as pairs. The matching is reco-centric,
    so one true neutrino split into two reco clusters yields two pairs; counting
    pairs then puts more in the numerator than the denominator can hold and the
    "efficiency" exceeds 1 (it reached 1.04 for numu CC on the sample this was
    built against, from 51 pairs over 44 distinct true clusters). Counting the
    true cluster once makes each curve what its axis claims: the fraction of
    signal interactions that produced at least one qualifying reco cluster,
    bounded by 1 by construction.

    This changes nothing for the high-signal curve in practice -- two reco clusters
    cannot both exceed 80% completeness of the same true cluster without heavily
    overlapping each other, and n_true_multi_pair_high was 0 -- but it makes that
    curve bounded by construction rather than by luck.

    Returns a dict with edges, denominator, numerator_high, numerator_any and
    n_true_multi_pair.
    """
    completeness_min, purity_min = threshold_pair(threshold)
    edges = energy_bin_edges(bin_width)

    # Multiplicity is a property of the EVENT, so it is counted once over every
    # channel and then applied to this channel's interactions.
    per_event = count_signal_interactions_per_event(vertex_records, min_true_energy)
    keep_multiplicity = MULTIPLICITY_CLASSES[multiplicity_class]

    in_volume_vertices = [v for v in (vertex_records or [])
                          if v.get('vertex_in_volume') is True
                          and v.get('interaction_channel') == channel
                          and keep_multiplicity(per_event.get(v['event'], 0))]
    with_energy = [v for v in in_volume_vertices if v.get('precut_energy_MeV')]
    denominator_energies = [v['precut_energy_MeV'] for v in with_energy
                            if v['precut_energy_MeV'] > min_true_energy]
    n_no_energy   = len(in_volume_vertices) - len(with_energy)
    n_below_energy = len(with_energy) - len(denominator_energies)

    precut_by_true = {(v['event'], v['cluster_id']): v.get('precut_energy_MeV')
                      for v in (vertex_records or []) if v.get('cluster_id') is not None}

    # {true cluster key: its pre-cut energy}, so each true cluster contributes once
    # however many reco clusters claimed it.
    any_true, high_true, relaxed_true = {}, {}, {}
    true_pair_counts, high_pair_counts = {}, {}
    for record in categorized_records or []:
        if record['channel'] != channel:
            continue
        if not (record['category'] == 'contaminated' or record['category'].startswith('high_signal_')):
            continue                       # out-of-volume and cosmic are not this channel's pairs
        if not keep_multiplicity(per_event.get(record['event'], 0)):
            continue
        key = (record['event'], record['pair_true_cluster_id'])
        energy = precut_by_true.get(key)
        # Same threshold as the denominator, so the numerator is a strict subset of
        # it. A matched cluster is above it anyway (it survived the true-side energy
        # cut, and the pre-cut sum can only be larger), but stating it here means the
        # two sets cannot drift apart if either cut changes.
        if not energy or energy <= min_true_energy:
            continue
        any_true[key] = energy
        true_pair_counts[key] = true_pair_counts.get(key, 0) + 1
        # Tested from the pair's own metrics rather than from the stored category,
        # so an efficiency can be built at ANY threshold from one categorisation --
        # the categories were fixed at HIGH_SIGNAL_THRESHOLD when the stack was
        # made, and re-running the whole job to move a threshold would be absurd.
        # At threshold == HIGH_SIGNAL_THRESHOLD this reproduces the categories
        # exactly.
        if ((record.get('pair_completeness') or 0) > completeness_min
                and (record.get('pair_purity') or 0) > purity_min):
            high_true[key] = energy
        # Same completeness bar, purity relaxed: isolates what the purity
        # requirement alone costs.
        if ((record.get('pair_completeness') or 0) > completeness_min
                and (record.get('pair_purity') or 0) > RELAXED_PURITY):
            relaxed_true[key] = energy
            high_pair_counts[key] = high_pair_counts.get(key, 0) + 1

    any_energies  = list(any_true.values())
    high_energies = list(high_true.values())
    relaxed_energies = list(relaxed_true.values())

    denominator, _ = np.histogram(denominator_energies, bins=edges)
    numerator_high, _ = np.histogram(high_energies, bins=edges)
    numerator_any, _ = np.histogram(any_energies, bins=edges)
    numerator_relaxed, _ = np.histogram(relaxed_energies, bins=edges)

    return {
        'channel':            channel,
        'multiplicity_class': multiplicity_class,
        'threshold':          threshold,
        'edges':              edges,
        'denominator':        denominator,
        'numerator_high':     numerator_high,
        'numerator_relaxed':  numerator_relaxed,
        'numerator_any':      numerator_any,
        'n_denominator':      len(denominator_energies),
        'n_denominator_no_energy': n_no_energy,
        'n_denominator_below_energy': n_below_energy,
        'min_true_energy':    min_true_energy,
        'n_high':             len(high_energies),
        'n_relaxed':          len(relaxed_energies),
        'n_any':              len(any_energies),
        # Split, because the two numerators are not equally exposed: a true cluster
        # can easily be shared by two partial reco clusters (both 'any'), while two
        # reco clusters both exceeding 80% completeness of the SAME true cluster
        # needs them to overlap heavily and is much rarer. If the high count is 0,
        # that curve is a per-interaction efficiency already.
        'n_true_multi_pair':      sum(1 for n in true_pair_counts.values() if n > 1),
        'n_true_multi_pair_high': sum(1 for n in high_pair_counts.values() if n > 1),
        'n_true_with_any_pair':   len(true_pair_counts),
        'n_true_with_high_pair':  len(high_pair_counts),
    }


# How each curve is drawn, in one place so the overlaid figure and the single-curve
# figures cannot disagree about a colour or a label.
# Where the coarse tail binnings start. Above this the per-bin denominators fall
# to one or two interactions and the curve swings between 0 and 1 on single
# events; merging the tail trades resolution for bins that can actually be read.
REBIN_SPLIT_MEV = 1000.0

# The binnings every efficiency figure is drawn in: (directory, filename tag,
# number of tail bins). None means the tail is left alone.
EFFICIENCY_BINNINGS = (
    ('default_bins',                None,      None),
    ('tail_2bins_above_1000MeV',    '_tail2',  2),
    ('tail_1bin_above_1000MeV',     '_tail1',  1),
)


def rebin_efficiency_tail(efficiency, n_tail_bins, split_at=REBIN_SPLIT_MEV,
                          axis_max=PLOT_X_MAX_MEV):
    """
    Merge everything above split_at into n_tail_bins equal bins.

    Pure arithmetic on the histograms build_selection_efficiency already made --
    numerator and denominator are counts, so merging bins is summing them. No
    second pass over the events, and the integrated numbers ('n_denominator',
    'n_high', 'n_any') are untouched because merging cannot change a total.

    The tail spans split_at to axis_max, and anything ABOVE axis_max is folded
    into the final bin rather than dropped: the fine binning runs to 5000 MeV
    while the drawn axis stops at 3000, so without the overflow this would
    silently lose the handful of interactions past the axis -- and those bins are
    exactly the ones the coarse binning exists to rescue.

    split_at must fall on an existing edge, which it does for every bin width in
    use here (100 and 200 MeV both divide 1000).

    Returns a new dict; the input is not modified. 'bin_width_label' carries the
    ORIGINAL fine width so filenames keep saying which run they came from rather
    than reporting the merged tail width.
    """
    edges = np.asarray(efficiency['edges'], dtype=float)
    head = edges[edges <= split_at]
    if len(head) == 0 or head[-1] != split_at:
        raise ValueError(f"{split_at} MeV is not a bin edge of this binning")

    tail_edges = np.linspace(split_at, axis_max, n_tail_bins + 1)[1:]
    new_edges = np.concatenate((head, tail_edges))

    # Which new bin each OLD bin belongs to. Old bins beyond axis_max land past
    # the end and are clamped onto the last one -- that is the overflow.
    old_centers = (edges[:-1] + edges[1:]) / 2
    index = np.clip(np.searchsorted(new_edges, old_centers, side='right') - 1,
                    0, len(new_edges) - 2)

    merged = dict(efficiency)
    merged['edges'] = new_edges
    # Every histogram in the dict, found by name rather than listed: a new
    # numerator added to build_selection_efficiency is rebinned automatically,
    # and cannot be left at the fine binning to collide with the merged
    # denominator at draw time.
    for key in (['denominator']
                + [k for k in efficiency if k.startswith('numerator_')]):
        counts = np.asarray(efficiency[key], dtype=float)
        merged[key] = np.bincount(index, weights=counts,
                                  minlength=len(new_edges) - 1)
    merged['bin_width_label'] = efficiency.get(
        'bin_width_label', float(edges[-1] - edges[-2]))
    return merged


def efficiency_bin_width_label(efficiency):
    """The bin width a filename should report: the fine one, even after rebinning."""
    edges = np.asarray(efficiency['edges'], dtype=float)
    return efficiency.get('bin_width_label', float(edges[-1] - edges[-2]))


# The purity bar for the "relaxed purity" curve. The COMPLETENESS bar stays at
# whatever threshold the figure is drawn for -- only purity is relaxed, which is
# what "relax the purity requirement" means.
RELAXED_PURITY = 0.10

EFFICIENCY_CURVES = {
    'numerator_any':  {'color': 'tab:gray', 'marker': 's', 'label': 'All Selected'},
    'numerator_high': {'color': 'tab:blue', 'marker': 'o',
                       'label': f'Signal (completeness & purity > {HIGH_SIGNAL_THRESHOLD:.0%})'},
    # Dashed, and the only curve that is: with the purity bar this loose it often
    # coincides with All Selected bin for bin, and a solid line under a solid line
    # is invisible -- the reader cannot tell "hidden" from "not drawn". Dashes
    # show through.
    'numerator_relaxed': {'color': 'tab:purple', 'marker': 'D', 'linestyle': '--',
                          'zorder': 4,
                          'label': f'Signal, purity relaxed to > {RELAXED_PURITY:.0%}'},
}

# The figures drawn per channel: (filename tag, curves on it). The overlaid one is
# where the two are compared -- the gap between them is exactly "matched, but not
# well reconstructed" -- and the single-curve ones are for showing one result on
# its own without the other competing for the axes.
EFFICIENCY_CURVE_SETS = [
    ('both',    ('numerator_any', 'numerator_high')),
    ('high',    ('numerator_high',)),
    ('goodbad', ('numerator_any',)),
    # Purity relaxed to RELAXED_PURITY: alone, against the signal curve, and
    # against both signal and All Selected. The three answer "how much of the
    # signal is lost to the purity bar alone" at increasing context.
    ('relaxed',          ('numerator_relaxed',)),
    ('relaxed_signal',   ('numerator_relaxed', 'numerator_high')),
    ('relaxed_signal_all', ('numerator_relaxed', 'numerator_high', 'numerator_any')),
]


# Confidence level for the efficiency bands: 68.27%, i.e. the "one sigma" a
# physics reader assumes when a band is drawn without being told otherwise.
EFFICIENCY_CL = 0.6827

# How much of the line colour the band keeps. Low enough that the line and its
# markers stay legible on top of it, high enough to read as the same colour.
_BAND_ALPHA = 0.22


def clopper_pearson(numerator, denominator, confidence_level=EFFICIENCY_CL):
    """
    Clopper-Pearson interval on each bin's efficiency, as (low, high) arrays.

    BINOMIAL, not Poisson. The numerator here is a SUBSET of the denominator --
    the true clusters that were selected, out of the interactions that could have
    been -- so the count that fluctuates is a number of successes in a fixed
    number of trials. sqrt(N) on either count would be the wrong distribution and
    would put the band outside [0, 1].

    Clopper-Pearson rather than the normal approximation because the bins that
    most need an interval are the ones the approximation handles worst: above
    ~1500 MeV the denominators fall to one or two interactions and the ratio sits
    at exactly 0 or 1, where the normal interval has ZERO width and the plot
    claims perfect knowledge of the bin it knows least about. Clopper-Pearson
    gives 0/1 -> [0, 0.84] and 1/1 -> [0.16, 1] at this confidence level, which
    reads as "no statistics here" instead.

    It is the conservative choice -- coverage is at least the nominal level, and
    often more -- which is the right way to be wrong on a plot whose tail bins
    hold single interactions.

    Bins with an empty denominator get (nan, nan); they are dropped by the caller
    rather than drawn as an efficiency of zero.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    alpha = 1.0 - confidence_level

    low  = np.full(denominator.shape, np.nan)
    high = np.full(denominator.shape, np.nan)
    filled = denominator > 0
    if not np.any(filled):
        return low, high

    k = numerator[filled]
    n = denominator[filled]
    # The beta quantiles are undefined at k = 0 and k = n; those endpoints are
    # exactly 0 and 1 respectively, which np.where supplies after the fact. The
    # shape arguments are clipped only to keep the ppf call itself finite -- the
    # values it returns there are discarded.
    lo = beta.ppf(alpha / 2, np.maximum(k, 1e-9), n - k + 1)
    hi = beta.ppf(1 - alpha / 2, k + 1, np.maximum(n - k, 1e-9))
    low[filled]  = np.where(k <= 0, 0.0, lo)
    high[filled] = np.where(k >= n, 1.0, hi)
    return low, high


def draw_efficiency_band(ax, centers, low, high, color):
    """
    The uncertainty band for one curve: a light fill in the curve's own colour.

    Drawn UNDER the line (low zorder) and at _BAND_ALPHA, so the efficiency line
    and its markers stay the thing you read first -- the band is context, not the
    measurement.
    """
    ax.fill_between(centers, low, high, color=color, alpha=_BAND_ALPHA,
                    linewidth=0, zorder=1)


def draw_efficiency_bars(ax, centers, ratio, low, high, color):
    """
    The same interval as a bar on each point, in the curve's own colour.

    ASYMMETRIC by construction -- Clopper-Pearson is not centred on the ratio, and
    at the endpoints it is wildly off-centre (1/1 gives [0.16, 1], a bar with no
    upper half at all). Passing yerr as a 2 x N array preserves that; a single
    symmetric error would misstate exactly the bins the interval exists for.

    Bars rather than a band where several curves share the axes: N bands overlap
    into a wash whose colour belongs to no curve, while N sets of bars stay
    attached to their own points.
    """
    ratio = np.asarray(ratio, dtype=float)
    yerr = np.vstack((ratio - np.asarray(low, dtype=float),
                      np.asarray(high, dtype=float) - ratio))
    # Clip the rounding noise that can make a bound sit a float epsilon the wrong
    # side of the ratio; matplotlib rejects a negative yerr outright.
    yerr = np.clip(yerr, 0.0, None)
    ax.errorbar(centers, ratio, yerr=yerr, fmt='none', ecolor=color,
                elinewidth=1.4, capsize=3, capthick=1.4, zorder=2)


# The uncertainty styles a figure can be drawn in. None draws no interval at all.
UNCERTAINTY_STYLES = (None, 'band', 'bars')

# Filename tag and directory name per style, so the three versions of a figure
# never collide and the directory says which is which.
_UNCERTAINTY_TAG = {None: '', 'band': '_err', 'bars': '_errbar'}
UNCERTAINTY_DIRS = {None: 'no_uncertainty', 'band': 'with_uncertainty_band',
                    'bars': 'with_uncertainty_bars'}


def draw_efficiency_uncertainty(ax, style, centers, ratio, numerator, denominator, color):
    """Draw one curve's interval in the requested style; no-op when style is None."""
    if not style:
        return
    low, high = clopper_pearson(numerator, denominator)
    if style == 'band':
        draw_efficiency_band(ax, centers, low, high, color)
    elif style == 'bars':
        draw_efficiency_bars(ax, centers, ratio, low, high, color)
    else:
        raise ValueError(f"unknown uncertainty style {style!r}")


def draw_selection_efficiency(efficiency, output_dir, level_name, filename_prefix, apa,
                              reco_cuts_label='AfterBeamWindowCut',
                              curves=('numerator_any', 'numerator_high'), curve_set_label='both',
                              filename='selection_efficiency', uncertainty=None):
    """
    One channel's efficiency against true deposited energy.

    `curves` selects which of EFFICIENCY_CURVES to draw and `curve_set_label` goes
    in the filename, so the overlaid figure and each single-curve figure coexist.
    All of them share one y range (0 to 1.25) so a curve looks the same wherever
    it is drawn.

    Bins where the denominator is empty are left out rather than drawn as zero --
    no interactions of this channel deposited that much energy, which is not an
    efficiency of zero.

    uncertainty ('band', 'bars' or None) adds a Clopper-Pearson interval to the
    HIGH-SIGNAL curve only. Not to good+bad: the two share a denominator, so their
    intervals are correlated, and drawing both invites the reader to compare them
    as if they were independent.
    """
    # One subdirectory per signal definition, so a reader after a particular
    # threshold opens one folder instead of filtering a flat directory by tag.
    output_dir = Path(output_dir) / threshold_dirname(
        efficiency.get('threshold', HIGH_SIGNAL_THRESHOLD))
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = efficiency['edges']
    centers = (edges[:-1] + edges[1:]) / 2
    denominator = efficiency['denominator'].astype(float)
    filled = denominator > 0

    # Short names for the overall-efficiency box, keyed like EFFICIENCY_CURVES.
    _OVERALL_NAMES = {'numerator_high': 'Signal', 'numerator_any': 'All Selected',
                      'numerator_relaxed': 'Signal (relaxed)'}
    n_denominator = efficiency.get('n_denominator', 0)
    overall_lines = []

    fig, ax = plt.subplots(figsize=(10, 7))
    for numerator_key in curves:
        style = EFFICIENCY_CURVES[numerator_key]
        label = style['label']
        # Overall = numerator TOTAL / denominator TOTAL (includes any interaction
        # above the plot axis), i.e. the single number quoted for the channel --
        # the same value efficiency.txt's AVERAGE EFFICIENCY block carries.
        n_num = efficiency.get('n_' + numerator_key[len('numerator_'):])
        if n_num is not None and n_denominator:
            overall_lines.append(
                f"{_OVERALL_NAMES.get(numerator_key, numerator_key)} Selection Efficiency = "
                f"{100.0 * n_num / n_denominator:.1f}%")
        if numerator_key == 'numerator_high':
            # threshold_label already carries the metric names, so no fixed prefix
            # here -- adding one produced "completeness & purity > completeness >
            # 60%, purity > 80%" on the asymmetric variants.
            label = f"Signal ({threshold_label(efficiency.get('threshold', HIGH_SIGNAL_THRESHOLD))})"
        if numerator_key == 'numerator_relaxed':
            completeness_min, _ = threshold_pair(
                efficiency.get('threshold', HIGH_SIGNAL_THRESHOLD))
            label = (f"Signal, completeness > {completeness_min:.0%}, "
                     f"purity > {RELAXED_PURITY:.0%}")
        numerator = efficiency[numerator_key].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(filled, numerator / denominator, np.nan)
        # Every curve gets its interval, including All Selected. They share a
        # denominator, so the bands are correlated -- two of them touching says
        # nothing about whether the curves differ.
        draw_efficiency_uncertainty(ax, uncertainty, centers[filled], ratio[filled],
                                    numerator[filled], denominator[filled], style['color'])
        ax.plot(centers[filled], ratio[filled], marker=style['marker'], color=style['color'],
                linestyle=style.get('linestyle', '-'), linewidth=1.8, markersize=6,
                label=label, zorder=style.get('zorder', 3))

    channel_label = {'numu_CC': r'$\nu_\mu$ CC', 'nue_CC': r'$\nu_e$ CC',
                     'NC': 'NC'}.get(efficiency['channel'], efficiency['channel'])
    ax.set_xlabel('True Deposited Cluster Energy (MeV)',
                  fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Selection Efficiency', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    set_fitted_title(ax, f'{channel_label} selection efficiency, vertex in volume '
                 f'-- reco: {reco_cuts_label}',
                 _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(edges[0], PLOT_X_MAX_MEV)
    ax.xaxis.set_major_locator(MultipleLocator(ENERGY_AXIS_TICK_MEV))
    # A ratio of counts: 1 is the ceiling, and headroom above it makes an
    # over-unity bin (see n_true_multi_pair) visible instead of clipped away.
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color='black', linewidth=1.0, linestyle=':')
    ax.legend(fontsize=_LEGEND_FONTSIZE, loc='upper right', framealpha=0.9)

    # Overall (integrated) efficiency for each curve, top-left so it clears the
    # upper-right legend.
    if overall_lines:
        ax.text(0.03, 0.97, "\n".join(overall_lines),
                transform=ax.transAxes, ha='left', va='top',
                fontsize=_LEGEND_FONTSIZE - 1, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.88, edgecolor='0.6'))

    bin_width = efficiency_bin_width_label(efficiency)
    tag = threshold_tag(efficiency.get('threshold', HIGH_SIGNAL_THRESHOLD))
    path = output_dir / (f"{filename}_{efficiency['channel']}_{reco_cuts_label}"
                         f"_{bin_width:.0f}MeV{tag}_{curve_set_label}"
                         f"{efficiency.get('binning_tag') or ''}"
                         f"{_UNCERTAINTY_TAG[uncertainty]}"
                         f"_{filename_prefix}_{apa}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return path


# ============================================================================
# NEUTRINO MULTIPLICITY
# ============================================================================

_MULTIPLICITY_STYLE = {'single': {'color': 'tab:blue', 'marker': 'o', 'label': '1 neutrino'},
                       'multi':  {'color': 'tab:red',  'marker': 's', 'label': '2+ neutrinos'},
                       'all':    {'color': 'black',    'marker': 'D', 'label': '1 and 2+ neutrinos'}}

# Which classes go on each version of the by-multiplicity figure.
#   'split'    1 neutrino against 2+ -- do they differ?
#   'vs_all'   1 neutrino against the combined sample -- how much do the 2+
#              events drag the overall number, which is the one usually quoted
MULTIPLICITY_FIGURES = [('split', ('single', 'multi')),
                        ('vs_all', ('single', 'all'))]


def draw_neutrino_multiplicity(vertex_records, output_dir, level_name, filename_prefix, apa,
                               min_true_energy=MIN_TRUE_ENERGY_MEV,
                               filename='signal_neutrino_multiplicity'):
    """
    How many events hold one signal interaction, how many hold two, and so on.

    This is the context every efficiency number needs: an efficiency measured on a
    sample that is mostly single-neutrino events describes single-neutrino events,
    whatever else is in it. Counted over the SAME population the denominator uses
    (in volume, above threshold, any channel), so the two cannot disagree about
    what a signal interaction is.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_event = count_signal_interactions_per_event(vertex_records, min_true_energy)
    if not per_event:
        return None
    counts = {}
    for n in per_event.values():
        counts[n] = counts.get(n, 0) + 1
    multiplicities = sorted(counts)
    n_events = sum(counts.values())
    n_interactions = sum(n * c for n, c in counts.items())

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.bar(multiplicities, [counts[n] for n in multiplicities],
                  color='tab:blue', edgecolor='black', linewidth=1.0, width=0.6)
    for n, bar in zip(multiplicities, bars):
        share_events = 100.0 * counts[n] / n_events
        share_inter  = 100.0 * n * counts[n] / n_interactions
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{counts[n]}\n{share_events:.1f}% of events\n{share_inter:.1f}% of interactions',
                ha='center', va='bottom', fontsize=_LEGEND_FONTSIZE - 2)

    ax.set_xlabel('Signal neutrino interactions in the event',
                  fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    set_fitted_title(ax, f'Signal neutrino multiplicity ({n_events} events, {n_interactions} interactions)',
                 _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.set_xticks(multiplicities)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(counts.values()) * 1.35)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    path = output_dir / f"{filename}_{filename_prefix}_{apa}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return path


def draw_efficiency_by_multiplicity(efficiencies_by_class, output_dir, level_name,
                                    filename_prefix, apa, numerator_key='numerator_high',
                                    reco_cuts_label='AfterBeamWindowCut',
                                    filename='selection_efficiency_by_multiplicity',
                                    uncertainty=None, classes=('single', 'multi'),
                                    set_label='split'):
    """
    One channel's efficiency for single-neutrino events and for multi-neutrino
    events, on one axes, for one numerator.

    Drawn together rather than as two separate figures because the question is
    whether they differ, and two curves on one scale answer it where two files do
    not. Bins empty in a class are dropped for that class only, so a sparse
    multi-neutrino curve does not pull the single-neutrino one about.

    efficiencies_by_class: {'single': eff, 'multi': eff} from
    build_selection_efficiency with the matching multiplicity_class.

    uncertainty applies only to the high-signal numerator, so the good+bad version
    of this figure is identical in all three styles. Here the intervals earn their
    place: the multi-neutrino class has an order of magnitude fewer interactions
    than the single, and without them the two curves look equally well measured.
    """
    present = {k: e for k, e in efficiencies_by_class.items()
               if k in classes and k in _MULTIPLICITY_STYLE and e['n_denominator']}
    if not present:
        return None
    channel = next(iter(present.values()))['channel']

    # Same signal-definition subdirectory as draw_selection_efficiency, so the
    # multiplicity split for a threshold sits beside that threshold's other
    # figures. All classes on one figure share the threshold (one build call).
    output_dir = Path(output_dir) / threshold_dirname(
        next(iter(present.values())).get('threshold', HIGH_SIGNAL_THRESHOLD))
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for class_name, efficiency in present.items():
        style = _MULTIPLICITY_STYLE[class_name]
        edges = efficiency['edges']
        centers = (edges[:-1] + edges[1:]) / 2
        denominator = efficiency['denominator'].astype(float)
        filled = denominator > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(filled, efficiency[numerator_key].astype(float) / denominator, np.nan)
        if numerator_key == 'numerator_high':
            draw_efficiency_uncertainty(
                ax, uncertainty, centers[filled], ratio[filled],
                efficiency[numerator_key].astype(float)[filled], denominator[filled],
                style['color'])
        ax.plot(centers[filled], ratio[filled], marker=style['marker'], color=style['color'],
                linestyle='-', linewidth=1.8, markersize=6, zorder=3,
                label=f"{style['label']} ({efficiency['n_denominator']} interactions)")

    channel_label = {'numu_CC': r'$\nu_\mu$ CC', 'nue_CC': r'$\nu_e$ CC',
                     'NC': 'NC'}.get(channel, channel)
    # The signal definition these efficiencies were built at. Taken from the
    # efficiency dict rather than passed in, so the label cannot disagree with
    # the numbers. All classes on one figure share it -- they come from one build
    # call -- so reading it off the first is enough.
    threshold = next(iter(present.values())).get('threshold', HIGH_SIGNAL_THRESHOLD)
    curve_label = EFFICIENCY_CURVES[numerator_key]['label']
    if numerator_key == 'numerator_high':
        curve_label = f'Signal ({threshold_label(threshold)})'
    ax.set_xlabel('True Deposited Cluster Energy (MeV)',
                  fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Selection Efficiency', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    set_fitted_title(ax, f'{channel_label}, {curve_label} -- by event multiplicity',
                 _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    edges = next(iter(present.values()))['edges']
    ax.set_xlim(edges[0], PLOT_X_MAX_MEV)
    ax.xaxis.set_major_locator(MultipleLocator(ENERGY_AXIS_TICK_MEV))
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color='black', linewidth=1.0, linestyle=':')
    ax.legend(fontsize=_LEGEND_FONTSIZE, loc='upper right', framealpha=0.9)

    bin_width = efficiency_bin_width_label(next(iter(present.values())))
    curve_tag = 'high' if numerator_key == 'numerator_high' else 'goodbad'
    # Only the signal curve depends on the threshold -- All Selected counts any
    # matched pair -- so only it carries the tag. Tagging both would write the
    # same good+bad figure once per threshold under several different names.
    if numerator_key == 'numerator_high':
        curve_tag = f"{threshold_tag(threshold).lstrip('_')}_{curve_tag}"
    path = output_dir / (f"{filename}_{channel}_{reco_cuts_label}_{bin_width:.0f}MeV"
                         f"_{curve_tag}_{set_label}"
                         f"{next(iter(present.values())).get('binning_tag') or ''}"
                         f"{_UNCERTAINTY_TAG[uncertainty]}"
                         f"_{filename_prefix}_{apa}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return path


# One per entry in EFFICIENCY_THRESHOLDS -- zip() against the sorted thresholds
# silently drops any curve past the end of this list, so keep it at least as long.
_THRESHOLD_COLORS = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                     'tab:olive']


def draw_efficiency_threshold_comparison(efficiencies_by_threshold, output_dir, level_name,
                                         filename_prefix, apa,
                                         reco_cuts_label='AfterBeamWindowCut',
                                         filename='selection_efficiency_thresholds',
                                         uncertainty=None):
    """
    One channel's efficiency at every threshold, plus good+bad, on one axes.

    This is the figure the separate ones cannot give: good+bad is the ceiling --
    the fraction of signal the selection finds at all -- and each threshold curve
    below it says how much of that survives a quality requirement. The vertical
    gaps between the curves are where reconstruction quality is being lost, and
    reading them together shows whether 80% is a cliff or a gentle slope.

    All curves share one denominator, so they are directly comparable and ordered:
    good+bad >= 50% >= 70% >= 80% in every bin, by construction.

    efficiencies_by_threshold: {threshold: efficiency dict}, all for the same
    channel, bin width and multiplicity class.

    uncertainty applies to every THRESHOLD curve -- each is a high-signal
    efficiency -- but not to the grey good+bad ceiling. Read the overlaps with
    care: all four curves come from one denominator and are nested by
    construction, so two intervals touching says nothing about whether the
    thresholds differ.

    'bars' is the better style HERE. Three bands over one axes compound their
    alpha wherever they overlap, and past ~1200 MeV -- where the denominators
    thin out and every interval opens up -- they merge into a single wash that
    hides the curves it is supposed to qualify.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    present = {t: e for t, e in efficiencies_by_threshold.items() if e['n_denominator']}
    if not present:
        return None
    any_efficiency = next(iter(present.values()))
    channel = any_efficiency['channel']
    edges = any_efficiency['edges']
    centers = (edges[:-1] + edges[1:]) / 2
    denominator = any_efficiency['denominator'].astype(float)
    filled = denominator > 0

    fig, ax = plt.subplots(figsize=(10, 7))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_any = np.where(filled, any_efficiency['numerator_any'].astype(float) / denominator, np.nan)
    ax.plot(centers[filled], ratio_any[filled], marker='s', color='tab:gray',
            linestyle='--', linewidth=2.0, markersize=6,
            label=f"All Selected ({any_efficiency['n_denominator']} interactions)")

    for color, threshold in zip(_THRESHOLD_COLORS,
                                sorted(present, key=threshold_pair, reverse=True)):
        efficiency = present[threshold]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(filled, efficiency['numerator_high'].astype(float) / denominator, np.nan)
        draw_efficiency_uncertainty(ax, uncertainty, centers[filled], ratio[filled],
                                    efficiency['numerator_high'].astype(float)[filled],
                                    denominator[filled], color)
        ax.plot(centers[filled], ratio[filled], marker='o', color=color,
                linestyle='-', linewidth=1.8, markersize=6, zorder=3,
                label=f"{threshold_label(threshold)}  "
                      f"(integrated {efficiency['n_high'] / efficiency['n_denominator']:.3f})")

    channel_label = {'numu_CC': r'$\nu_\mu$ CC', 'nue_CC': r'$\nu_e$ CC',
                     'NC': 'NC'}.get(channel, channel)
    ax.set_xlabel('True Deposited Cluster Energy (MeV)',
                  fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Selection Efficiency', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    set_fitted_title(ax, f'{channel_label} selection efficiency vs quality threshold',
                 _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(edges[0], PLOT_X_MAX_MEV)
    ax.xaxis.set_major_locator(MultipleLocator(ENERGY_AXIS_TICK_MEV))
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color='black', linewidth=1.0, linestyle=':')
    ax.legend(fontsize=_LEGEND_FONTSIZE - 1, loc='lower right', framealpha=0.9)

    bin_width = efficiency_bin_width_label(any_efficiency)
    path = output_dir / (f"{filename}_{channel}_{reco_cuts_label}_{bin_width:.0f}MeV"
                         f"{any_efficiency.get('binning_tag') or ''}"
                         f"{_UNCERTAINTY_TAG[uncertainty]}"
                         f"_{filename_prefix}_{apa}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return path


# ============================================================================
# THE NUMBERS BEHIND THE PLOTS
# ============================================================================

def write_selection_performance_info(by_key, categorized_records, efficiencies, output_dir,
                                     level_name, components=None,
                                     bin_width=ENERGY_BIN_WIDTH_MEV,
                                     reco_cuts_label='AfterBeamWindowCut',
                                     filename=None):
    """Category counts, the closure check, and each channel's efficiency table."""
    components = components if components is not None else SELECTION_COMPONENTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (filename or f"selection_performance_info_{reco_cuts_label}_"
                                     f"{bin_width:.0f}MeV.txt")

    n_total = len(categorized_records or [])
    lines = []
    lines.append("=" * 88)
    lines.append(f"SELECTION PERFORMANCE ({level_name}) -- reco selection: {reco_cuts_label}")
    lines.append("=" * 88)
    lines.append(f"Reco energy = {RECO_WORK_FUNCTION_EV} eV * charge / {RECO_RECOMBINATION_FACTOR}")
    lines.append(f"Match requires purity >= {MIN_MATCH_PURITY}; high signal requires "
                 f"completeness AND purity > {HIGH_SIGNAL_THRESHOLD}")
    lines.append("")
    lines.append("Categories of the selected reco clusters:")
    for component in order_selection_components_for_legend(components, by_key):
        n = len(by_key.get(component['key'], []))
        share = (100.0 * n / n_total) if n_total else 0.0
        lines.append(f"  {component['key']:<24s} {n:6d}  ({share:5.1f}%)")
    n_stacked = sum(len(v) for v in by_key.values())
    lines.append(f"  {'TOTAL':<24s} {n_stacked:6d}")
    lines.append(f"  selected reco clusters   {n_total:6d}"
                 f"   {'MATCH' if n_stacked == n_total else 'MISMATCH -- categories lost clusters'}")
    lines.append("")

    for efficiency in efficiencies or []:
        lines.append("-" * 88)
        lines.append(f"EFFICIENCY -- {efficiency['channel']}, vertex in volume")
        lines.append("-" * 88)
        lines.append(f"  denominator (true interactions): {efficiency['n_denominator']}")
        if efficiency['n_denominator_no_energy']:
            lines.append(f"    NOTE {efficiency['n_denominator_no_energy']} had no pre-cut cluster "
                         f"energy and are NOT in the denominator")
        lines.append(f"  numerator, high signal:          {efficiency['n_high']}")
        lines.append(f"  numerator, good+bad:             {efficiency['n_any']}")
        if efficiency['n_denominator']:
            lines.append(f"  integrated efficiency, high:     "
                         f"{efficiency['n_high'] / efficiency['n_denominator']:.4f}")
            lines.append(f"  integrated efficiency, good+bad: "
                         f"{efficiency['n_any'] / efficiency['n_denominator']:.4f}")
        lines.append(f"  distinct true clusters with a pair: any {efficiency['n_true_with_any_pair']}, "
                     f"high {efficiency['n_true_with_high_pair']}")
        lines.append(f"  numerators count DISTINCT true clusters, so both ratios are bounded by 1")
        if efficiency['n_true_multi_pair']:
            lines.append(f"    ({efficiency['n_true_multi_pair']} true cluster(s) were claimed by more "
                         f"than one good+bad reco cluster and are counted once)")
        if efficiency['n_true_multi_pair_high']:
            lines.append(f"    ({efficiency['n_true_multi_pair_high']} true cluster(s) were claimed by "
                         f"more than one HIGH-SIGNAL reco cluster and are counted once)")
        lines.append("")
        edges = efficiency['edges']
        lines.append(f"  {'energy bin [MeV]':<22s}{'denom':>8s}{'high':>8s}{'any':>8s}"
                     f"{'eff_high':>11s}{'eff_any':>10s}")
        for i in range(len(edges) - 1):
            d = int(efficiency['denominator'][i])
            if d == 0:
                continue
            h, a = int(efficiency['numerator_high'][i]), int(efficiency['numerator_any'][i])
            lines.append(f"  {f'{edges[i]:.0f} - {edges[i+1]:.0f}':<22s}{d:>8d}{h:>8d}{a:>8d}"
                         f"{h / d:>11.3f}{a / d:>10.3f}")
        lines.append("")
    lines.append("=" * 88)

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return path


def _efficiency_totals(efficiency, split_mev=EFFICIENCY_SPLIT_MEV):
    """
    (all, below, above): three {'denom','high','any'} dicts for one channel's
    efficiency, split at split_mev of true deposited energy.

    'all' is the integrated count (includes any interaction above the plot axis).
    'below' sums the histogram bins whose left edge is < split_mev; 'above' is
    the remainder, so an interaction beyond the plot axis -- always high energy --
    lands there. A bin that straddles the split (only possible when split_mev is
    off the 100 MeV grid) is assigned by its left edge.
    """
    edges = np.asarray(efficiency['edges'])
    below_bins = edges[:-1] < split_mev
    counts = {'denom': efficiency['denominator'],
              'high':  efficiency['numerator_high'],
              'any':   efficiency['numerator_any']}
    total = {'denom': efficiency['n_denominator'],
             'high':  efficiency['n_high'],
             'any':   efficiency['n_any']}
    below = {k: int(np.asarray(v)[below_bins].sum()) for k, v in counts.items()}
    above = {k: total[k] - below[k] for k in total}
    return total, below, above


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def write_efficiency_summary(efficiencies_by_width, output_dir, level_name,
                             reco_cuts_label='AfterBeamWindowCut', filename='efficiency.txt',
                             efficiencies_by_multiplicity=None, vertex_records=None):
    """
    Every case's efficiency: the integrated number to quote, then the same thing
    bin by bin so the integrated value can be checked against where it comes from.

    Parameters:
    - efficiencies_by_width: {bin width: [build_selection_efficiency, ...]}. The
        integrated counts are taken before any histogramming and are identical at
        every width, so they come from whichever is present; the per-bin table
        uses the FINEST width available, being the one worth eyeballing.

    Integrated values cover ALL energies, including any interaction above the
    plotted axis, so they can differ slightly from a reader's sum down the per-bin
    column. Where that happens it is stated per channel below.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    if not efficiencies_by_width:
        return None
    table_width  = min(efficiencies_by_width)
    efficiencies = efficiencies_by_width[table_width]

    lines = []
    lines.append("=" * 92)
    lines.append(f"SELECTION EFFICIENCY -- INTEGRATED ({level_name}), reco selection: {reco_cuts_label}")
    lines.append("=" * 92)
    lines.append("")
    lines.append("DENOMINATOR  every true interaction of the channel with its vertex IN VOLUME,")
    lines.append("             from mc.json -- including those that produced no selected reco")
    lines.append("             cluster at all, which is the inefficiency being measured.")
    lines.append("NUMERATORS   distinct true clusters that produced at least one selected reco:")
    lines.append(f"  high signal  its paired reco has completeness AND purity > {HIGH_SIGNAL_THRESHOLD:.0%}")
    lines.append("  good+bad     any in-volume pair of that channel, whatever its quality")
    lines.append("")
    lines.append(f"A reco cluster counts as matched at purity >= {MIN_MATCH_PURITY}. Numerators count")
    lines.append("DISTINCT TRUE CLUSTERS, so one neutrino split into several reco clusters counts")
    lines.append("once and both ratios are bounded by 1.")
    lines.append("")
    lines.append(f"Only interactions depositing MORE THAN {MIN_TRUE_ENERGY_MEV:.0f} MeV are in the")
    lines.append("denominator: below that the true-side energy cut removes the cluster, so no")
    lines.append("selection could have found it and charging the selection for the loss would")
    lines.append("measure the cut instead. Energy is the pre-cut cluster sum, the same quantity")
    lines.append("the bins below use.")
    lines.append("")
    header = (f"  {'channel':<10s}{'denominator':>13s}{'high signal':>13s}{'eff high':>11s}"
              f"{'good+bad':>11s}{'eff good+bad':>14s}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for efficiency in efficiencies or []:
        denominator = efficiency['n_denominator']
        eff_high = efficiency['n_high'] / denominator if denominator else float('nan')
        eff_any  = efficiency['n_any']  / denominator if denominator else float('nan')
        lines.append(f"  {efficiency['channel']:<10s}{denominator:>13d}{efficiency['n_high']:>13d}"
                     f"{eff_high:>11.4f}{efficiency['n_any']:>11d}{eff_any:>14.4f}")
    lines.append("")

    for efficiency in efficiencies or []:
        notes = []
        if efficiency['n_denominator_no_energy']:
            notes.append(f"{efficiency['n_denominator_no_energy']} interaction(s) had no pre-cut "
                         f"cluster energy and are NOT in the denominator")
        if efficiency.get('n_denominator_below_energy'):
            notes.append(f"{efficiency['n_denominator_below_energy']} interaction(s) deposited "
                         f"<= {efficiency.get('min_true_energy', MIN_TRUE_ENERGY_MEV):.0f} MeV and are "
                         f"NOT in the denominator")
        if efficiency['n_true_multi_pair']:
            notes.append(f"{efficiency['n_true_multi_pair']} true cluster(s) were claimed by more than "
                         f"one good+bad reco cluster; each counts once")
        if efficiency['n_true_multi_pair_high']:
            notes.append(f"{efficiency['n_true_multi_pair_high']} true cluster(s) were claimed by more "
                         f"than one high-signal reco cluster; each counts once")
        binned = int(efficiency['denominator'].sum())
        if binned != efficiency['n_denominator']:
            notes.append(f"{efficiency['n_denominator'] - binned} interaction(s) fall above the "
                         f"{ENERGY_AXIS_MAX_MEV:.0f} MeV plot axis and are in these totals but not the curve")
        if notes:
            lines.append(f"  {efficiency['channel']}:")
            for note in notes:
                lines.append(f"    - {note}")

    # AVERAGE efficiency: the integrated ratio (sum of numerator bins / sum of
    # denominator bins), which is the denominator-weighted mean of the per-bin
    # efficiencies -- the single number to quote -- and the same ratio split at
    # EFFICIENCY_SPLIT_MEV. ALL CHANNELS pools the signal channels' counts.
    split = EFFICIENCY_SPLIT_MEV
    lines.append("")
    lines.append("-" * 92)
    lines.append(f"AVERAGE EFFICIENCY -- integrated ratio, split at {split:.0f} MeV of true deposited energy")
    lines.append("-" * 92)
    lines.append("The integrated ratio = sum(numerator bins) / sum(denominator bins), i.e. the")
    lines.append("denominator-weighted mean of the per-bin efficiencies. Each range shows")
    lines.append(f"'high' (completeness AND purity > {HIGH_SIGNAL_THRESHOLD:.0%}) / 'g+b' (good+bad, any pair).")
    lines.append(f"Interactions above the {ENERGY_AXIS_MAX_MEV:.0f} MeV plot axis count in the >= {split:.0f} MeV range.")
    lines.append("")
    lines.append(f"  {'channel':<13s}{'all: high / g+b':>20s}{f'<{split:.0f}: high / g+b':>21s}{f'>={split:.0f}: high / g+b':>22s}")
    lines.append("  " + "-" * 74)

    def _avg_row(label, total, below, above):
        parts = []
        for scope in (total, below, above):
            parts.append(f"{_ratio(scope['high'], scope['denom']):>6.4f} / "
                         f"{_ratio(scope['any'], scope['denom']):>6.4f}")
        return f"  {label:<13s}{parts[0]:>20s}{parts[1]:>21s}{parts[2]:>22s}"

    pooled = {scope: {'denom': 0, 'high': 0, 'any': 0} for scope in ('all', 'below', 'above')}
    for efficiency in efficiencies or []:
        total, below, above = _efficiency_totals(efficiency, split)
        for name, d in (('all', total), ('below', below), ('above', above)):
            for k in ('denom', 'high', 'any'):
                pooled[name][k] += d[k]
        lines.append(_avg_row(efficiency['channel'], total, below, above))
    lines.append(_avg_row('ALL CHANNELS', pooled['all'], pooled['below'], pooled['above']))

    # Split by how many signal neutrinos the event held. An efficiency measured on a
    # sample that is mostly single-neutrino events describes single-neutrino events;
    # this is what says whether the two classes actually differ.
    if efficiencies_by_multiplicity:
        lines.append("")
        lines.append("-" * 92)
        lines.append("BY EVENT MULTIPLICITY -- signal interactions in the event, counted over all channels")
        lines.append("-" * 92)
        if vertex_records is not None:
            per_event = count_signal_interactions_per_event(vertex_records)
            spread = {}
            for n in per_event.values():
                spread[n] = spread.get(n, 0) + 1
            total_events = sum(spread.values())
            total_inter  = sum(n * c for n, c in spread.items())
            lines.append(f"  events with >=1 signal interaction: {total_events}"
                         f"   signal interactions: {total_inter}")
            for n in sorted(spread):
                lines.append(f"    {n} neutrino(s): {spread[n]:5d} events"
                             f"  ({100.0 * n * spread[n] / total_inter:5.1f}% of interactions)")
            lines.append("")
        lines.append(f"  {'channel':<10s}{'multiplicity':<14s}{'denom':>7s}{'high':>7s}"
                     f"{'eff high':>11s}{'good+bad':>10s}{'eff good+bad':>14s}")
        for channel, by_class in efficiencies_by_multiplicity.items():
            for class_name in ('single', 'multi'):
                efficiency = by_class.get(class_name)
                if not efficiency or not efficiency['n_denominator']:
                    continue
                d = efficiency['n_denominator']
                lines.append(f"  {channel:<10s}{_MULTIPLICITY_STYLE[class_name]['label']:<14s}"
                             f"{d:>7d}{efficiency['n_high']:>7d}{efficiency['n_high'] / d:>11.4f}"
                             f"{efficiency['n_any']:>10d}{efficiency['n_any'] / d:>14.4f}")

    # Bin by bin, at the finest binning drawn, so the integrated numbers above can
    # be checked against the distribution they came from.
    for efficiency in efficiencies or []:
        if not efficiency['n_denominator']:
            continue
        lines.append("")
        lines.append("-" * 92)
        lines.append(f"{efficiency['channel']} -- per {table_width:.0f} MeV bin of true deposited energy")
        lines.append("-" * 92)
        lines.append(f"  {'energy bin [MeV]':<20s}{'denominator':>13s}{'high signal':>13s}{'eff high':>11s}"
                     f"{'good+bad':>11s}{'eff good+bad':>14s}")
        edges = efficiency['edges']
        for i in range(len(edges) - 1):
            d = int(efficiency['denominator'][i])
            if d == 0:
                continue
            h, a = int(efficiency['numerator_high'][i]), int(efficiency['numerator_any'][i])
            lines.append(f"  {f'{edges[i]:.0f} - {edges[i+1]:.0f}':<20s}{d:>13d}{h:>13d}"
                         f"{h / d:>11.4f}{a:>11d}{a / d:>14.4f}")
        totals = (int(efficiency['denominator'].sum()), int(efficiency['numerator_high'].sum()),
                  int(efficiency['numerator_any'].sum()))
        lines.append(f"  {'TOTAL (binned)':<20s}{totals[0]:>13d}{totals[1]:>13d}"
                     f"{totals[1] / totals[0]:>11.4f}{totals[2]:>11d}{totals[2] / totals[0]:>14.4f}")
    lines.append("=" * 92)

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_efficiency_by_threshold_summary(binned_by_threshold, output_dir, level_name,
                                          reco_cuts_label='AfterBeamWindowCut',
                                          filename='efficiency_by_threshold.txt'):
    """
    The same per-bin numerator/denominator table as write_efficiency_summary,
    written once for EVERY entry in EFFICIENCY_THRESHOLDS rather than the
    default (completeness & purity > 80%) only.

    write_efficiency_summary is deliberately left alone -- it is the one-glance
    default-threshold summary every run has always had -- this is the extra
    file for when a comparison at a DIFFERENT signal definition (e.g.
    completeness_gt_60pc, purity unconstrained) is needed and would otherwise
    require re-running the job again: the numbers are cheap to histogram (they
    come from each pair's own completeness/purity, already computed) so nothing
    stops writing all six now that the job is already doing the work.

    binned_by_threshold: {channel: {threshold: build_selection_efficiency(...)}}
    at whatever bin width/rebinning the caller wants recorded (this function
    does not rebin) -- pass the notebook's own binned_by_threshold from inside
    the EFFICIENCY_BINNINGS_SELECTED loop to match exactly what a figure at
    that binning drew. numerator_any does not depend on the threshold (it is
    "any in-volume pair, any quality") and is printed at every threshold anyway
    so each block is self-contained and diffable against efficiency.txt.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    lines = ["=" * 92,
             f"SELECTION EFFICIENCY BY THRESHOLD ({level_name}), reco selection: {reco_cuts_label}",
             "=" * 92, "",
             "Same quantities as efficiency.txt (high signal = completeness AND purity",
             "above the threshold below; good+bad = any in-volume pair, any quality --",
             "identical at every threshold, repeated here so each block stands alone),",
             "at every signal definition this codebase draws (see EFFICIENCY_THRESHOLDS),",
             "at whichever bin width and rebinning this file was written for.", ""]

    for channel, by_threshold in binned_by_threshold.items():
        for threshold in EFFICIENCY_THRESHOLDS:
            efficiency = by_threshold.get(threshold)
            if efficiency is None or not efficiency['n_denominator']:
                continue
            lines.append("-" * 92)
            lines.append(f"{channel} -- {threshold_label(threshold)} "
                         f"[{threshold_dirname(threshold)}]")
            lines.append("-" * 92)
            lines.append(f"  {'energy bin [MeV]':<20s}{'denominator':>13s}{'high signal':>13s}"
                         f"{'eff high':>11s}{'good+bad':>11s}{'eff good+bad':>14s}")
            edges = efficiency['edges']
            for i in range(len(edges) - 1):
                d = int(efficiency['denominator'][i])
                if d == 0:
                    continue
                h, a = int(efficiency['numerator_high'][i]), int(efficiency['numerator_any'][i])
                lines.append(f"  {f'{edges[i]:.0f} - {edges[i+1]:.0f}':<20s}{d:>13d}{h:>13d}"
                             f"{h / d:>11.4f}{a:>11d}{a / d:>14.4f}")
            totals = (int(efficiency['denominator'].sum()), int(efficiency['numerator_high'].sum()),
                      int(efficiency['numerator_any'].sum()))
            lines.append(f"  {'TOTAL (binned)':<20s}{totals[0]:>13d}{totals[1]:>13d}"
                         f"{totals[1] / totals[0]:>11.4f}{totals[2]:>11d}{totals[2] / totals[0]:>14.4f}")
            # Integrated (all energies, including any interaction above the plot
            # axis) -- the number a comparison plot's text box should quote,
            # exactly like efficiency.txt's top summary table does for thr80.
            n_d, n_h, n_a = efficiency['n_denominator'], efficiency['n_high'], efficiency['n_any']
            lines.append(f"  {'TOTAL (integrated)':<20s}{n_d:>13d}{n_h:>13d}"
                         f"{n_h / n_d:>11.4f}{n_a:>11d}{n_a / n_d:>14.4f}")
            lines.append("")

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_selection_performance_root(by_key, all_energies, efficiencies, output_dir,
                                     components=None, bin_width=ENERGY_BIN_WIDTH_MEV,
                                     reco_cuts_label='AfterBeamWindowCut',
                                     filename='selection_performance_histograms.root'):
    """
    The stack's bands, the selected total and each channel's efficiency inputs, as
    TH1Ds, so the figures can be restyled without re-running the job.

    Efficiency is stored as its NUMERATOR and DENOMINATOR histograms rather than
    as a ratio: a stored ratio cannot be rebinned (the ratio of sums is not the
    sum of ratios), while these can, and dividing them reproduces the curve.
    """
    import uproot

    components = components if components is not None else SELECTION_COMPONENTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    edges = energy_bin_edges(bin_width)

    with uproot.recreate(path) as root_file:
        stack_dir = f"selection_{reco_cuts_label}_{bin_width:.0f}MeV"
        for component in components:
            values = [r['reco_energy_mev'] for r in by_key.get(component['key'], [])]
            counts, _ = np.histogram(values, bins=edges)
            root_file[f"{stack_dir}/{component['key']}"] = (counts, edges)
        total_counts, _ = np.histogram(all_energies or [], bins=edges)
        root_file[f"{stack_dir}/all_selected_reco"] = (total_counts, edges)

        for efficiency in efficiencies or []:
            eff_dir = f"efficiency_{efficiency['channel']}_{bin_width:.0f}MeV"
            root_file[f"{eff_dir}/denominator"]    = (efficiency['denominator'], efficiency['edges'])
            root_file[f"{eff_dir}/numerator_high"] = (efficiency['numerator_high'], efficiency['edges'])
            root_file[f"{eff_dir}/numerator_any"]  = (efficiency['numerator_any'], efficiency['edges'])
    return path
