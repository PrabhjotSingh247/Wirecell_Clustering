"""
SIGNAL AND BACKGROUND DISTRIBUTIONS -- driven by
AnalysisDistributions/SignalBackground_Distributions.ipynb.

A STACKED histogram of true cluster energy, one stack component per physics
category, so signal and the backgrounds it has to be separated from are read off
one axis with a common binning and a shared y scale.

WHAT IS IN THE STACK

Components are declared once, in SIGNAL_BACKGROUND_COMPONENTS below, and
everything else here is driven by that list: the colours, the legend, and the
counts written to the text table. Adding a background is adding one entry there
-- no drawing code changes. The ORDER of the stack is taken from that list too:
it is declared TOP-DOWN, and order_components_for_stack() reverses it at draw
time, so the first entry is both the topmost band and the first legend key.

Today the list holds the three interaction channels with the vertex inside the
wire-readout sensitive volume, plus everything neutrino-induced from outside it:

    numu_CC_in_volume   numu charged-current -- the signal this analysis selects
                        for, and much the largest of the three.
    nue_CC_in_volume    nue charged-current -- a background: a CC interaction of
                        the wrong flavour, which deposits the same kind of energy
                        as the signal and so cannot be separated from it by the
                        variable on this axis alone.
    NC_in_volume        neutral current, any flavour -- a background with no
                        charged lepton, so it deposits less of the incident
                        energy and its band sits lower in energy than the CC ones.
    neutrino_out_of_volume
                        any channel, vertex OUTSIDE the volume -- energy that
                        leaked in from an interaction the analysis does not
                        accept. Deliberately not split by channel: see
                        select_neutrino_out_of_volume.

The four together are every true neutrino cluster whose vertex is known, since
numu_CC / nue_CC / NC partition the channels (metadata.classify_neutrino_interaction
gives exactly one to each interaction) and in/out partitions the volume. A
neutrino with NO mc.json vertex has vertex_in_volume=None and is claimed by
nothing -- unknown rather than assigned to a side.

Still outside the stack: the cosmic clusters, which are most of the sample.
summary.txt reports how many true clusters no component claims, so the size of
what is missing stays visible while the stack is incomplete.

THE RECO OVERLAY IS OPTIONAL AND CURRENTLY UNUSED

draw_stacked_true_energy still accepts reco_records and will draw them as a step
curve over the stack, but the notebook no longer passes any: a stacked histogram
of true clusters and a count of reco clusters are different populations, and
putting them on one axes invites reading a comparison that the two do not
support (one true cluster split into two reco clusters counts once in the stack
and twice in the curve). The machinery, and reco_cluster_energy_mev with it, is
kept because a reco-side figure of its own would use it unchanged.

WHY A STACK RATHER THAN OVERLAID CURVES

The question these plots answer is how much of the sample in a given energy bin
is signal, so the components must add up: the total height of a bin is the
number of clusters in it and each band is that category's share. Overlaid curves
answer a different question (the shape of each category on its own) and cannot
be read for a composition.

WHAT FILLS IT

The x variable is the TRUE DEPOSITED energy of the true cluster -- the sum of
sed-smear's per-point 'e' field over the cluster, which is what
draw_variables.build_true_cluster_variable_records stores as 'total_energy'. It
is energy that was actually deposited in the active volume and survived the
selection, not the incident neutrino energy: a cluster whose interaction put
half its energy into a neutron that left the detector appears here at what the
argon saw. The y variable is a number of CLUSTERS, not of interactions -- with
neutrino clusters reassigned to 99990+nu_idx there is exactly one cluster per
interaction that deposited anything, so the two coincide for the components here
and would stop coinciding for a cosmic component.

Bin width is ENERGY_BIN_WIDTH_MEV below (200 MeV), on an axis fixed to
[0, ENERGY_AXIS_MAX_MEV] with ticks every ENERGY_AXIS_TICK_MEV.

CATEGORY LABELS come from metadata.build_neutrino_vertex_records by way of
attach_interaction_channel() below: 'interaction_channel' (numu_CC / nue_CC / NC,
see metadata.classify_neutrino_interaction) and 'vertex_in_volume' (the same
volume bounds as the fiducial cut). Both are joined onto the cluster records by
exact key, never spatially.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from pathlib import Path

from metadata import build_neutrino_channel_map
from draw_variables import ENERGY_BIN_WIDTH_MEV as _DRAW_VARIABLES_ENERGY_BIN_WIDTH_MEV

# Bin width of THIS histogram, in MeV. Wider than the 100 MeV draw_variables.py
# uses for its true-energy plot: a stack is read by comparing band thicknesses
# within a bin, and at these statistics narrower bins hold one or two clusters.
#
# It must stay a WHOLE MULTIPLE of that 100 MeV -- the check below enforces it --
# so every bin here is an exact union of consecutive draw_variables bins and the
# two plots' edges line up rather than straddling each other.
ENERGY_BIN_WIDTH_MEV = 200.0

if ENERGY_BIN_WIDTH_MEV % _DRAW_VARIABLES_ENERGY_BIN_WIDTH_MEV != 0:
    raise ValueError(
        f"ENERGY_BIN_WIDTH_MEV ({ENERGY_BIN_WIDTH_MEV}) must be a whole multiple of "
        f"draw_variables.ENERGY_BIN_WIDTH_MEV ({_DRAW_VARIABLES_ENERGY_BIN_WIDTH_MEV}) "
        f"for the two energy histograms' bin edges to line up")

# ============================================================================
# RECO CHARGE -> ENERGY
# ============================================================================
# The reco overlay needs its clusters on the same MeV axis as the true ones, so
# each cluster's collected charge is turned into a deposited energy by the
# first-principles argon numbers rather than by a fitted calibration:
#
#     E_deposited = W_ion * Q / R
#
# W_ion = 23.6 eV is the energy it takes to make one electron-ion pair in liquid
# argon, and R = 0.7 is the fraction of those electrons that escape recombination
# and drift to the wires (a MIP at SBND's 500 V/cm). Dividing by R puts back the
# charge that recombined and was never collected. Q is the cluster's summed
# charge, which this assumes is IN ELECTRONS -- see the caveats below.
#
# CAVEATS, because this is an approximation and its failures are all one-sided:
#   - R is treated as a single number. It really depends on dE/dx, so a dense
#     stopping track recombines more than a MIP and this UNDER-estimates its
#     energy.
#   - It assumes every escaping electron is collected and recorded: no electron
#     lifetime attenuation, no dead channels, no charge lost outside the cluster
#     by the reconstruction. Each of those makes the estimate too LOW.
#   - It assumes the charge unit is one electron. If it is not, everything here
#     is off by that scale factor.
# Cross-check on the last two: this gives 1/(W/R) = 29662 charge per MeV, against
# 22000-26000 from fitting true energy vs reco charge over the low-charge region
# (EnergyReconstruction/fit_energy_calibration.py). The fit is 12-26% lower,
# i.e. the real detector collects rather less than this formula assumes -- so
# expect the overlay to sit somewhat HIGH in energy against a matched true
# cluster. The two agreeing to that level is also what says the charge unit is
# about one electron.
RECO_WORK_FUNCTION_EV     = 23.6    # W_ion, eV per electron-ion pair in LAr
RECO_RECOMBINATION_FACTOR = 0.7     # R, fraction of ionisation electrons that escape
RECO_MEV_PER_CHARGE = (RECO_WORK_FUNCTION_EV * 1e-6) / RECO_RECOMBINATION_FACTOR


def reco_cluster_energy_mev(total_charge):
    """Deposited energy in MeV from a reco cluster's summed charge: W * Q / R."""
    return float(total_charge) * RECO_MEV_PER_CHARGE


# Names the RECO selection the overlay was built from -- the true side's cuts are
# fixed, so this labels the only thing that varies between versions of the plot.
# 'NoCuts' is every reco cluster in the event: no beam-window cut, no fiducial
# cut, no minimum point count.
DEFAULT_RECO_CUTS_LABEL = 'NoCuts'

# The y scales to draw each figure on -- see draw_stacked_true_energy's y_scale.
# LINEAR only: log earned its place when a reco overlay towering over the stack
# shared the axes, and with the overlay gone the stack spans a readable range on
# its own. Add 'log' back to this tuple to get both again; the filename carries
# _logy / _liny either way, so the two never collide.
Y_SCALES = ('linear',)

# Every figure is also drawn at each of these bin widths, in MeV. Wider bins put
# enough clusters in a bin for the stack's band thicknesses to be comparable;
# narrower ones resolve structure the wide bins average over. Both are cheap --
# the expensive part of a run is reading and pairing the events, not drawing --
# so the choice need not be made in advance. The width is in every filename.
BIN_WIDTHS_MEV = (200.0, 100.0)

# The overlay: a black step line, unfilled, so it reads as a different KIND of
# thing from the filled bands underneath and never hides them.
_RECO_OVERLAY_STYLE = dict(color='black', linestyle='-', linewidth=2.0)


# Font sizes and headroom, matching draw_variables.py so the two modules'
# figures sit side by side without looking like they came from different tools.
_AXIS_LABEL_FONTSIZE = 15
_TITLE_FONTSIZE      = 16
_TICK_LABEL_FONTSIZE = 13
# Smaller than the axis labels: TWO legends share the top of the axes, side by
# side, and at 13 the stack's five-entry box and the overlay's box together were
# wider than the axes and overlapped each other.
_LEGEND_FONTSIZE     = 11

# y is LOG. The reco overlay runs to ~1000 clusters in its first bin while the
# whole true stack is ~70, so on a linear axis scaled to the overlay the stack is
# a few pixels tall and the composition the plot exists to show is unreadable. A
# log axis puts both populations on one picture at the cost of bin heights no
# longer being comparable by area.
#
# _Y_BOTTOM is half a cluster, so a bin holding exactly one is a visible step
# above the axis rather than sitting on it (log has no zero to rest on, and
# anything at 0 simply is not drawn).
#
# _Y_HEADROOM_FACTOR multiplies the tallest bin -- a FACTOR, not a fraction,
# because the axis is log. Both legends sit INSIDE the axes, so this is what
# keeps them off the data, and on a log axis it takes more than a decade of room
# to do that. Raise it if a legend ever lands on a histogram: the cost is empty
# space at the top, which is cheap, and the alternative is a covered plot.
_Y_BOTTOM            = 0.5
_Y_HEADROOM_FACTOR   = 25.0

# Headroom for the LINEAR version, as a fraction of the tallest bin rather than a
# factor -- on a linear axis a factor of 25 would squash every histogram into the
# bottom 4% of the frame. The legends still sit inside the axes on both versions.
_Y_HEADROOM_LINEAR   = 0.45


# ============================================================================
# CATEGORY LABELS ON THE CLUSTER RECORDS
# ============================================================================

def attach_interaction_channel(true_records, vertex_records):
    """
    Add 'interaction_channel' to each true NEUTRINO cluster record, in place, and
    return the same list.

    build_true_cluster_variable_records already carries 'vertex_in_volume'; the
    interaction channel is the other half of what the stack splits on and is not
    on those records, so it is joined here from the event's vertex records.

    The join key is (event, true_cluster_id) via metadata.build_neutrino_channel_map,
    which keys on (event, cluster_id) -- a neutrino true cluster's id IS
    99990+nu_idx and the vertex record carries the same id, so this is an exact
    lookup with no spatial matching and no tolerance.

    Clusters that are not neutrinos, and neutrinos whose interaction has no
    channel (no mc.json record, or a vertex record that could not be classified),
    get interaction_channel=None and are then claimed by no component -- left out
    rather than guessed into one.
    """
    channel_map = build_neutrino_channel_map(vertex_records or [])
    for record in true_records or []:
        channel = None
        if record.get('is_neutrino'):
            channel = channel_map.get((record['event'], record['true_cluster_id']))
        record['interaction_channel'] = channel
    return true_records


def select_channel_in_volume(channel):
    """
    Build the selector for one interaction channel with the vertex in volume:
    true neutrino clusters whose interaction_channel is `channel` and whose
    vertex_in_volume is True.

    A factory rather than one hand-written function per channel: the channels
    differ by a single string, and copies of the same three-line filter drift
    apart -- one gets a fix the others do not. Selectors built here are mutually
    exclusive by construction, because a cluster carries exactly one channel,
    which is the property a stack depends on.

    vertex_in_volume must be exactly True -- a record where it is None (built
    without vertex_records, or an interaction with no vertex in mc.json) is
    dropped rather than assumed to be inside, the same rule
    draw_variables.select_true_neutrino_records applies.
    """
    def selector(true_records):
        return [r for r in (true_records or [])
                if r.get('is_neutrino')
                and r.get('interaction_channel') == channel
                and r.get('vertex_in_volume') is True]
    selector.__name__ = f"select_{channel}_in_volume"
    selector.__doc__  = f"True neutrino clusters: {channel} interaction, vertex in volume."
    return selector


select_numu_cc_in_volume = select_channel_in_volume('numu_CC')
select_nue_cc_in_volume  = select_channel_in_volume('nue_CC')
select_nc_in_volume      = select_channel_in_volume('NC')


# Both completeness and purity must EXCEED this for a matched signal cluster to
# count as well reconstructed. 0.8 = 80%.
#
# COMPLETENESS here is the energy-weighted fraction of a true cluster that its
# matched reco cluster picked up -- the quantity this repo renamed from
# "efficiency" in 1c6d3a6, and the name the legend uses too, so the plot and the
# rest of the codebase say the same word for the same thing.
NUMU_QUALITY_THRESHOLD = 0.8


def attach_pair_metrics(true_records, pair_metadata_list):
    """
    Add 'pair_completeness' and 'pair_purity' to each true cluster record, in
    place, from its 1-to-1 matched reco pair. Returns the same list.

    Joined by (event, true_cluster_id) against
    metadata.add_metadata_true_reco_pair_cluster output -- an exact key, as with
    the channel join.

    A true cluster that won no pair gets None for both. That is NOT the same as
    zero and must not be read as one: it means the cluster was never matched, so
    there is no reco cluster to have been complete or pure with respect to. The
    quality selectors below require values strictly above a threshold, so None
    fails them without any special case.
    """
    by_true = {(m['event'], m['true_cluster_id']): m for m in (pair_metadata_list or [])}
    for record in true_records or []:
        pair = by_true.get((record['event'], record['true_cluster_id']))
        record['pair_completeness'] = pair.get('completeness') if pair else None
        record['pair_purity']       = pair.get('purity')       if pair else None
    return true_records


def _is_well_reconstructed(record, threshold=NUMU_QUALITY_THRESHOLD):
    """Both completeness and purity above threshold. Unmatched (None) is False."""
    completeness = record.get('pair_completeness')
    purity       = record.get('pair_purity')
    return (completeness is not None and purity is not None
            and completeness > threshold and purity > threshold)


def select_numu_cc_in_volume_high_completeness(true_records):
    """Signal clusters whose match has BOTH completeness and purity above 80%."""
    return [r for r in select_numu_cc_in_volume(true_records) if _is_well_reconstructed(r)]


def select_numu_cc_in_volume_contaminated(true_records):
    """
    Every OTHER signal cluster: the complement of the above within numu CC
    in-volume, so the two together are exactly select_numu_cc_in_volume().

    Defined as a complement rather than as its own threshold test on purpose. A
    cluster fails the quality cut for several different reasons -- low
    completeness, low purity, both, or never having been matched at all -- and
    writing them out as a second condition would eventually miss one and drop
    clusters silently out of the stack. Complement cannot.
    """
    return [r for r in select_numu_cc_in_volume(true_records) if not _is_well_reconstructed(r)]


def select_neutrino_out_of_volume(true_records):
    """
    True neutrino clusters whose interaction vertex is OUTSIDE the volume, of any
    interaction channel.

    Not split by channel, unlike the in-volume components. Out-of-volume is a
    rejection category rather than a physics one: what is on the plot is energy
    that leaked into the active volume from an interaction that happened
    somewhere this analysis does not accept, and whether the parent was CC or NC
    does not change how it has to be removed. Splitting it three ways would add
    two more bands of one or two clusters each without adding a distinction
    anyone acts on. (Split it later by swapping this for three
    select_channel_out_of_volume selectors if that changes.)

    vertex_in_volume must be exactly False. A neutrino whose flag is None -- no
    mc.json vertex to test -- is NOT out of volume, it is unknown, and is left
    for no component to claim rather than being swept in here; summary.txt counts
    what nothing claims.
    """
    return [r for r in (true_records or [])
            if r.get('is_neutrino') and r.get('vertex_in_volume') is False]


# The components. 'select' takes the true cluster records of whatever level is
# being drawn and returns the ones belonging to that component.
#
# THIS LIST IS DECLARED TOP-DOWN: first entry is the topmost band of the stack
# and the first key in the legend. order_components_for_stack() reverses it to
# get the drawing order (matplotlib stacks bottom-first); the legend uses it as
# written. Changing the order of the plot means changing the order here.
#
# Components must be MUTUALLY EXCLUSIVE: a stack adds its bands, so a cluster
# claimed by two of them would be counted twice in the total. check_components()
# below enforces that rather than trusting it.
SIGNAL_BACKGROUND_COMPONENTS = [
    {
        'key':    'numu_CC_in_volume',
        'label':  r'$\nu_\mu$ CC, vertex in volume',
        'color':  'tab:blue',
        'select': select_numu_cc_in_volume,
    },
    {
        'key':    'nue_CC_in_volume',
        'label':  r'$\nu_e$ CC, vertex in volume',
        'color':  'tab:orange',
        'select': select_nue_cc_in_volume,
    },
    {
        'key':    'NC_in_volume',
        'label':  'NC, vertex in volume',
        'color':  'tab:green',
        'select': select_nc_in_volume,
    },
    {
        'key':    'neutrino_out_of_volume',
        'label':  r'$\nu$ (any channel), vertex out of volume',
        'color':  'tab:purple',
        'select': select_neutrino_out_of_volume,
    },
]


def order_components_for_stack(components, selected_by_key=None):
    """
    The drawing order: the declared list REVERSED, because matplotlib stacks
    bottom-first while the component lists are written top-down.

    Fixed by declaration rather than by size. The bands are a physics ordering --
    the signal channel, then the other in-volume channels, then what came from
    outside the volume -- and that ordering means the same thing in every run and
    at every level, so the stacks stay comparable to one another. A size-ordered
    stack reshuffles itself whenever a component grows past its neighbour, which
    is exactly when two plots most need to be read side by side.

    `selected_by_key` is unused and accepted only so the call sites can pass it
    the same way they pass it to the legend ordering.
    """
    return list(components)[::-1]


# A SECOND stack: the same four categories, but with the signal split by how
# well it was reconstructed. Everything else is identical and shared by
# reference, so a change to a background component reaches both stacks.
#
# The split is exact -- high_completeness and contaminated are complements within
# numu CC in-volume (see select_numu_cc_in_volume_contaminated) -- so this stack
# holds precisely the clusters SIGNAL_BACKGROUND_COMPONENTS does, only bound into
# five bands instead of four. Their totals must agree, which makes one a check on
# the other.
_NON_SIGNAL_COMPONENTS = [c for c in SIGNAL_BACKGROUND_COMPONENTS
                          if c['key'] != 'numu_CC_in_volume']

SIGNAL_BACKGROUND_COMPONENTS_NUMU_QUALITY = [
    {
        'key':    'numu_CC_in_volume_high_completeness',
        'label':  r'$\nu_\mu$ CC in-volume, high completeness',
        'color':  'tab:blue',
        'select': select_numu_cc_in_volume_high_completeness,
    },
    {
        'key':    'numu_CC_in_volume_contaminated',
        'label':  r'$\nu_\mu$ CC in-volume, contaminated',
        'color':  'tab:cyan',
        'select': select_numu_cc_in_volume_contaminated,
    },
] + _NON_SIGNAL_COMPONENTS
# Also top-down: the two halves of the signal band, then the backgrounds in the
# order they are declared above. _NON_SIGNAL_COMPONENTS is a filter of that list
# and so keeps it.


def order_components_for_legend(components, selected_by_key=None):
    """
    LEGEND order from DRAWING order: the reverse of what it is given.

    Callers hold the list order_components_for_stack returned -- bottom band
    first -- because that is the order the bands were drawn and the handles
    collected in. Reversing it here puts the topmost band first, so the legend
    read down the box names the bands read down the picture, and both come back
    to the single top-down declaration.

    `selected_by_key` is unused -- kept in the signature because the call sites
    predate the fixed ordering and there is nothing to gain from touching them.
    """
    return list(components)[::-1]


def check_components(components, selected_by_key):
    """
    Every cluster is in at most one component. Returns the list of
    (component key, component key, number of shared clusters) overlaps found,
    empty when the stack is well formed.

    Called by the drawer on every figure: an overlap does not raise, because a
    half-drawn stack is more use than none while a new component is being
    written, but it is reported so a total that no longer equals the sum of its
    parts cannot pass unnoticed.
    """
    def _ids(records):
        return {(r['event'], r['true_cluster_id']) for r in records}

    overlaps = []
    keys = [c['key'] for c in components]
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            shared = _ids(selected_by_key[left]) & _ids(selected_by_key[right])
            if shared:
                overlaps.append((left, right, len(shared)))
    return overlaps


# ============================================================================
# BINNING
# ============================================================================

# The energy axis is FIXED at [0, ENERGY_AXIS_MAX_MEV], not fitted to the data.
# A stack is read by comparing bins between components and between runs, and an
# axis that follows whichever sample happens to be loudest moves the bins under
# that comparison: the same signal cluster lands in a different-looking plot when
# a background with a longer tail is added, or when a run covers more files. A
# fixed axis costs empty space at the top and buys plots that can be laid side by
# side.
ENERGY_AXIS_MAX_MEV = 5000.0

# Spacing of the labelled x ticks, in MeV. Independent of the 100 MeV bin width:
# a tick per bin would be 50 unreadable labels, so the bins are the resolution
# and the ticks are the scale you read positions against.
ENERGY_AXIS_TICK_MEV = 500.0


# Where the first bin ends. None means "follow whatever bin width is in use",
# giving a UNIFORM grid from zero -- 0, 200, 400, ... at 200 MeV, 0, 100, 200, ...
# at 100 MeV.
#
# It must NOT be pinned to one number now that energy_bin_edges is called at more
# than one width: a fixed 200 here silently produced a 200 MeV first bin followed
# by 100 MeV bins when 100 was asked for.
#
# Set it to a number SMALLER than the width to split the first bin deliberately.
# That is worth doing when the reco overlay is uncut: the true side has a 100 MeV
# energy cut so nothing true can fall below 100 MeV, while 44% of the NoCuts reco
# clusters do, and a first edge at 100 keeps that sub-threshold pile-up in a bin
# of its own rather than merging it into the first bin the two populations
# actually share.
FIRST_BIN_EDGE_MEV = None


def set_fitted_title(ax, text, fontsize, max_lines=2, min_fontsize=9, **kwargs):
    """
    Set a title guaranteed not to run wider than the axes it belongs to.

    Titles here are built by concatenation -- channel, then reco selection, then
    threshold -- so their length depends on what is being drawn and a size that
    fits one figure overflows the next. Rather than pick a fontsize that fits the
    worst case everywhere, this measures the rendered title against the axes and
    fixes it only when it is actually too wide: first by wrapping onto up to
    max_lines lines at word boundaries, then, if that is still not enough, by
    shrinking down to min_fontsize.

    Measuring needs a renderer, so this draws the canvas once. That costs a
    fraction of a second per figure and is why it is not applied to axis labels,
    which are short and fixed.
    """
    title = ax.set_title(text, fontsize=fontsize, **kwargs)
    figure = ax.get_figure()
    renderer = figure.canvas.get_renderer()

    def overflow():
        """How much wider than the axes the title is, in pixels."""
        return (title.get_window_extent(renderer=renderer).width
                - ax.get_window_extent(renderer=renderer).width)

    if overflow() <= 0:
        return title

    # Wrapping first: two lines at the same size stay more readable than one line
    # at two thirds the size.
    words = text.split()
    for n_lines in range(2, max_lines + 1):
        per_line = -(-len(words) // n_lines)
        wrapped = "\n".join(" ".join(words[i:i + per_line])
                             for i in range(0, len(words), per_line))
        title.set_text(wrapped)
        if overflow() <= 0:
            return title

    # Still too wide: shrink by the measured ratio, which lands in one step, and
    # iterate only to absorb the rounding.
    for _ in range(4):
        excess = overflow()
        if excess <= 0:
            break
        current = title.get_fontsize()
        width = title.get_window_extent(renderer=renderer).width
        title.set_fontsize(max(min_fontsize, current * width_ratio(width, excess)))
        if title.get_fontsize() <= min_fontsize:
            break
    return title


def width_ratio(width, excess):
    """Shrink factor taking a title of `width` px down to `width - excess` px."""
    return max(0.5, (width - excess) / width) if width > 0 else 1.0


def energy_bin_edges(bin_width=ENERGY_BIN_WIDTH_MEV, axis_max=ENERGY_AXIS_MAX_MEV,
                     first_edge=None):
    """
    Fixed edges: [0, first_edge], then bin_width-wide bins to axis_max --
    [0, 200, 400, ... 4800, 5000] with the defaults, i.e. a uniform grid from
    zero, since first_edge equals bin_width there. The same edges for every
    component, every level and every run.

    At a width that does not divide axis_max - first_edge the LAST bin comes out
    narrow, because axis_max closes the axis wherever the run of full-width bins
    happens to stop. Keeping the axis exactly [0, axis_max] is worth more than a
    uniform final bin -- and since y is a raw COUNT rather than a density, an
    odd-width bin misreads only as a narrower box, not as a wrong height.

    Every edge stays on the 100 MeV grid draw_variables.py bins its true-energy
    plot with (100 divides both first_edge and bin_width), so the two plots'
    edges still line up.

    Takes no data: see ENERGY_AXIS_MAX_MEV above for why the range is fixed
    rather than fitted. Callers that need to know whether anything fell off the
    top should use count_overflow().
    """
    if first_edge is None:
        first_edge = FIRST_BIN_EDGE_MEV if FIRST_BIN_EDGE_MEV is not None else bin_width
    edges = [0.0]
    edge = first_edge
    while edge < axis_max:
        edges.append(edge)
        edge += bin_width
    edges.append(axis_max)
    return np.array(edges, dtype=float)


def count_overflow(values, axis_max=ENERGY_AXIS_MAX_MEV):
    """
    How many values sit above the fixed axis and are therefore absent from the
    histogram.

    np.histogram drops them silently, which on a fixed axis is a real way to lose
    entries -- so the drawer calls this and says so rather than letting a plot
    quietly disagree with its own cluster count.
    """
    return sum(1 for v in values if v is not None and v > axis_max)


# ============================================================================
# DRAWING
# ============================================================================

def shared_y_top(true_records, reco_record_lists, components=None,
                 bin_width=ENERGY_BIN_WIDTH_MEV, y_scale='log'):
    """
    One y-axis top for a set of figures that should be read against each other:
    the tallest bin over the stack AND over every reco overlay in
    reco_record_lists, plus headroom for the legends.

    The headroom differs by scale (see _Y_HEADROOM_FACTOR / _Y_HEADROOM_LINEAR),
    so call this once per y_scale and hand each result to the figures drawn on
    that scale.

    Pass the result to every draw_stacked_true_energy call in the set. Without
    it each figure scales to its own tallest bin, and two plots whose only
    difference is the reco selection end up with axes an order of magnitude
    apart -- which makes the cut look like it changed the true stack, when the
    stack is identical in both and only the overlay moved.
    """
    components = components if components is not None else SIGNAL_BACKGROUND_COMPONENTS
    edges = energy_bin_edges(bin_width)

    selected_by_key = {c['key']: c['select'](true_records) for c in components}
    stack_totals = np.sum([np.histogram([r['total_energy'] for r in selected_by_key[c['key']]],
                                        bins=edges)[0]
                           for c in components], axis=0)
    tallest = int(stack_totals.max()) if len(np.atleast_1d(stack_totals)) else 0

    for records in reco_record_lists or []:
        values = [reco_cluster_energy_mev(r['total_charge']) for r in records or []]
        if values:
            tallest = max(tallest, int(np.histogram(values, bins=edges)[0].max()))

    tallest = max(tallest, 1)
    return tallest * _Y_HEADROOM_FACTOR if y_scale == 'log' else tallest * (1 + _Y_HEADROOM_LINEAR)


def draw_stacked_true_energy(true_records, output_dir, level_name, filename_prefix, apa,
                             file_name=None, components=None,
                             bin_width=ENERGY_BIN_WIDTH_MEV,
                             reco_records=None, reco_cuts_label=DEFAULT_RECO_CUTS_LABEL,
                             y_top=None, variant_label=None, y_scale='log',
                             filename='signal_background_true_energy_stack',
                             title=None):
    """
    The stacked true-energy histogram: x = true deposited cluster energy in
    bin_width MeV bins, y = number of clusters, one stack band per component.

    Parameters:
    - true_records: true cluster variable records for this level, already through
        attach_interaction_channel() -- build_true_cluster_variable_records output
        plus 'interaction_channel'
    - output_dir: written here (created if missing)
    - level_name: goes in the title ("Job Level", "Event 5", ...)
    - filename_prefix: goes in the filename ("job", "file", "event_5")
    - apa: identification, as elsewhere; "Combined" for this pipeline
    - file_name: appended to the title when given
    - components: defaults to SIGNAL_BACKGROUND_COMPONENTS
    - reco_records: build_reco_cluster_variable_records output. Drawn as ONE
        unfilled step curve OVER the stack, not as a band in it -- see below.
        None draws the stack alone.
    - reco_cuts_label: which reco selection those records came from ('NoCuts',
        ...). Goes in the title, the legend and the filename, so two selections
        can be told apart at a glance and never overwrite each other's output.
    - y_top: top of the y axis. None scales to this figure's own tallest bin;
        pass shared_y_top() to put a set of figures on ONE scale so they can be
        compared by eye.
    - variant_label: names the STACK, where reco_cuts_label names the overlay.
        Needed whenever two figures share a reco selection and differ only in how
        the stack is broken up -- without it they would write to the same file.
    - y_scale: 'log' or 'linear'. Goes in the filename as _logy / _liny, so both
        versions of one figure coexist. Log is the readable one when the reco
        overlay is uncut (it runs 20x above the stack); linear is the one where
        bin heights can be compared by area.
    - title: the base plot title. None gives "Signal & Background"; the reco
        selection and variant_label are still appended to whatever is passed.

    WHY THE RECO CURVE IS AN OVERLAY AND NOT A BAND. The stack partitions ONE
    population -- true clusters -- so its bands add to a meaningful total. A reco
    cluster is a different object: it is not a category of true cluster, it has
    no truth label, and a true cluster with two reco clusters contributes once to
    the stack and twice to the curve. Stacking it would produce a total that
    counts some interactions twice and means nothing. Overlaid, the comparison it
    invites is the right one: how the reconstruction's population compares with
    the truth's, bin by bin.

    Returns (selected_by_key, reco_values): the {component key: selected records}
    mapping and the reco energies in MeV, so the caller can write the same
    numbers to a table without re-selecting or re-converting.

    The x axis is FIXED at [0, ENERGY_AXIS_MAX_MEV] rather than fitted to the
    sample -- see that constant. Anything above it is reported rather than
    silently dropped.
    """
    components = components if components is not None else SIGNAL_BACKGROUND_COMPONENTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_by_key = {c['key']: c['select'](true_records) for c in components}
    # Bottom band first -- see order_components_for_stack. Everything downstream
    # (the hist call, the legend, the returned mapping's use by the table)
    # follows this order, so the figure and the table agree.
    components = order_components_for_stack(components, selected_by_key)

    overlaps = check_components(components, selected_by_key)
    for left, right, n_shared in overlaps:
        print(f"    WARNING: stack components '{left}' and '{right}' share {n_shared} "
              f"cluster(s) -- the stack total double counts them")

    values_by_key = {key: [r['total_energy'] for r in records]
                     for key, records in selected_by_key.items()}
    all_values = [v for values in values_by_key.values() for v in values]
    edges = energy_bin_edges(bin_width)

    n_overflow = count_overflow(all_values)
    if n_overflow:
        print(f"    NOTE: {n_overflow} cluster(s) above {ENERGY_AXIS_MAX_MEV:.0f} MeV "
              f"are off the fixed axis and not drawn")

    fig, ax = plt.subplots(figsize=(10, 7))
    # histtype='stepfilled', NOT the default 'bar'. Both stack, but 'bar' draws
    # each bin as its own rectangle with its own outline, which reads as a bar
    # chart of independent categories -- and energy is a continuous variable
    # arbitrarily divided into bins, not a set of categories. 'stepfilled' draws
    # the stack as one filled silhouette with a single step outline, the usual
    # look for a binned distribution, and the fill still shows the composition
    # the stack exists for.
    reco_values = [reco_cluster_energy_mev(r['total_charge']) for r in (reco_records or [])]
    n_reco_overflow = count_overflow(reco_values)
    if n_reco_overflow:
        print(f"    NOTE: {n_reco_overflow} reco cluster(s) above {ENERGY_AXIS_MAX_MEV:.0f} MeV "
              f"are off the fixed axis and not drawn")

    label_by_key = {c['key']: f"{c['label']} ({len(values_by_key[c['key']])})" for c in components}
    labels = [label_by_key[c['key']] for c in components]
    _, _, patch_groups = ax.hist(
        [values_by_key[c['key']] for c in components],
        bins=edges, stacked=True, histtype='stepfilled',
        color=[c['color'] for c in components],
        label=labels,
        edgecolor='black', linewidth=1.2,
    )

    reco_handle = None
    if reco_values:
        # step, not stepfilled: an outline over the filled stack. Drawn after the
        # stack so it is never painted over by a band.
        counts, _ = np.histogram(reco_values, bins=edges)
        reco_handle = ax.step(edges, np.append(counts, counts[-1]), where='post',
                              **_RECO_OVERLAY_STYLE)[0]

    # Short enough to fit the axes width on one line. The level, the APA and the
    # input file are deliberately NOT in it: they are constant across everything
    # this notebook draws in a run, and they are already in the output path and in
    # summary.txt.
    #
    # The reco selection is named ONLY when a reco overlay is drawn. Without one
    # nothing on the figure depends on it -- the stack is pure truth -- and naming
    # it would claim a dependence the plot does not have.
    title = title or "Signal & Background"
    if reco_values:
        title += f" -- reco: {reco_cuts_label}"
    if variant_label:
        title += f" -- {variant_label}"
    ax.set_xlabel('Deposited Cluster Energy (MeV)',
                  fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Number of Clusters', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    set_fitted_title(ax, title, _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(edges[0], edges[-1])
    ax.xaxis.set_major_locator(MultipleLocator(ENERGY_AXIS_TICK_MEV))

    # LOG y -- see _Y_BOTTOM. The limits are set from the bin contents rather
    # than left to autoscale, which on a log axis picks a bottom from the
    # smallest non-zero bin and so moves between runs.
    ax.set_yscale(y_scale)
    if y_top is None:
        stack_totals = np.sum([np.histogram(values_by_key[c['key']], bins=edges)[0]
                               for c in components], axis=0)
        tallest = max(int(stack_totals.max()) if len(np.atleast_1d(stack_totals)) else 0,
                      int(np.histogram(reco_values, bins=edges)[0].max()) if reco_values else 0,
                      1)
        y_top = (tallest * _Y_HEADROOM_FACTOR if y_scale == 'log'
                 else tallest * (1 + _Y_HEADROOM_LINEAR))
    # Linear starts at zero -- counts rest on the axis and an empty bin should
    # read as empty. Log cannot (no zero on a log axis), so it starts at half a
    # cluster; see _Y_BOTTOM.
    ax.set_ylim(_Y_BOTTOM if y_scale == 'log' else 0, y_top)
    if y_scale != 'log':
        # Whole clusters only. On log the decade ticks are already right.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # No stats box: the per-component counts are in the legends and everything a
    # box would have held (mean, median, range, per-bin contents) is in
    # signal_background_info.txt, to more digits than a corner of a figure can
    # carry.

    # Legend order comes from order_components_for_legend: the drawing order
    # reversed, so the topmost band is the first key. Keyed off the component
    # list rather than left to matplotlib, which registers stacked artists in its
    # own order and would follow neither the stack nor the declaration.
    #
    # The handles are the drawn polygons rather than proxies, so a key cannot end
    # up showing a colour the band does not have.
    handle_by_key = {c['key']: (group[0] if len(group) else None)
                     for c, group in zip(components, patch_groups)}
    keyed = [(handle_by_key[c['key']], label_by_key[c['key']])
             for c in order_components_for_legend(components, selected_by_key)
             if handle_by_key.get(c['key']) is not None]
    if keyed:
        stack_legend = ax.legend([h for h, _ in keyed], [l for _, l in keyed],
                                 fontsize=_LEGEND_FONTSIZE, framealpha=0.9,
                                 loc='upper right',
                                 title='True clusters (stacked)', title_fontsize=_LEGEND_FONTSIZE)
        stack_legend._legend_box.align = 'left'
    else:
        stack_legend = None

    if reco_handle is not None:
        # A SECOND legend, upper left, opposite the stack's (which is upper right).
    # The overlay is a different kind of object
        # from the stacked bands -- reco clusters, not true ones, not summing to
        # the stack's total -- and a separate box says so more clearly than one
        # more row in the same list would. ax.legend() replaces any existing
        # legend, so the first has to be re-added as an artist before the second
        # is made, or it silently disappears.
        if stack_legend is not None:
            ax.add_artist(stack_legend)
        # The entry is just the count: the selection name is already in the plot
        # title AND in this legend's own title, and repeating it a third time is
        # what made this box wide enough to collide with the stack's legend.
        reco_legend = ax.legend([reco_handle], [f"{len(reco_values)} clusters"],
                                fontsize=_LEGEND_FONTSIZE, framealpha=0.9,
                                loc='upper left',
                                title=f'Reco clusters ({reco_cuts_label})',
                                title_fontsize=_LEGEND_FONTSIZE)
        reco_legend._legend_box.align = 'left'

    variant_suffix = f"_{variant_label}" if variant_label else ""
    reco_suffix    = f"_{reco_cuts_label}" if reco_values else ""
    scale_suffix   = "_logy" if y_scale == 'log' else "_liny"
    path = output_dir / (f"{filename}{reco_suffix}{variant_suffix}"
                         f"_{bin_width:.0f}MeV{scale_suffix}_{filename_prefix}_{apa}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return selected_by_key, reco_values


# ============================================================================
# THE HISTOGRAMS THEMSELVES, IN ROOT
# ============================================================================

def write_signal_background_root(entries, output_dir, bin_width=ENERGY_BIN_WIDTH_MEV,
                                 filename='signal_background_histograms.root'):
    """
    Every histogram on every figure, as browsable TH1Ds in ONE .root file, so the
    plots can be restyled -- colours, order, log/linear, fits, rebinning -- without
    re-running the job that produced them. On this dataset that run is ~18 minutes;
    reading these back is instant.

    Layout: one TDirectory per figure, named <reco selection>[_<variant>], holding
    one TH1D per stack component (named by component key) plus 'reco_overlay'.
    Binning is identical to the figures', so summing the component histograms
    reproduces the stack exactly.

    Parameters:
    - entries: list of dicts with keys 'dir_name', 'selected_by_key',
        'reco_values', 'components' and optionally 'bin_width' -- one per figure,
        as returned by draw_stacked_true_energy plus the labels it was called
        with. 'bin_width' MUST be given when entries were drawn at different
        widths, or their histograms would all be rebinned to one width here and
        would no longer match the figures they came from.
    - output_dir: written here (created if missing)

    Written with uproot, NOT PyROOT: the ROOT installations on this machine
    conflict and PyROOT does not import (same reason
    EnergyReconstruction/fit_energy_calibration.py reads with uproot). The file is
    a real ROOT file either way -- ROOT, PyROOT elsewhere, or uproot can all open
    it.

    NOTE: this writes histograms directly rather than through
    root_histogram_writer.save_th1, which returns early on an empty input. A
    component with no clusters in a given run (nue_CC_in_volume is 0 in some
    samples) must still appear, as an empty histogram, or the file's structure
    would silently depend on the data and a restyling macro would break on the
    run where a category happens to be empty.
    """
    import uproot

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    with uproot.recreate(path) as root_file:
        for entry in entries:
            edges      = energy_bin_edges(entry.get('bin_width') or bin_width)
            dir_name   = entry['dir_name']
            components = entry.get('components') or SIGNAL_BACKGROUND_COMPONENTS
            selected   = entry.get('selected_by_key') or {}
            for component in components:
                values = [r['total_energy'] for r in selected.get(component['key'], [])]
                counts, _ = np.histogram(values, bins=edges)
                root_file[f"{dir_name}/{component['key']}"] = (counts, edges)
            reco_counts, _ = np.histogram(entry.get('reco_values') or [], bins=edges)
            root_file[f"{dir_name}/reco_overlay"] = (reco_counts, edges)
    return path


# ============================================================================
# THE NUMBERS BEHIND THE PLOT
# ============================================================================

def write_signal_background_info(selected_by_key, output_dir, level_name,
                                 components=None, bin_width=ENERGY_BIN_WIDTH_MEV,
                                 reco_values=None, reco_cuts_label=DEFAULT_RECO_CUTS_LABEL,
                                 variant_label=None, filename=None):
    """
    Per-component counts and the per-bin contents of the stack, so a bin can be
    checked against the plot and against the cluster tables without reading it
    off a figure.

    Written from the SAME mapping and reco values the drawer returned, not from a
    fresh selection, so the table and the figure cannot disagree.

    The reco column is kept OUT of the 'total' column: the total is the stack's,
    and reco clusters are not part of the stack (see draw_stacked_true_energy).
    """
    components = components if components is not None else SIGNAL_BACKGROUND_COMPONENTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The reco selection is in the filename, as it is in the figure's: one output
    # directory holds one table per selection, and two selections drawn into the
    # same directory must not overwrite each other's numbers.
    variant_suffix = f"_{variant_label}" if variant_label else ""
    reco_suffix    = f"_{reco_cuts_label}" if reco_values else ""
    path = output_dir / (filename or
                         f"signal_background_info{reco_suffix}{variant_suffix}_{bin_width:.0f}MeV.txt")

    all_values = [r['total_energy'] for records in selected_by_key.values() for r in records]
    edges = energy_bin_edges(bin_width)
    # Same order as the figure, so a column here is the band at that height.
    components = order_components_for_stack(components, selected_by_key)

    lines = []
    lines.append("=" * 78)
    lines.append(f"SIGNAL & BACKGROUND -- TRUE CLUSTER ENERGY STACK ({level_name})"
                 + (f" -- {variant_label}" if variant_label else ""))
    lines.append("=" * 78)
    lines.append("x = true deposited energy of the true cluster (sum of sed-smear per-point 'e'),")
    binning = (f"{bin_width:.0f} MeV bins"
               if FIRST_BIN_EDGE_MEV in (None, bin_width)
               else f"binned [0, {FIRST_BIN_EDGE_MEV:.0f}] then {bin_width:.0f} MeV wide")
    lines.append(f"    {binning} over [0, {ENERGY_AXIS_MAX_MEV:.0f}];  y = number of clusters.")
    lines.append("")
    lines.append("Components (bottom of the stack first, i.e. smallest first):")
    for component in components:
        records = selected_by_key.get(component['key'], [])
        energies = [r['total_energy'] for r in records]
        lines.append(f"  {component['key']:<24s} {len(records):6d} clusters")
        if energies:
            lines.append(f"  {'':<24s}        mean {np.mean(energies):9.1f} MeV, "
                         f"median {np.median(energies):9.1f} MeV, "
                         f"range [{np.min(energies):.1f}, {np.max(energies):.1f}] MeV")
    lines.append(f"  {'TOTAL IN STACK':<24s} {len(all_values):6d} clusters")
    if reco_values:
        lines.append("")
        lines.append(f"Overlaid, NOT part of the stack -- reco clusters ({reco_cuts_label}), "
                     f"energy = {RECO_WORK_FUNCTION_EV} eV * charge / {RECO_RECOMBINATION_FACTOR}:")
        lines.append(f"  {'reco_' + reco_cuts_label:<24s} {len(reco_values):6d} clusters")
        lines.append(f"  {'':<24s}        mean {np.mean(reco_values):9.1f} MeV, "
                     f"median {np.median(reco_values):9.1f} MeV, "
                     f"range [{np.min(reco_values):.1f}, {np.max(reco_values):.1f}] MeV")
        n_reco_over = count_overflow(reco_values)
        if n_reco_over:
            lines.append(f"  {n_reco_over} reco cluster(s) above the fixed "
                         f"{ENERGY_AXIS_MAX_MEV:.0f} MeV axis are NOT in the table below")
    n_overflow = count_overflow(all_values)
    if n_overflow:
        lines.append(f"  {n_overflow} cluster(s) above the fixed {ENERGY_AXIS_MAX_MEV:.0f} MeV axis "
                     f"are NOT in the per-bin table below")
    lines.append("")

    lines.append("Per-bin contents:")
    header = (f"  {'energy bin [MeV]':<22s}"
              + "".join(f"{c['key']:>26s}" for c in components)
              + f"{'total':>10s}"
              + (f"{'reco (overlay)':>18s}" if reco_values else ""))
    lines.append(header)
    counts_by_key = {}
    for component in components:
        energies = [r['total_energy'] for r in selected_by_key.get(component['key'], [])]
        counts_by_key[component['key']], _ = np.histogram(energies, bins=edges)
    reco_counts, _ = np.histogram(reco_values or [], bins=edges)
    for i in range(len(edges) - 1):
        row = f"  {f'{edges[i]:.0f} - {edges[i+1]:.0f}':<22s}"
        total = 0
        for component in components:
            n = int(counts_by_key[component['key']][i])
            total += n
            row += f"{n:>26d}"
        row += f"{total:>10d}"
        if reco_values:
            row += f"{int(reco_counts[i]):>18d}"
        lines.append(row)
    lines.append("=" * 78)

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return path
