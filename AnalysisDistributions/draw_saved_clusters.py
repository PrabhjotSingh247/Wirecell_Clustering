"""
SAVED CLUSTER VIEWS -- driven by
AnalysisDistributions/SignalBackground_Distributions.ipynb.

XZ, YZ and XY pictures of individual reco-true pairs, sampled across the
completeness-purity plane so that every corner of it has a concrete example
attached. The scatter says a pair sits at 30% completeness and 90% purity; these
say what that actually looks like.

HOW PAIRS ARE CHOSEN

The completeness-purity square is divided into a 10x10 grid of 10% cells. For
each interaction channel, ONE pair is saved per cell -- 100 cells x 3 channels at
most, in practice far fewer since most cells are empty.

FIRST ENCOUNTERED, NOT UNIFORMLY RANDOM. The event loop sees each event once and
the point clouds are far too large to keep to the end of the job, so a cell is
filled by the first pair that lands in it and later candidates are skipped. Over
a job the events arrive in file order, which is not correlated with a pair's
completeness or purity, so the sample is arbitrary rather than biased -- but it
is not a uniform draw from the cell, and a cell holding 200 pairs shows the same
one every run. Drawing a genuinely random pair would need a second pass over the
selected events, which is a bigger change than the pictures are worth.

The two-neutrino events ARE a uniform random draw -- see
ClusterViewSampler.offer_two_neutrino.

DIRECTORY LAYOUT

    job_summary/
        Saved_Clusters/
            completeness_100_90_purity_100_90_event_chunk0_37/
                pair_numu_CC_chunk0_37.png
            completeness_100_90_purity_90_80_event_chunk0_12/
                pair_NC_chunk0_12.png
            ...
        selection_completeness_vs_purity/
            completeness_purity.txt
        two_neutrino_in_beam/
            two_neutrino_chunk0_21.png

Saved_Clusters holds the completeness-purity grid and nothing else: one directory
per (cell, event), with that event's pair view for every channel that landed in
the cell.  The two-neutrino set is not a cell of that grid, so it sits beside
it rather than inside it, directly under job_summary.  The index goes next to the
scatter plot it explains, and lists all of them with paths written relative to
itself. In practice that is ONE file per directory: a cell is
filled once per channel, the three channels are filled by different events, and
only the event's FIRST neutrino is ever drawn (FIRST_NEUTRINO_CLUSTER_ID), so two
channels can only share a directory if one event's first neutrino somehow served
both -- which cannot happen. The per-channel filename is what makes the picture
identifiable; the directory name is what makes the cell browsable.

WHAT LEFT. The cosmic candidate views (Draw_Selection_Cosmics.ipynb, via
draw_selection_cosmics.py) and the below-60%-completeness pairs
(DrawRecoTrueClusters_Below_60pc_Completeness.ipynb) used to be drawn here. Both
are per-cluster pictures rather than a sampled illustration of a distribution,
and both now live with the other per-cluster views under their own notebooks.

WHY A SEPARATE MODULE. draw_selection_performance.py counts clusters;
this draws them. They share no code and have opposite cost profiles -- one figure
here is three panels of a full point cloud -- so the switch that turns these off
(SAVE_CLUSTER_VIEWS in the notebook) can skip this import path entirely.
"""

import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# The three projections, as (x index, y index, x label, y label). Columns of the
# standard point arrays: 0=x, 1=y, 2=z.
_VIEWS = [
    (2, 0, 'Z (cm)', 'X (cm)', 'XZ'),
    (2, 1, 'Z (cm)', 'Y (cm)', 'YZ'),
    (0, 1, 'X (cm)', 'Y (cm)', 'XY'),
]

_GRID_CELLS = 10          # 10 x 10 cells of 10% each

# Only pairs whose true cluster is the FIRST neutrino of its event (nu_idx 1, so
# cluster id 99991) are drawn. In a multi-neutrino event a picture of the second
# neutrino shows one interaction while another sits in the same readout, and
# nothing in the frame says which points belong to which -- the view stops being
# a clean illustration of its completeness and purity. Restricting to the first
# keeps every saved picture unambiguous, at the cost of never illustrating a
# second-neutrino pair.
FIRST_NEUTRINO_CLUSTER_ID = 99991.0

# How many pair views each completeness-purity cell keeps. The cell directory no
# longer carries an event, so several examples of the same region live together
# and can be compared without opening three directories.
PAIRS_PER_CELL = 3

# Events where two or more in-volume true neutrinos both have a selected reco
# cluster in the beam window. Their own directory and quota, drawn like the
# cosmics: these are the cases where one flash covers two interactions, and a
# coarse grouping cannot tell them apart. Selected per EVENT rather than per cell,
# so the directory sits directly under the job summary (job_root), not under
# Saved_Clusters.
TWO_NEUTRINO_DIR_NAME = 'two_neutrino_in_beam'
SAVED_TWO_NEUTRINO_VIEWS = 5

# Fixed so a re-run of the same input draws the same two-neutrino events. Change
# it to get a different draw from the same job.
VIEW_SAMPLE_SEED = 12345

_ZOOM_MARGIN = 0.15       # padding around the drawn points, as a fraction of span

_TRUE_STYLE = dict(color='tab:red',  marker='.', s=8,  alpha=0.55, label='true cluster')
_RECO_STYLE = dict(color='tab:blue', marker='.', s=8,  alpha=0.55, label='reco cluster')

# The true interaction vertex, from mc.json's root node -- a single black star on
# the true panels. Big, opaque and drawn on top (zorder) because it has to be
# findable inside a cloud of hundreds of red points, and outlined in white so it
# stays visible where the cloud is dense enough to swallow a plain black marker.
_VERTEX_STYLE = dict(color='black', marker='*', s=260, zorder=5, alpha=1.0,
                     edgecolors='white', linewidths=0.8)

# Its legend entry, as a proxy rather than the scatter's own label: the row
# legend uses markerscale=3 so the s=8 cluster dots can be seen at all, and the
# s=260 star put through that scaling fills the legend box. Sized here to sit
# beside the dots instead.
_VERTEX_LEGEND_MARKERSIZE = 5

# The fiducial boundary, dotted, on every XZ / YZ / XY panel -- true and reco
# alike. The bounds come from selections.FIDUCIAL_BOUNDS_BY_AXIS, the same six
# numbers vertex_in_volume is decided on, so a picture cannot show a boundary
# the pipeline did not use.
#
# It is a REFERENCE LINE, not a cut line: the points are not filtered against it
# (they are cut by the wire-readout volume), so a cluster crossing it is normal
# and expected. What the boundary tells you is whether the true VERTEX -- the
# black star on the true row -- is inside, which is what makes the interaction
# signal or out-of-volume background.
_FIDUCIAL_LINE_STYLE = dict(color='0.30', linestyle=':', linewidth=1.5,
                            alpha=0.9, zorder=1)


def _draw_fiducial_boundary(ax, ix, iy):
    """
    The fiducial edges for one panel, as dotted axis-spanning lines: the two
    bounds of the panel's x coordinate as vertical lines, the two of its y
    coordinate as horizontal ones.

    axvline/axhline rather than a rectangle, deliberately. These panels are
    ZOOMED to the cluster, which is small next to the detector, so a rectangle
    would almost always be drawn entirely outside the frame and its corners are
    never visible anyway; spanning lines show the one or two edges the cluster is
    actually near. They also span the axes in figure fraction rather than data
    coordinates, so they cannot pull the autoscale around -- the limits stay the
    ones computed from the points.

    Returns True when at least one edge falls inside the current limits, so the
    caller can add a legend entry only on the panels that show something.
    """
    from selections import FIDUCIAL_BOUNDS_BY_AXIS, FIDUCIAL_EXCLUDED_BY_AXIS

    drawn = False
    for axis, line, get_lim in ((ix, ax.axvline, ax.get_xlim),
                                (iy, ax.axhline, ax.get_ylim)):
        bounds = FIDUCIAL_BOUNDS_BY_AXIS.get(axis)
        if bounds is None:
            continue
        lo_lim, hi_lim = get_lim()
        # The outer edges, plus the cathode gap in the middle of x -- the
        # fiducial volume is one box per TPC, not one box, and a boundary drawn
        # without the gap would show an accepted band straight through the
        # cathode (see selections.in_fiducial_volume).
        edges = list(bounds) + list(FIDUCIAL_EXCLUDED_BY_AXIS.get(axis, ()))
        for value in edges:
            line(value, **_FIDUCIAL_LINE_STYLE)
            if lo_lim <= value <= hi_lim:
                drawn = True
    return drawn


_FIDUCIAL_LEGEND_HANDLE_KWARGS = dict(color=_FIDUCIAL_LINE_STYLE['color'],
                                      linestyle=_FIDUCIAL_LINE_STYLE['linestyle'],
                                      linewidth=_FIDUCIAL_LINE_STYLE['linewidth'])

_TITLE_FONTSIZE = 15
_LABEL_FONTSIZE = 12
_LEGEND_FONTSIZE = 10


def grid_cell(completeness, purity, cells=_GRID_CELLS):
    """
    Which 10% x 10% cell a pair falls in, as (completeness index, purity index).

    A value of exactly 1.0 belongs to the top cell rather than to a cell of its
    own, so the grid is 10x10 and not 11x11.
    """
    def index(value):
        return min(int((value or 0.0) * cells), cells - 1)
    return index(completeness), index(purity)


def cell_directory_name(completeness, purity, cells=_GRID_CELLS):
    """
    The directory a pair view is written to, e.g.

        completeness_100_90_purity_80_70

    One directory per CELL, holding up to PAIRS_PER_CELL examples from different
    events -- the event is in the filename instead. Each bin is written HIGH
    first, so the directories sort in the order the completeness-purity plane is
    usually read, best at the top, rather than alphabetically from the worst.
    """
    completeness_index, purity_index = grid_cell(completeness, purity, cells)
    width = 100 // cells

    def span(index):
        return f"{(index + 1) * width}_{index * width}"

    return f"completeness_{span(completeness_index)}_purity_{span(purity_index)}"


class ClusterViewSampler:
    """
    Remembers which (channel, cell) slots have been filled and which have not, so
    the event loop can ask "should I draw this one?" and get a cheap answer.

    Kept as an object rather than a module-level dict because a notebook is
    re-run in place: a fresh sampler per job means a re-run fills the same slots
    again instead of silently drawing nothing the second time.
    """

    def __init__(self, cells=_GRID_CELLS, seed=VIEW_SAMPLE_SEED,
                 per_cell=PAIRS_PER_CELL, max_two_neutrino=SAVED_TWO_NEUTRINO_VIEWS):
        self.cells = cells
        self.per_cell = per_cell
        self.filled = Counter()      # (completeness index, purity index) -> count
        self.n_pairs = 0
        # Two-neutrino events are a reservoir sample -- see offer_two_neutrino.
        self.max_two_neutrino = max_two_neutrino
        self.two_neutrino_slots = []
        self.n_two_neutrino_candidates = 0
        self._rng = np.random.default_rng(seed)
        # What was actually saved, for the index file. Metadata only -- no point
        # clouds -- so this stays small however many views are drawn.
        self.saved = []

    def wants_pair(self, channel, completeness, purity):
        """
        Room left in this cell? Counted per CELL, not per (channel, cell): the
        directory holds PAIRS_PER_CELL examples of a region of the plane, and
        which channels they happen to be is recorded in the filenames.
        """
        if channel is None:
            return False
        return self.filled[grid_cell(completeness, purity, self.cells)] < self.per_cell

    def take_pair(self, channel, completeness, purity, record=None, path=None,
                  event_label=None, true_energy=None):
        cell = grid_cell(completeness, purity, self.cells)
        self.filled[cell] += 1
        self.n_pairs += 1
        if record is not None:
            self.saved.append({
                'kind':             'pair',
                'true_energy_mev':  true_energy,
                'channel':          channel,
                'completeness_bin': cell[0] * 10,
                'purity_bin':       cell[1] * 10,
                'event':            event_label,
                'completeness':     record.get('pair_completeness'),
                'purity':           record.get('pair_purity'),
                'true_cluster_id':  record.get('pair_true_cluster_id'),
                'reco_cluster_id':  record.get('reco_cluster_id'),
                'reco_energy_mev':  record.get('reco_energy_mev'),
                'category':         record.get('category'),
                'path':             str(path) if path else None,
            })

    @property
    def n_two_neutrino(self):
        return len(self.two_neutrino_slots)

    def offer_two_neutrino(self):
        """
        Reservoir slot for a two-neutrino event, or None if it is not in the
        sample.

        RESERVOIR SAMPLING (Algorithm R), so the events kept at the end are a
        uniform random draw from every candidate the job saw -- unlike the pairs,
        which are first-encountered. The first max_two_neutrino candidates fill
        the reservoir; candidate k beyond that replaces a random existing one with
        probability max_two_neutrino / k. The cost of uniformity is redrawing: a
        replaced figure is deleted and the new one drawn in its place.
        """
        self.n_two_neutrino_candidates += 1
        if len(self.two_neutrino_slots) < self.max_two_neutrino:
            return len(self.two_neutrino_slots)
        slot = int(self._rng.integers(0, self.n_two_neutrino_candidates))
        return slot if slot < self.max_two_neutrino else None

    def take_two_neutrino(self, slot, path=None, event_label=None, detail=None):
        entry = {
            'kind':             'two_neutrino',
            'channel':          None,
            'completeness_bin': None,
            'purity_bin':       None,
            'event':            event_label,
            'completeness':     None,
            'purity':           None,
            'true_cluster_id':  None,
            'reco_cluster_id':  None,
            'reco_energy_mev':  None,
            'category':         'two_neutrino',
            'detail':           detail or {},
            'path':             str(path) if path else None,
        }
        if slot < len(self.two_neutrino_slots):
            evicted = self.two_neutrino_slots[slot]
            if evicted.get('path'):
                Path(evicted['path']).unlink(missing_ok=True)
            self.saved.remove(evicted)
            self.two_neutrino_slots[slot] = entry
        else:
            self.two_neutrino_slots.append(entry)
        self.saved.append(entry)

    def summary(self):
        return (f"{self.n_pairs} pair view(s), "
                f"{self.n_two_neutrino} two-neutrino view(s) "
                f"from {self.n_two_neutrino_candidates} candidate(s)")


def _draw_panels(fig_title, point_sets, output_path, legend_lines, footer_note=None):
    """
    One figure, three panels (XZ / YZ / XY), each with the same point sets.

    footer_note, when given, is (label, url) printed below the panels -- same
    meaning and same reasoning as in _draw_row_panels, which see.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fiducial_visible = False
    for ax, (ix, iy, xlabel, ylabel, name) in zip(axes, _VIEWS):
        all_x, all_y = [], []
        for points, style in point_sets:
            if points is None or len(points) == 0:
                continue
            points = np.asarray(points)
            ax.scatter(points[:, ix], points[:, iy], **style)
            all_x.append(points[:, ix])
            all_y.append(points[:, iy])
        if all_x:
            # Zoom to the drawn points: these clusters are small next to the
            # detector, and a full-detector frame would show two specks.
            x = np.concatenate(all_x)
            y = np.concatenate(all_y)
            for lo, hi, setter in ((x.min(), x.max(), ax.set_xlim),
                                   (y.min(), y.max(), ax.set_ylim)):
                pad = max((hi - lo) * _ZOOM_MARGIN, 5.0)
                setter(lo - pad, hi + pad)
        # AFTER the limits: the boundary is decoration and must not drag the
        # zoom, and whether an edge is visible can only be asked of a set frame.
        if _draw_fiducial_boundary(ax, ix, iy):
            fiducial_visible = True
        ax.set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE, fontweight='bold')
        ax.set_title(name, fontsize=_LABEL_FONTSIZE, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
    # The legend carries the numbers that make the picture interpretable, so it
    # goes on the first panel where it is read before the eye moves right.
    handles, labels = axes[0].get_legend_handles_labels()
    if fiducial_visible:
        handles.append(Line2D([], [], **_FIDUCIAL_LEGEND_HANDLE_KWARGS))
        labels.append('fiducial boundary')
    axes[0].legend(handles, labels, fontsize=_LEGEND_FONTSIZE, loc='upper left', framealpha=0.9)
    fig.suptitle(fig_title + "\n" + "   |   ".join(legend_lines),
                 fontsize=_TITLE_FONTSIZE, fontweight='bold')
    fig.tight_layout(rect=(0, 0.05 if footer_note else 0, 1, 0.90))
    if footer_note:
        label, url = footer_note
        note = label if not url else f"{label}:  {url}"
        fig.text(0.5, 0.012, note, ha='center', va='bottom',
                 fontsize=_LABEL_FONTSIZE, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='gray', alpha=0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return output_path


def _draw_row_panels(fig_title, rows, output_path, legend_lines, footer_note=None):
    """
    One figure, len(rows) x 3 panels: each row is (row title, [(points, style)]),
    the three columns are XZ / YZ / XY.

    footer_note, when given, is (label, url) printed on one line BELOW the bottom
    row of panels. It is text and not a link: PNG cannot carry a hyperlink
    (matplotlib's url= is honoured only by the vector backends), so the url is
    printed in full to be read, and callers that need a clickable one write it to
    a companion file.

    Axes are SHARED down each column -- every row of a column gets the same
    limits, computed from every point in the figure -- so a feature that appears
    in one row and not another is a real difference and not a change of zoom.
    That is the whole reason for splitting true and reco onto separate rows: the
    eye compares position, and it can only do that on a common frame.

    A row may be given as (row title, sets) or as (row title, sets, vertex),
    where vertex is an (x, y, z) drawn on that row's three panels as a black star
    -- the true interaction vertex, so a true cluster can be read against the
    point the interaction started from. The vertex is folded into the SHARED
    limits: a marker outside the frame would be no marker at all, and for an
    out-of-volume interaction the distance from the vertex to the deposits is the
    very thing the picture is being looked at for. That does mean an interaction
    whose vertex is far from its deposits draws a wider frame than the points
    alone would need.
    """
    n = len(rows)
    # (title, sets) and (title, sets, vertex) are both accepted, so the callers
    # that have no vertex to give stay as they are.
    rows = [(row[0], row[1], row[2] if len(row) > 2 else None) for row in rows]
    fig, axes = plt.subplots(n, 3, figsize=(19, 5.6 * n), squeeze=False)
    everything = [np.asarray(p) for _, sets, _ in rows for p, _ in sets
                  if p is not None and len(p)]
    vertices = [np.asarray(v, dtype=float).reshape(1, 3) for _, _, v in rows
                if v is not None]
    for col, (ix, iy, xlabel, ylabel, name) in enumerate(_VIEWS):
        framed = everything + vertices
        lims = []
        for axis in (ix, iy):
            values = np.concatenate([p[:, axis] for p in framed]) if framed else None
            if values is None or not len(values):
                lims.append(None)
                continue
            lo, hi = values.min(), values.max()
            pad = max((hi - lo) * _ZOOM_MARGIN, 5.0)
            lims.append((lo - pad, hi + pad))
        for row, (row_title, sets, vertex) in enumerate(rows):
            ax = axes[row][col]
            for points, style in sets:
                if points is not None and len(points):
                    points = np.asarray(points)
                    ax.scatter(points[:, ix], points[:, iy], **style)
            if vertex is not None:
                ax.scatter([vertex[ix]], [vertex[iy]], **_VERTEX_STYLE)
            if lims[0]:
                ax.set_xlim(*lims[0])
            if lims[1]:
                ax.set_ylim(*lims[1])
            # AFTER the limits, for the same two reasons as in _draw_panels.
            fiducial_visible = _draw_fiducial_boundary(ax, ix, iy)
            ax.set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE, fontweight='bold')
            ax.set_title(f"{name} -- {row_title}", fontsize=_LABEL_FONTSIZE, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if vertex is not None:
                handles.append(Line2D([], [], linestyle='none', color='black',
                                      marker='*', markeredgecolor='white',
                                      markersize=_VERTEX_LEGEND_MARKERSIZE))
                labels.append('true vertex')
            if fiducial_visible:
                handles.append(Line2D([], [], **_FIDUCIAL_LEGEND_HANDLE_KWARGS))
                labels.append('fiducial boundary')
            if col == 0 and handles:
                ax.legend(handles, labels, fontsize=_LEGEND_FONTSIZE,
                          loc='upper left', framealpha=0.9, markerscale=3)
    fig.suptitle(fig_title + "\n" + "   |   ".join(legend_lines),
                 fontsize=_TITLE_FONTSIZE, fontweight='bold')
    bottom = 0.035 if footer_note else 0.0
    fig.tight_layout(rect=(0, bottom, 1, 0.94 if n > 1 else 0.90))
    if footer_note:
        # BELOW the bottom row, centred, on one line -- the figure is wide enough
        # for the whole url, and out here it does not cover any data. Space for it
        # is reserved in the tight_layout rect above rather than left to
        # bbox_inches='tight', so the panels shrink to make room instead of the
        # note sitting flush against the axis labels.
        label, url = footer_note
        note = label if not url else f"{label}:  {url}"
        fig.text(0.5, 0.012, note, ha='center', va='bottom',
                 fontsize=_LABEL_FONTSIZE, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='gray', alpha=0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return output_path


def draw_pair_views(record, clusters_true, clusters_reco, output_root, event_label):
    """
    One reco-true pair in XZ, YZ and XY, with the numbers that identify it.

    Parameters:
    - record: a categorize_reco_clusters record (carries the pair's purity,
        completeness, channel, category and both cluster ids)
    - clusters_true / clusters_reco: this event's point dicts
    - output_root: the Saved_Clusters directory (the grid cells)
    - event_label: e.g. 'chunk1_37'; the per-event subdirectory is named from it
    """
    true_points = clusters_true.get(record['pair_true_cluster_id'])
    reco_points = clusters_reco.get(record['reco_cluster_id'])
    if reco_points is None:
        return None

    # Column 5 is the per-point true energy, so this is the true cluster's
    # deposited energy AFTER the cuts -- the same quantity the truth-side stacks
    # are filled with, not the pre-cut sum the efficiency is binned in.
    true_energy = float(np.asarray(true_points)[:, 5].sum()) if true_points is not None else None

    in_volume = 'in-volume' if record['category'] != 'out_of_volume' else 'OUT-of-volume'
    legend_lines = [
        f"event {event_label}",
        f"{record['channel']}",
        in_volume,
        f"purity {record['pair_purity']:.3f}",
        f"completeness {record['pair_completeness']:.3f}",
        f"true E {true_energy:.0f} MeV" if true_energy is not None else "true E n/a",
        f"true id {record['pair_true_cluster_id']:.0f}",
        f"reco id {record['reco_cluster_id']:.3f}",
    ]
    directory = cell_directory_name(record['pair_completeness'], record['pair_purity'])
    return _draw_row_panels(
        f"Reco-true pair -- {record['channel']}",
        [("TRUE cluster", [(true_points, _TRUE_STYLE)]),
         ("RECO cluster", [(reco_points, _RECO_STYLE)])],
        Path(output_root) / directory / f"pair_{record['channel']}_{event_label}.png",
        legend_lines)


_TWO_NU_TRUE_STYLES = [dict(color='tab:red', marker='.', s=8, alpha=0.55),
                       dict(color='tab:orange', marker='.', s=8, alpha=0.55),
                       dict(color='tab:brown', marker='.', s=8, alpha=0.55)]
_TWO_NU_RECO_STYLES = [dict(color='tab:blue', marker='.', s=8, alpha=0.55),
                       dict(color='tab:green', marker='.', s=8, alpha=0.55),
                       dict(color='tab:purple', marker='.', s=8, alpha=0.55)]


def two_neutrino_groups(categorized_records):
    """
    {true cluster id: [records]} for events where TWO OR MORE in-volume true
    neutrinos each have a selected reco cluster -- otherwise {}.

    This is the population that makes a coarse, flash-based grouping dangerous:
    one beam flash covers both interactions, so nothing in the timing separates
    them, and merging on it fuses two neutrinos into one object.
    """
    by_true = {}
    for record in categorized_records or []:
        if record['category'] != 'contaminated' and not record['category'].startswith('high_signal_'):
            continue
        if record['pair_true_cluster_id'] is None:
            continue
        by_true.setdefault(record['pair_true_cluster_id'], []).append(record)
    return by_true if len(by_true) >= 2 else {}


def draw_two_neutrino_views(by_true, clusters_true, clusters_reco, output_root, event_label):
    """
    One event holding two or more separately-reconstructed neutrinos: the true
    clusters on the top row, the reco clusters that pair to them below, one
    colour per neutrino so it is visible whether they overlap or sit apart.
    """
    true_sets, reco_sets, lines = [], [], [f"event {event_label}"]
    for n, (true_id, records) in enumerate(sorted(by_true.items())):
        true_points = clusters_true.get(true_id)
        if true_points is not None and len(true_points):
            style = dict(_TWO_NU_TRUE_STYLES[n % len(_TWO_NU_TRUE_STYLES)])
            style['label'] = f'true {true_id:.0f}'
            true_sets.append((true_points, style))
        for m, record in enumerate(records):
            reco_points = clusters_reco.get(record['reco_cluster_id'])
            if reco_points is None or not len(reco_points):
                continue
            style = dict(_TWO_NU_RECO_STYLES[n % len(_TWO_NU_RECO_STYLES)])
            style['label'] = (f"reco {record['reco_cluster_id']:.0f} -> {true_id:.0f}"
                              if m == 0 else None)
            reco_sets.append((reco_points, style))
        best = max(records, key=lambda r: r['pair_completeness'] or 0.0)
        lines.append(f"{true_id:.0f} {best['channel'] or '?'}: "
                     f"compl {best['pair_completeness']:.2f} pur {best['pair_purity']:.2f}")
    if not true_sets or not reco_sets:
        return None
    return _draw_row_panels(
        f"{len(by_true)} in-volume neutrinos in one beam window",
        [("TRUE clusters", true_sets), ("RECO clusters", reco_sets)],
        Path(output_root) / TWO_NEUTRINO_DIR_NAME / f"two_neutrino_{event_label}.png",
        lines)


def save_event_cluster_views(categorized_records, clusters_true, clusters_reco,
                             sampler, output_root, event_label,
                             first_neutrino_only=True,
                             job_root=None,
                             draw_pair_cells=True, draw_two_neutrino=True):
    """
    Draw whatever this event contributes to the sample: pairs for cells not yet
    filled and two-neutrino events offered to the reservoir. Returns the number of
    figures drawn (which counts a two-neutrino view that is later evicted and
    deleted).

    first_neutrino_only restricts pair views to the event's FIRST neutrino -- see
    FIRST_NEUTRINO_CLUSTER_ID.

    output_root takes the grid cells and nothing else; job_root takes
    TWO_NEUTRINO_DIR_NAME -- see DIRECTORY LAYOUT. It defaults to output_root, so a
    caller that does not care still gets one self-contained tree.

    draw_pair_cells=False suppresses the grid cells, so output_root is never
    created; draw_two_neutrino=False does the same for TWO_NEUTRINO_DIR_NAME. Both
    exist because these views describe the INPUT SAMPLE rather than the code under
    study, so a job re-run on the same sample would only redraw what an earlier one
    already has.
    """
    drawn = 0
    job_root = Path(job_root) if job_root is not None else Path(output_root)

    # Two-neutrino events first: the decision is per EVENT, not per cluster, so it
    # does not belong in the per-record loop below.
    by_true = two_neutrino_groups(categorized_records) if draw_two_neutrino else {}
    if by_true:
        slot = sampler.offer_two_neutrino()
        if slot is not None:
            path = draw_two_neutrino_views(by_true, clusters_true, clusters_reco,
                                           job_root, event_label)
            if path:
                sampler.take_two_neutrino(
                    slot, path=path, event_label=event_label,
                    detail={'n_neutrinos': len(by_true),
                            'true_ids': sorted(by_true),
                            'reco_ids': sorted(r['reco_cluster_id']
                                               for rs in by_true.values() for r in rs)})
                drawn += 1

    for record in categorized_records or []:
        # Cosmic candidates are drawn by Draw_Selection_Cosmics.ipynb
        # (draw_selection_cosmics.py), not here.
        if record['category'] == 'cosmic':
            continue
        # Only in-volume pairs are sampled across the grid: out-of-volume pairs are
        # a rejection category, and their completeness/purity are not what the grid
        # is about.
        if record['category'] == 'out_of_volume' or record['pair_true_cluster_id'] is None:
            continue
        if not draw_pair_cells:
            continue
        if first_neutrino_only and record['pair_true_cluster_id'] != FIRST_NEUTRINO_CLUSTER_ID:
            continue
        channel = record['channel']
        if not sampler.wants_pair(channel, record['pair_completeness'], record['pair_purity']):
            continue
        path = draw_pair_views(record, clusters_true, clusters_reco, output_root, event_label)
        if path:
            true_points = clusters_true.get(record['pair_true_cluster_id'])
            true_energy = (float(np.asarray(true_points)[:, 5].sum())
                           if true_points is not None else None)
            sampler.take_pair(channel, record['pair_completeness'], record['pair_purity'],
                              record=record, path=path, event_label=event_label,
                              true_energy=true_energy)
            drawn += 1
    return drawn


def _relative_dir(root, base, dir_name):
    """
    'dir_name/' as a path the reader can follow from the index file itself.

    The per-event sets are written outside the index's own directory
    (see DIRECTORY LAYOUT), so naming them bare would point at directories that do
    not exist beside the index. os.path.relpath rather than Path.relative_to
    because the target is normally a sibling, needing a '../' the latter refuses
    to produce; an unrelated root (a different drive on Windows) falls back to the
    absolute path.
    """
    target = Path(root) / dir_name
    try:
        return os.path.relpath(target, base) + '/'
    except ValueError:
        return str(target) + '/'


def write_cluster_view_index(sampler, output_root, filename='completeness_purity.txt',
                             job_root=None, index_root=None):
    """
    An index of every view drawn, so a plot can be chosen from the text rather
    than by opening files one at a time.

    Sorted by channel then by cell, and the cells are laid out as a grid at the
    end: reading down a column shows how the pictures change with purity at fixed
    completeness, which is the comparison the sample exists to make. Cells with no
    entry are shown empty, because knowing that a corner of the plane never
    occurs is as useful as seeing one that does.

    job_root must be the one save_event_cluster_views drew with, so the directories
    the index names are the ones the files are in; it defaults to output_root,
    matching that function's own default.

    index_root is where the file itself goes, which is NOT output_root: the index
    explains the completeness-purity scatter, so it belongs beside that plot, and
    keeping it out of Saved_Clusters means a job that draws no grid cells leaves no
    Saved_Clusters directory at all. Every path in the text is written relative to
    it, so the whole index stays followable from wherever it lands.
    """
    output_root = Path(output_root)
    index_root = Path(index_root) if index_root is not None else output_root
    index_root.mkdir(parents=True, exist_ok=True)
    path = index_root / filename
    two_nu_dir = _relative_dir(job_root if job_root is not None else output_root,
                               index_root, TWO_NEUTRINO_DIR_NAME)
    # The grid cells are named relative to the index too, so a pair row can be
    # followed from wherever the index sits. Empty when the index is inside
    # output_root, which is what the cell names alone already mean.
    cell_prefix = _relative_dir(output_root, index_root, '')
    cell_prefix = '' if cell_prefix == './' else cell_prefix

    pairs   = [s for s in sampler.saved if s['kind'] == 'pair']
    two_nu  = [s for s in sampler.saved if s['kind'] == 'two_neutrino']

    lines = []
    lines.append("=" * 108)
    lines.append("SAVED CLUSTER VIEWS -- index")
    lines.append("=" * 108)
    # The grid sections are written only when pair views were actually drawn --
    # with save_event_cluster_views(draw_pair_cells=False) there are none, and
    # describing a layout that is not on disc would send the reader looking for
    # directories that do not exist.
    if pairs:
        lines.append("One reco-true pair per 10% x 10% cell of the completeness-purity plane, per")
        lines.append("interaction channel, drawn in XZ, YZ and XY. The pair shown for a cell is the")
        lines.append("FIRST one the event loop met there, not a uniform draw from it.")
        lines.append("")
        lines.append(f"LAYOUT. One directory per CELL, up to {sampler.per_cell} example(s) in each:")
        lines.append("")
        lines.append(f"    {cell_prefix}completeness_100_90_purity_80_70/pair_<channel>_<event>.png")
        lines.append("")
        lines.append("Each bin is written high value first, so the directories sort best-first.")
        lines.append("Every pair view has TWO ROWS -- the true cluster above, the reco cluster")
        lines.append("below -- sharing one set of axes per column, so positions can be compared.")
        lines.append("Only the event's FIRST neutrino is drawn.")
        lines.append("")
    lines.append("")
    if two_nu:
        lines.append(f"TWO-NEUTRINO EVENTS -> {two_nu_dir}")
        lines.append("")
        lines.append(f"{sampler.max_two_neutrino} event(s) drawn at random from the")
        lines.append(f"{sampler.n_two_neutrino_candidates} in which two or more in-volume true neutrinos each")
        lines.append("produced a selected reco cluster -- one beam flash covering two interactions,")
        lines.append("which is the case a flash-based grouping cannot separate.")
        lines.append("")
        lines.append("")
    lines.append(f"{len(pairs)} pair view(s), {len(two_nu)} two-neutrino view(s).")
    lines.append("")
    if pairs:
        lines.append("-" * 108)
        lines.append(f"  {'channel':<9s}{'comp bin':>9s}{'pur bin':>9s}{'completeness':>14s}{'purity':>9s}"
                     f"{'E true':>9s}{'E reco':>9s}{'true id':>10s}{'reco id':>11s}  file")
        lines.append("-" * 108)
    for entry in sorted(pairs, key=lambda s: (s['channel'], -s['completeness_bin'], s['purity_bin'])):
        name = Path(entry['path']).name if entry['path'] else ''
        lines.append(
            f"  {entry['channel']:<9s}{entry['completeness_bin']:>7d}-{entry['completeness_bin']+10:<2d}"
            f"{entry['purity_bin']:>6d}-{entry['purity_bin']+10:<2d}"
            f"{entry['completeness']:>14.4f}{entry['purity']:>9.4f}"
            f"{(entry.get('true_energy_mev') or 0):>9.0f}{entry['reco_energy_mev']:>9.0f}"
            f"{entry['true_cluster_id']:>10.0f}{entry['reco_cluster_id']:>11.3f}"
            f"  {cell_prefix}{cell_directory_name(entry['completeness'], entry['purity'])}/{name}")

    if two_nu:
        lines.append("")
        lines.append("-" * 108)
        lines.append(f"  {'two-neutrino events':<34s}{'event':>12s}{'neutrinos':>11s}"
                     f"  true ids -> reco ids")
        lines.append("-" * 108)
        for entry in two_nu:
            d = entry.get('detail') or {}
            trues = ','.join(f"{v:.0f}" for v in d.get('true_ids', []))
            recos = ','.join(f"{v:.0f}" for v in d.get('reco_ids', []))
            lines.append(f"  {'':<34s}{entry['event']:>12s}{d.get('n_neutrinos', 0):>11d}"
                         f"  {trues} -> {recos}")

    # The occupancy grid: which cells have a picture and which never occurred.
    for channel in sorted({s['channel'] for s in pairs}):
        cells = {(s['completeness_bin'], s['purity_bin']) for s in pairs if s['channel'] == channel}
        lines.append("")
        lines.append("-" * 108)
        lines.append(f"{channel}: cells with a saved view  (X = saved, . = no pair fell here)")
        lines.append("-" * 108)
        lines.append("     purity ->   " + "".join(f"{p:>5d}" for p in range(0, 100, 10)))
        for completeness in range(90, -10, -10):
            row = "".join("    X" if (completeness, p) in cells else "    ."
                          for p in range(0, 100, 10))
            lines.append(f"  completeness {completeness:>3d}" + row)
    lines.append("=" * 108)

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return path
