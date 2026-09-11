import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# The beam window, in us. Widened from [0.33, 1.93] (1.6 us) to [0.20, 2.20]
# (2.0 us): 7 of the 11 true neutrinos that found no reco match were lost to the
# beam-window cut rather than to reconstruction -- the charge was there, its
# charge-light flash simply fell outside the window -- so the edges are worth
# opening up. Everything downstream reads these two constants: the reco-side cut
# in the evaluation notebooks, the beam-window counts in SelectionAnalysis, both
# investigation scripts, and the shaded band plus the flash-offset columns in the
# plots and tables here.
BEAM_WINDOW_MIN_US = 0.20
BEAM_WINDOW_MAX_US = 2.20
BEAM_WINDOW_WIDTH_US = BEAM_WINDOW_MAX_US - BEAM_WINDOW_MIN_US  # 2.0 us
FINE_BIN_WIDTH_US = 0.010  # 10 ns


def _aligned_bin_edges(data_min, data_max, bin_width, align_point):
    """
    Bin edges of the given width, guaranteed to include an edge exactly at
    align_point (e.g. the beam-window boundary), tiled outward to cover
    [data_min, data_max].
    """
    n_below = max(int(np.ceil((align_point - data_min) / bin_width)), 0)
    n_above = max(int(np.ceil((data_max - align_point) / bin_width)), 0)
    start = align_point - n_below * bin_width
    stop = align_point + n_above * bin_width
    return np.arange(start, stop + bin_width / 2, bin_width)


def draw_flashes(flash_metadata_list, output_dir, apa, level_name, filename_prefix, file_name=None, write_text_table=False):
    """
    Plot optical flash time (x-axis) vs number of matched img-global reco
    clusters (y-axis, one entry per cluster-flash record so multi-cluster
    flashes count correctly), from the per-cluster-flash records built by
    metadata.build_cluster_flash_metadata(). Used at Event, File, and Job
    level alike -- flash_metadata_list can span multiple events.

    Rendered as markers at binned (time, count) points, not raw per-flash
    scatter: two flashes can land within nanoseconds of each other (seen in
    practice) and a raw scatter would render them as a single overlapping
    marker, hiding a real second entry. Binning first and placing one marker
    per bin at (bin_center, summed_count) keeps that count visible while
    still looking like a marker plot rather than a bar histogram.

    Saves two PNGs: a coarse one (bin width = beam-window width, i.e.
    BEAM_WINDOW_MAX_US - BEAM_WINDOW_MIN_US, over the full flash-time range)
    so the beam window is directly comparable to its neighboring bins, and a
    fine one (bin width = FINE_BIN_WIDTH_US, x in [-2, 4] us) that can
    resolve flashes close together in time. Both have bin edges aligned so
    one edge sits exactly on the beam-window boundary, and both shade/label
    the beam window. Empty bins are not plotted. If write_text_table=True,
    also saves a flashes.txt table listing every record's cluster ID and
    flash time for exact cross-checking (event level only by convention --
    file/job level calls just aggregate onto the plots, not the text table).

    Parameters:
    - flash_metadata_list: List of dicts from build_cluster_flash_metadata(),
        each with 'event', 'flash_index', 'flash_time' (and other fields, unused here)
    - output_dir: Output directory
    - apa: APA identifier (label only, e.g. "Combined")
    - level_name: Label for the title (e.g. 'Event 9', 'File Level', 'Job Level')
    - filename_prefix: Suffix used in the output filename (e.g. 'event_9', 'file', 'job')
    - file_name: Optional input file name for title
    - write_text_table: If True, also write flashes.txt (default False)
    """
    if not flash_metadata_list:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One entry per cluster-flash record (not deduplicated by flash) so a
    # binned count directly equals "number of imaging clusters" -- a flash
    # matched to 3 clusters contributes 3 entries at that same time.
    all_flash_times = [record['flash_time'] for record in flash_metadata_list]

    title = f'Flash Time vs Matched Imaging Clusters: {level_name}, {apa}'
    if file_name:
        title += f' ({file_name})'

    # Coarse: bin width = beam-window width, full flash-time range. Markers
    # at (bin_center, count); empty bins skipped.
    coarse_edges = _aligned_bin_edges(min(all_flash_times), max(all_flash_times), BEAM_WINDOW_WIDTH_US, BEAM_WINDOW_MIN_US)
    coarse_counts, coarse_edges = np.histogram(all_flash_times, bins=coarse_edges)
    coarse_centers = (coarse_edges[:-1] + coarse_edges[1:]) / 2
    nonzero = coarse_counts > 0
    plt.figure(figsize=(10, 6))
    plt.scatter(coarse_centers[nonzero], coarse_counts[nonzero], s=60, color='darkorange', edgecolor='black', alpha=0.8)
    plt.axvspan(BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US, color='gold', alpha=0.25, label='beam-window')
    plt.xlabel('Flash Time [us]', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Imaging Clusters', fontsize=12, fontweight='bold')
    plt.title(f'{title}\n(coarse bins, width={BEAM_WINDOW_WIDTH_US:.2f} us)', fontsize=12, fontweight='bold', wrap=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / f'flash_time_vs_num_clusters_{filename_prefix}_{apa}.png', dpi=100, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    # Fine: bin width = FINE_BIN_WIDTH_US, zoomed to [-2, 4] us around the
    # beam window. Same marker-at-bin-center approach.
    fine_edges = _aligned_bin_edges(-2, 4, FINE_BIN_WIDTH_US, BEAM_WINDOW_MIN_US)
    fine_counts, fine_edges = np.histogram(all_flash_times, bins=fine_edges)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2
    nonzero = fine_counts > 0
    plt.figure(figsize=(10, 6))
    plt.scatter(fine_centers[nonzero], fine_counts[nonzero], s=60, color='darkorange', edgecolor='black', alpha=0.8)
    plt.axvspan(BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US, color='gold', alpha=0.25, label='beam-window')
    plt.xlim(-2, 4)
    plt.xlabel('Flash Time [us]', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Imaging Clusters', fontsize=12, fontweight='bold')
    plt.title(f'{title}\n(fine bins, width={FINE_BIN_WIDTH_US*1000:.0f} ns)', fontsize=12, fontweight='bold', wrap=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / f'flash_time_vs_num_clusters_{filename_prefix}_{apa}_beam_window_zoom.png', dpi=100, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    # flashes.txt: one row per cluster-flash record, for manual cross-checking
    # of cluster IDs against their flash times. Event level only (opt-in).
    if write_text_table:
        columns = [
            ('event', 15), ('flash_index', 12), ('reco_cluster_id', 16),
            ('flash_time_us', 15), ('flash_apa', 10), ('in_beam_window', 15),
        ]
        with open(output_dir / 'flashes.txt', 'w') as f:
            f.write(''.join(name.ljust(width) for name, width in columns) + '\n')
            for record in sorted(flash_metadata_list, key=lambda r: (str(r['event']), r['flash_index'])):
                in_beam = 'yes' if BEAM_WINDOW_MIN_US <= record['flash_time'] <= BEAM_WINDOW_MAX_US else 'no'
                values = [
                    str(record['event']), str(record['flash_index']), str(int(record['reco_cluster_id'])),
                    f"{record['flash_time']:.3f}", str(record['flash_apa']), in_beam,
                ]
                f.write(''.join(v.ljust(width) for v, (_, width) in zip(values, columns)) + '\n')


def draw_clustering_flashes(cluster_flash_records, output_dir, apa, level_name, filename_prefix, file_name=None, write_text_table=False):
    """
    Plot optical flash time (x-axis) vs number of matched clustering-global
    clusters (y-axis), from the per-(clustering_cluster_id, img_cluster_id)
    records built by metadata.build_img_cluster_flash_metadata(). Used at
    Event, File, and Job level alike -- cluster_flash_records can span
    multiple events. Same binning/marker/beam-window styling as draw_flashes(),
    just counting clustering clusters instead of img-global clusters.

    A clustering cluster that maps to several img clusters sharing the SAME
    flash (e.g. re-fragmented pieces of one track, re-merged) is counted once
    for that flash, not once per img cluster -- deduplicated by
    (event, clustering_cluster_id, flash_index) before binning. A clustering
    cluster that maps to img clusters with DIFFERENT flash times (e.g. a
    cathode-crossing merge matched by two flashes, one per APA) counts once
    per distinct flash, which is exactly the signal worth seeing. (Genuine
    cathode-crossing pairs are already averaged into a single record by
    build_img_cluster_flash_metadata before this function ever sees them.)

    Parameters:
    - cluster_flash_records: List of dicts from build_img_cluster_flash_metadata(),
        each with 'event', 'clustering_cluster_id', 'img_cluster_id', 'img_cluster_ids',
        'flash_index', 'flash_time', 'flash_apa', 'is_cathode_crossing', 'n_matched_points'
    - output_dir: Output directory
    - apa: APA identifier (label only, e.g. "Combined")
    - level_name: Label for the title (e.g. 'Event 9', 'File Level', 'Job Level')
    - filename_prefix: Suffix used in the output filename (e.g. 'event_9', 'file', 'job')
    - file_name: Optional input file name for title
    - write_text_table: If True, also write flashes.txt (default False; event level only by convention)
    """
    if not cluster_flash_records:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    all_flash_times = []
    for record in cluster_flash_records:
        key = (record['event'], record['clustering_cluster_id'], record['flash_index'])
        if key in seen:
            continue
        seen.add(key)
        all_flash_times.append(record['flash_time'])

    n_in_beam_window = sum(1 for t in all_flash_times if BEAM_WINDOW_MIN_US <= t <= BEAM_WINDOW_MAX_US)
    n_out_of_beam_window = len(all_flash_times) - n_in_beam_window

    title = f'Flash Time vs Matched Clusters: {level_name}, {apa}'
    if file_name:
        title += f' ({file_name})'

    # Coarse: bin width = beam-window width, full flash-time range.
    coarse_edges = _aligned_bin_edges(min(all_flash_times), max(all_flash_times), BEAM_WINDOW_WIDTH_US, BEAM_WINDOW_MIN_US)
    coarse_counts, coarse_edges = np.histogram(all_flash_times, bins=coarse_edges)
    coarse_centers = (coarse_edges[:-1] + coarse_edges[1:]) / 2
    nonzero = coarse_counts > 0
    plt.figure(figsize=(10, 6))
    plt.scatter(coarse_centers[nonzero], coarse_counts[nonzero], s=60, color='steelblue', edgecolor='black', alpha=0.8)
    plt.axvspan(BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US, color='gold', alpha=0.25,
                label=f'beam-window (in={n_in_beam_window}, out={n_out_of_beam_window})')
    plt.xlabel('Flash Time [us]', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    plt.title(f'{title}\n(coarse bins, width={BEAM_WINDOW_WIDTH_US:.2f} us)', fontsize=12, fontweight='bold', wrap=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / f'flash_time_vs_num_clusters_{filename_prefix}_{apa}.png', dpi=100, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    # Fine: bin width = FINE_BIN_WIDTH_US, zoomed to [-2, 4] us around the beam window.
    fine_edges = _aligned_bin_edges(-2, 4, FINE_BIN_WIDTH_US, BEAM_WINDOW_MIN_US)
    fine_counts, fine_edges = np.histogram(all_flash_times, bins=fine_edges)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2
    nonzero = fine_counts > 0
    plt.figure(figsize=(10, 6))
    plt.scatter(fine_centers[nonzero], fine_counts[nonzero], s=60, color='steelblue', edgecolor='black', alpha=0.8)
    plt.axvspan(BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US, color='gold', alpha=0.25,
                label=f'beam-window (in={n_in_beam_window})')
    plt.xlim(-2, 4)
    plt.xlabel('Flash Time [us]', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    plt.title(f'{title}\n(fine bins, width={FINE_BIN_WIDTH_US*1000:.0f} ns)', fontsize=12, fontweight='bold', wrap=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / f'flash_time_vs_num_clusters_{filename_prefix}_{apa}_beam_window_zoom.png', dpi=100, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    # flashes.txt: one row per resolved clustering-cluster flash match, for
    # manual cross-checking of cluster IDs (both clustering-global and their
    # matched img-global cluster(s)) against flash times. Event level only (opt-in).
    if write_text_table:
        columns = [
            ('event', 15), ('clustering_cluster_id', 22), ('img_cluster_ids', 18),
            ('n_matched_points', 17), ('flash_time_us', 15), ('flash_apa', 10),
            ('in_beam_window', 15), ('cathode_crossing', 17),
        ]
        with open(output_dir / 'flashes.txt', 'w') as f:
            f.write(''.join(name.ljust(width) for name, width in columns) + '\n')
            for record in sorted(cluster_flash_records, key=lambda r: (str(r['event']), r['clustering_cluster_id'])):
                in_beam = 'yes' if BEAM_WINDOW_MIN_US <= record['flash_time'] <= BEAM_WINDOW_MAX_US else 'no'
                img_ids_str = ','.join(str(int(cid)) for cid in record['img_cluster_ids'])
                values = [
                    str(record['event']), str(int(record['clustering_cluster_id'])), img_ids_str,
                    str(record['n_matched_points']), f"{record['flash_time']:.6f}", str(record['flash_apa']),
                    in_beam, str(record['is_cathode_crossing']),
                ]
                f.write(''.join(v.ljust(width) for v, (_, width) in zip(values, columns)) + '\n')


def draw_cluster_flash_time_bar(cluster_flash_records, event, apa, output_dir, file_name=None):
    """
    Bar chart of flash time (y-axis) per clustering-global cluster ID
    (x-axis), from the records built by metadata.build_img_cluster_flash_metadata().
    Event level only.

    A clustering cluster with more than one distinct flash match (e.g. a
    cathode-crossing merge matched by two flashes, one per APA) gets one bar
    per distinct flash, dodged side-by-side at that cluster's x position,
    rather than losing the second flash to overwriting. Deduplicated the
    same way as draw_clustering_flashes(): by (clustering_cluster_id,
    flash_index), so img clusters sharing one flash don't produce duplicate bars.
    Each bar is labeled with its exact flash time (3 decimals), printed above
    the bar if positive or below it if negative.

    Parameters:
    - cluster_flash_records: List of dicts from build_img_cluster_flash_metadata(),
        each with 'clustering_cluster_id', 'flash_index', 'flash_time'
    - event: Event number
    - apa: APA label (e.g. "Combined")
    - output_dir: Output directory
    - file_name: Optional input file name for title
    """
    if not cluster_flash_records:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    per_cluster_times = {}
    for record in cluster_flash_records:
        key = (record['clustering_cluster_id'], record['flash_index'])
        if key in seen:
            continue
        seen.add(key)
        per_cluster_times.setdefault(record['clustering_cluster_id'], []).append(record['flash_time'])

    cluster_ids = sorted(per_cluster_times.keys())
    max_flashes_per_cluster = max(len(times) for times in per_cluster_times.values())
    bar_width = 0.8 / max_flashes_per_cluster

    all_times = [t for times in per_cluster_times.values() for t in times]
    y_span = (max(all_times) - min(all_times)) if len(all_times) > 1 else abs(all_times[0]) or 1.0
    label_offset = y_span * 0.02

    plt.figure(figsize=(max(10, len(cluster_ids) * 0.5), 6))
    for slot in range(max_flashes_per_cluster):
        xs, ys = [], []
        for i, cid in enumerate(cluster_ids):
            times = per_cluster_times[cid]
            if slot < len(times):
                xs.append(i + (slot - (max_flashes_per_cluster - 1) / 2) * bar_width)
                ys.append(times[slot])
        plt.bar(xs, ys, width=bar_width, color='steelblue', edgecolor='black', alpha=0.8)
        # Print the flash time value on each bar (above if positive, below if negative).
        for x, y in zip(xs, ys):
            va = 'bottom' if y >= 0 else 'top'
            plt.text(x, y + (label_offset if y >= 0 else -label_offset), f'{y:.3f}',
                      ha='center', va=va, fontsize=6, rotation=90)

    plt.axhspan(BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US, color='gold', alpha=0.25, label='beam-window')
    plt.xticks(range(len(cluster_ids)), [f'{int(cid)}' for cid in cluster_ids], rotation=90, fontsize=7)
    plt.xlabel('Clustering Cluster ID', fontsize=12, fontweight='bold')
    plt.ylabel('Flash Time [us]', fontsize=12, fontweight='bold')
    title = f'Flash Time per Cluster: Event {event}, {apa}'
    if file_name:
        title += f' ({file_name})'
    plt.title(title, fontsize=12, fontweight='bold', wrap=True)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.legend()
    plt.savefig(output_dir / f'cluster_flash_time_bar_event{event}_{apa}.png', dpi=100, bbox_inches='tight', pad_inches=0.3)
    plt.close()


def draw_img_global_clusters(all_clusters, flash_clusters, beam_window_clusters, event, apa, output_dir, file_name=None):
    """
    Draw img-global reco clusters in XZ, YZ, XY projections as a 3x3 grid of
    panels: row 1 = all clusters, row 2 = only clusters with a matched optical
    flash (any time), row 3 = only clusters whose matched flash falls inside
    the beam window. XZ/YZ put Z on the x-axis (matching
    DrawRecoTrueClusters.py's DrawTrueRecoClustersXZ/YZ convention). Event
    level only -- not meant to be aggregated to file/job level.

    Parameters:
    - all_clusters, flash_clusters, beam_window_clusters: dicts {cluster_id: points},
        points columns [x, y, z, cluster_id, charge]. These must be grouped by
        img-global.json's RAW point-wise cluster_id (i.e. NOT reassigned via
        reassign_cluster_ID_reco), since that's the namespace op_cluster_ids uses.
        flash_clusters and beam_window_clusters must be subsets of all_clusters
        (same cluster_id keys) -- that's what lets every panel use the same color
        per cluster ID.
    - event: Event number
    - apa: APA label (e.g. "Combined")
    - output_dir: Output directory
    - file_name: Optional input file name for title
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Same color for the same cluster_id in every panel, instead of each
    # subplot's independent color cycle (which would assign colors by
    # plotting order within that axis -- a cluster could end up a different
    # color in "All Clusters" vs "Clusters With Flash"). Built once from
    # all_clusters since the other two rows are subsets of it.
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cluster_colors = {
        cluster_id: color_cycle[i % len(color_cycle)]
        for i, cluster_id in enumerate(sorted(all_clusters.keys()))
    }

    # (row_label, clusters, show_legend) -- "All Clusters" is typically dozens
    # of clusters (no selection cuts applied upstream yet); a per-cluster
    # legend there is unreadable and dominates the figure, so skip it.
    rows = [
        ("All Clusters", all_clusters, False),
        ("Clusters With Flash", flash_clusters, True),
        ("Clusters In Beam Window", beam_window_clusters, True),
    ]
    # (view_label, x_col, y_col, xlim, ylim, xlabel, ylabel)
    views = [
        ("XZ", 2, 0, (0, 500), (-250, 250), "z [cm]", "x [cm]"),
        ("YZ", 2, 1, (0, 500), (-250, 250), "z [cm]", "y [cm]"),
        ("XY", 0, 1, (-250, 250), (-250, 250), "x [cm]", "y [cm]"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for row_idx, (row_label, clusters, show_legend) in enumerate(rows):
        for col_idx, (view_label, x_col, y_col, xlim, ylim, xlabel, ylabel) in enumerate(views):
            ax = axes[row_idx, col_idx]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            title = f"{row_label}: Event {event}, {apa}, {view_label}"
            if file_name:
                title += f" ({file_name})"
            ax.set_title(title, fontsize=10, wrap=True)

            for cluster_id, points in clusters.items():
                points = np.array(points)
                color = cluster_colors[cluster_id]
                ax.scatter(points[:, x_col], points[:, y_col], s=1, alpha=0.5, color=color)
                if show_legend:
                    ax.plot([], [], color=color, label=f'Cluster {cluster_id:.0f}')
            if clusters and show_legend:
                ax.legend(fontsize=6, loc='upper right')
            elif clusters:
                ax.text(0.98, 0.98, f'{len(clusters)} clusters', transform=ax.transAxes,
                        fontsize=8, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / f"img_global_clusters_event{event}_{apa}.png", dpi=100)
    plt.close()


def draw_unmatched_neutrino_flash_times(neutrino_rows, output_dir, apa, level_name, filename_prefix,
                                         file_name=None):
    """
    Flash-time distribution of the neutrinos whose reco partner had an
    out-of-window flash -- category 'reco_outside_beam_window' (a genuine timing
    loss, flash near the window edge) plus 'wrong_charge_light_matching' rows
    that carry a winner_flash_time (charge-light handed the cluster a wildly
    out-of-time cosmic flash). Shows both readings in one picture.

    Level-agnostic like the writers in writeinformation.py -- pass one event's
    rows for the event-level copy, a whole file's or job's for the aggregated one.

    Two stacked panels, following draw_flashes' existing coarse/fine convention
    above, because these flash times span hundreds of us while the interesting
    near-misses sit tens of ns from the window edge -- one linear axis cannot
    show both:
      TOP    : full range, bins of BEAM_WINDOW_WIDTH_US aligned to the window edge
      BOTTOM : zoomed to [-2, 4] us, bins of FINE_BIN_WIDTH_US
    Both mark the beam window with the same gold axvspan used everywhere else in
    this module, with its numeric limits spelled out in the legend.

    Parameters:
    - neutrino_rows: list of dicts from categorize_unmatched_true_neutrinos()
    - output_dir, apa, level_name, filename_prefix, file_name: same convention as draw_flashes

    Saved as unmatched_neutrino_flash_times_{filename_prefix}_{apa}.png.
    No-op (returns without writing) if no row carries an out-of-window winner flash.
    """
    flash_times = [r['winner_flash_time'] for r in neutrino_rows
                    if r['category'] in ('reco_outside_beam_window', 'wrong_charge_light_matching')
                    and r['winner_flash_time'] is not None]
    if not flash_times:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every one of these is out of the window by construction (that is what the
    # category means), so the informative split is near-miss vs. grossly out --
    # counted here at the zoom panel's edge rather than at the window's.
    n_near = sum(1 for t in flash_times if -2 <= t <= 4)
    n_far  = len(flash_times) - n_near

    beam_window_label = (f'beam window: {BEAM_WINDOW_MIN_US} - {BEAM_WINDOW_MAX_US} us '
                         f'(width {BEAM_WINDOW_WIDTH_US:.2f} us)')

    title = f'Flash Time of Would-Have-Matched Reco (beam-window losses): {level_name}, {apa}'
    if file_name:
        title += f' ({file_name})'

    fig, (ax_coarse, ax_fine) = plt.subplots(2, 1, figsize=(11, 11))

    def _count_axis(ax, counts):
        """Counts are small integers here (often all 1), where matplotlib's
        autoscale produces a meaningless 0.96-1.05 y-range. Anchor at 0 and tick
        in whole clusters instead."""
        top = max(int(counts.max()), 1)
        ax.set_ylim(0, top + 0.5)
        ax.set_yticks(range(0, top + 1))

    # TOP -- full range. Same scatter-at-nonzero-bin-centers approach as
    # draw_flashes: with a handful of entries spread over hundreds of us, a bar
    # histogram would be almost entirely empty bins.
    coarse_edges = _aligned_bin_edges(min(flash_times), max(flash_times), BEAM_WINDOW_WIDTH_US, BEAM_WINDOW_MIN_US)
    coarse_counts, coarse_edges = np.histogram(flash_times, bins=coarse_edges)
    coarse_centers = (coarse_edges[:-1] + coarse_edges[1:]) / 2
    nonzero = coarse_counts > 0
    ax_coarse.scatter(coarse_centers[nonzero], coarse_counts[nonzero], s=60,
                      color='steelblue', edgecolor='black', alpha=0.8,
                      label=f'would-have-matched reco (n={len(flash_times)})')
    ax_coarse.axvspan(BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US, color='gold', alpha=0.25, label=beam_window_label)
    ax_coarse.set_xlabel('Flash Time [us]', fontsize=12, fontweight='bold')
    ax_coarse.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax_coarse.set_title(f'{title}\n(full range, bins of {BEAM_WINDOW_WIDTH_US:.2f} us)', fontsize=12, fontweight='bold', wrap=True)
    ax_coarse.grid(True, linestyle='--', alpha=0.3)
    _count_axis(ax_coarse, coarse_counts)
    ax_coarse.legend(fontsize=9)

    # BOTTOM -- zoomed to the same [-2, 4] us as draw_flashes' fine panel, where
    # the near-misses (tens of ns outside the edge) become readable.
    fine_edges = _aligned_bin_edges(-2, 4, FINE_BIN_WIDTH_US, BEAM_WINDOW_MIN_US)
    fine_counts, fine_edges = np.histogram(flash_times, bins=fine_edges)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2
    nonzero = fine_counts > 0
    ax_fine.scatter(fine_centers[nonzero], fine_counts[nonzero], s=60,
                    color='steelblue', edgecolor='black', alpha=0.8,
                    label=f'in view: {n_near}, outside [-2, 4] us: {n_far}')
    ax_fine.axvspan(BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US, color='gold', alpha=0.25, label=beam_window_label)
    ax_fine.set_xlim(-2, 4)
    ax_fine.set_xlabel('Flash Time [us]', fontsize=12, fontweight='bold')
    ax_fine.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax_fine.set_title(f'Zoom on the beam window (bins of {FINE_BIN_WIDTH_US*1000:.0f} ns)', fontsize=12, fontweight='bold', wrap=True)
    ax_fine.grid(True, linestyle='--', alpha=0.3)
    _count_axis(ax_fine, fine_counts)
    ax_fine.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f'unmatched_neutrino_flash_times_{filename_prefix}_{apa}.png',
                dpi=100, bbox_inches='tight', pad_inches=0.3)
    plt.close()
