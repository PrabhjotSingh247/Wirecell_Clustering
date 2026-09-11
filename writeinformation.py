"""
Functions that WRITE human-readable information files (the .txt tables that sit
next to the plots). Kept separate from metadata.py, which builds the in-memory
records: metadata.py answers "what do we know about this cluster/interaction",
this module answers "how is that written to disk for a person to read".

Every writer here is level-agnostic -- pass one event's records for an
event-level file or a whole job's for the aggregated one -- and returns the Path
it wrote, or None when there was nothing to write.
"""
from pathlib import Path

def write_true_cluster_info(cluster_type_records, output_dir, filename="true_cluster_info.txt"):
    """
    Write true_cluster_info.txt: one row per event listing how many neutrinos
    it has.

    Level-agnostic -- pass ONE event's records for the event-level copy, or a
    whole file's/job's for the aggregated copy. The format is identical at
    every level (one row per distinct 'event' key found in the records), which
    is the point of having a single writer: the event-level file is a slice of
    the job-level one, byte for byte.

    reassign_cluster_ID_true_charge_light keeps each neutrino interaction as
    its own true cluster (99990+nu_idx), so counting neutrino clusters per
    event IS counting neutrinos per event -- no need to print nu_idx_values or
    repeat a row per interaction.

    There is no beam-window column: beam-window membership is a RECO-side
    quantity only (see build_true_cluster_type_records for why it cannot be
    stated for a true cluster, and write_reco_cluster_info for the reco-side
    counts).

    Parameters:
    - cluster_type_records: List of dicts from build_true_cluster_type_records(),
        each with 'event', 'is_neutrino'
    - output_dir: Directory to write into (created if missing)
    - filename: Output file name

    Returns:
        Path written, or None if there was nothing to write
    """
    if not cluster_type_records:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    neutrino_counts_by_event = {}
    for r in cluster_type_records:
        if r['is_neutrino']:
            neutrino_counts_by_event[r['event']] = neutrino_counts_by_event.get(r['event'], 0) + 1

    with open(out_path, "w") as f:
        f.write(f"{'event':<20} {'num_neutrinos':>14}\n")
        for event_key in sorted(neutrino_counts_by_event):
            f.write(f"{event_key:<20} {neutrino_counts_by_event[event_key]:>14}\n")

        multi_neutrino_events = sorted((k, v) for k, v in neutrino_counts_by_event.items() if v > 1)
        f.write(f"\n{'='*60}\n")
        f.write(f"Events with more than one neutrino: {len(multi_neutrino_events)}\n")
        for event_key, count in multi_neutrino_events:
            f.write(f"  {event_key}: {count} neutrinos\n")

    return out_path

def write_reco_cluster_info(reco_beam_window_records, output_dir, filename="reco_cluster_info.txt"):
    """
    Write reco_cluster_info.txt: one row per event listing how many DISTINCT
    clustering-global clusters have a beam-window-matched flash.

    Level-agnostic, same as write_true_cluster_info above -- pass one event's
    record for the event-level copy, or the whole job's list for the
    aggregated one.

    This is the RECO-side proxy for "multiple neutrino-like activity in the
    beam spill" -- grouped by reco cluster + matched flash timing, NOT by true
    nu_idx (that's write_true_cluster_info's num_neutrinos column, which is
    ground truth). Beam-window membership is stated ONLY here, on the reco
    side, where the flash time is measured rather than inferred.

    Parameters:
    - reco_beam_window_records: List of dicts, each with 'file_name', 'event',
        'event_num', 'num_clusters_in_beam_window', 'cluster_ids'
    - output_dir: Directory to write into (created if missing)
    - filename: Output file name

    Returns:
        Path written, or None if there was nothing to write
    """
    if not reco_beam_window_records:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    with open(out_path, "w") as f:
        f.write(f"{'event':<20} {'num_clusters_in_beam_window':>28} {'cluster_ids':<40}\n")
        for r in sorted(reco_beam_window_records, key=lambda r: (r['file_name'], r['event_num'])):
            ids_str = ",".join(f"{c:.0f}" for c in r['cluster_ids']) if r['cluster_ids'] else "-"
            f.write(f"{r['event']:<20} {r['num_clusters_in_beam_window']:>28} {ids_str:<40}\n")

        multi_cluster_events = sorted(
            (r['event'], r['num_clusters_in_beam_window'], r['cluster_ids'])
            for r in reco_beam_window_records if r['num_clusters_in_beam_window'] > 1
        )
        f.write(f"\n{'='*60}\n")
        f.write(f"Events with 2+ clusters in beam window: {len(multi_cluster_events)}\n")
        for event_key, count, cluster_ids in multi_cluster_events:
            ids_str = ",".join(f"{c:.0f}" for c in cluster_ids)
            f.write(f"  {event_key}: {count} clusters ({ids_str})\n")

    return out_path

def write_neutrino_vertex_info(vertex_records, output_dir, filename="true_neutrino_info.txt",
                               volume_label="wire-readout sensitive box"):
    """
    Everything known about each true neutrino interaction, one row per
    interaction, from build_neutrino_vertex_records().

    cluster_energy_MeV (from the sed true points) is the energy used everywhere
    else in the pipeline; mc_Etot/mc_Edep are mc.json reference values that no
    cut uses -- the header says so, so a reader can't mistake one for the other.
    """
    if not vertex_records:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    def _fmt(value, spec):
        return format(value, spec) if value is not None else "n/a"

    with open(out_path, "w") as f:
        f.write(f"{'='*200}\n")
        f.write("TRUE NEUTRINO INTERACTIONS (one row per interaction, from mc.json interaction-vertex nodes)\n")
        f.write(f"{'='*200}\n")
        f.write("vertex_x/y/z  : interaction vertex from mc.json (cm, same frame as the true points)\n")
        f.write(f"in_volume     : vertex inside the {volume_label}\n")
        f.write("cluster_energy: TRUE cluster energy summed from the sed true points -- the energy used\n")
        f.write("                for evaluation and the energy cut everywhere in this pipeline\n")
        f.write("mc_Etot/mc_Edep: mc.json reference energies (incident neutrino total / mc-side deposited).\n")
        f.write("                NOT used for any cut or evaluation -- shown for cross-reference only\n")
        f.write("has_cluster   : whether this interaction produced a true cluster that survived the cuts\n")
        f.write("channel       : CC/NC from the interacting flavor plus the FIRST list of daughters\n")
        f.write("                (direct children of the vertex): a numu with a muon there -> numu_CC,\n")
        f.write("                a nue with an electron -> nue_CC, no muon and no electron -> NC\n")
        f.write("lepton        : the charged lepton that made it CC (blank for NC)\n")
        f.write("daughters     : that first daughter list, the evidence behind the channel\n")
        f.write("A '*' after an NC channel marks an interaction that DID have a charged lepton, but of\n")
        f.write("the other flavor -- not the partner of the neutrino that made it, so not CC. Real\n")
        f.write("physics, not a parse error: a numu event with an outgoing numu and a positron from a\n")
        f.write("photon is NC with an e+ in the daughter list.\n")
        f.write(f"{'='*200}\n\n")

        f.write(f"{'event':<12} {'nu_idx':>6} {'flavor':<8} {'channel':<9} {'lepton':<7} {'cluster_id':>11} "
                f"{'vertex_x':>10} {'vertex_y':>10} {'vertex_z':>10} {'in_volume':>10} "
                f"{'cluster_energy':>15} {'n_points':>9} {'mc_Etot':>10} {'mc_Edep':>10} {'has_cluster':>12} "
                f"  {'daughters'}\n")
        for r in sorted(vertex_records, key=lambda r: (r['file_name'], r['event_num'], r['nu_idx'] or 0)):
            channel = str(r.get('interaction_channel', 'n/a'))
            if r.get('lepton_flavor_mismatch'):
                channel += "*"
            daughters = ",".join(r.get('daughter_particles') or []) or "-"
            f.write(
                f"{r['event']:<12} {str(r['nu_idx']):>6} {str(r['flavor']):<8} {channel:<9} "
                f"{str(r.get('primary_lepton') or ''):<7} "
                f"{_fmt(r['cluster_id'], '.0f'):>11} "
                f"{_fmt(r['vertex_x'], '.2f'):>10} {_fmt(r['vertex_y'], '.2f'):>10} {_fmt(r['vertex_z'], '.2f'):>10} "
                f"{str(r['vertex_in_volume']):>10} "
                f"{_fmt(r['cluster_energy_MeV'], '.2f'):>15} {r['n_true_points']:>9} "
                f"{_fmt(r['mc_total_energy_MeV'], '.1f'):>10} {_fmt(r['mc_edep_MeV'], '.1f'):>10} "
                f"{str(r['has_true_cluster']):>12}   {daughters}\n")

        n_total    = len(vertex_records)
        n_in       = sum(1 for r in vertex_records if r['vertex_in_volume'] is True)
        n_out      = sum(1 for r in vertex_records if r['vertex_in_volume'] is False)
        n_cluster  = sum(1 for r in vertex_records if r['has_true_cluster'])
        n_in_clu   = sum(1 for r in vertex_records if r['vertex_in_volume'] is True and r['has_true_cluster'])
        n_out_clu  = sum(1 for r in vertex_records if r['vertex_in_volume'] is False and r['has_true_cluster'])

        f.write(f"\n{'='*200}\n")
        f.write(f"Total true neutrino interactions: {n_total}\n")
        f.write(f"  vertex IN  volume: {n_in:>5}   (with a true cluster: {n_in_clu})\n")
        f.write(f"  vertex OUT volume: {n_out:>5}   (with a true cluster: {n_out_clu})\n")
        f.write(f"  produced a true cluster: {n_cluster} / {n_total}\n")
        f.write("Note: an OUT-of-volume interaction can still produce a true cluster -- a daughter\n")
        f.write("particle entering the active volume deposits there even though the vertex is outside.\n")

        # CC/NC breakdown, split the same three ways the rest of this table is
        # read: all interactions, those in volume, and those that deposited a
        # surviving cluster.
        channels = sorted({str(r.get('interaction_channel', 'n/a')) for r in vertex_records})
        f.write(f"\n{'='*200}\n")
        f.write("CC / NC breakdown (channel from the first list of daughters)\n")
        f.write(f"{'channel':<10} {'total':>7} {'in volume':>11} {'out volume':>11} {'with cluster':>14}\n")
        for channel in channels:
            rows = [r for r in vertex_records if str(r.get('interaction_channel', 'n/a')) == channel]
            f.write(f"{channel:<10} {len(rows):>7} "
                    f"{sum(1 for r in rows if r['vertex_in_volume'] is True):>11} "
                    f"{sum(1 for r in rows if r['vertex_in_volume'] is False):>11} "
                    f"{sum(1 for r in rows if r['has_true_cluster']):>14}\n")
        n_cc = sum(1 for r in vertex_records if r.get('interaction_type') == 'CC')
        n_nc = sum(1 for r in vertex_records if r.get('interaction_type') == 'NC')
        f.write(f"{'CC (all)':<10} {n_cc:>7}\n")
        f.write(f"{'NC (all)':<10} {n_nc:>7}\n")
        n_mismatch = sum(1 for r in vertex_records if r.get('lepton_flavor_mismatch'))
        if n_mismatch:
            f.write(f"\nOf the NC interactions, {n_mismatch} carried a charged lepton of the OTHER "
                    f"flavor (marked '*' above):\n")
            for r in vertex_records:
                if r.get('lepton_flavor_mismatch'):
                    f.write(f"  {r['event']} nu_idx {r['nu_idx']}: {r['flavor']} interaction with "
                            f"{r.get('mismatched_lepton')} among "
                            f"{','.join(r.get('daughter_particles') or [])}\n")

    return out_path

def write_removed_neutrino_info(vertex_records, output_dir, filename="removed_true_neutrino_info.txt"):
    """
    The true neutrino interactions that did NOT survive the true selections --
    everything the plots leave out, kept so the losses can be studied rather
    than silently disappearing.

    Requires build_neutrino_vertex_records() to have been given
    clusters_true_precut/min_cluster_energy; without them precut_energy_MeV and
    removal_reason are blank and this file can only say that the interaction is
    missing, not why.

    Grouped and summarised by removal_reason, since the categories call for
    different responses: "no true deposits" is a genuinely invisible interaction
    (nothing to recover), while "below energy cut" is a threshold choice and
    "removed by geometric cuts" is a fiducial/min-points effect.

    The dead-area cut is not among those geometric cuts: it runs upstream, in
    preprocess_deadarea_cut.py, so an interaction whose only deposits fell inside a
    dead channel region arrives here already empty and is reported as "no true
    deposits" -- which is what it is, since those points could never have been
    reconstructed. See build_neutrino_vertex_records' docstring.
    """
    removed = [r for r in vertex_records if not r.get('has_true_cluster')]
    if not removed:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    def _fmt(value, spec):
        return format(value, spec) if value is not None else "n/a"

    n_total = len(vertex_records)
    with open(out_path, "w") as f:
        f.write(f"{'='*140}\n")
        f.write("TRUE NEUTRINO INTERACTIONS REMOVED BY THE TRUE SELECTIONS\n")
        f.write(f"{'='*140}\n")
        f.write("These interactions exist in mc.json but have no true cluster left after the cuts, so they\n")
        f.write("are EXCLUDED from every true-neutrino plot (vertices, vertex-volume, flavor). Listed here\n")
        f.write("so the losses stay visible and can be revisited.\n\n")
        f.write("precut_energy / precut_points: the cluster BEFORE any cut -- what the interaction actually\n")
        f.write("                deposited in the active volume, summed from the sed true points\n")
        f.write("removal_reason: which stage is responsible (see below)\n")
        f.write("mc_Etot/mc_Edep: mc.json reference energies, not used by any cut\n")
        f.write(f"{'='*140}\n\n")

        f.write(f"{'event':<12} {'nu_idx':>6} {'flavor':<8} {'in_volume':>10} "
                f"{'vertex_x':>10} {'vertex_y':>10} {'vertex_z':>10} "
                f"{'precut_energy':>14} {'precut_points':>14} {'mc_Etot':>9} {'mc_Edep':>9}  removal_reason\n")
        for r in sorted(removed, key=lambda r: (r['file_name'], r['event_num'], r['nu_idx'] or 0)):
            f.write(
                f"{r['event']:<12} {str(r['nu_idx']):>6} {str(r['flavor']):<8} {str(r['vertex_in_volume']):>10} "
                f"{_fmt(r['vertex_x'], '.2f'):>10} {_fmt(r['vertex_y'], '.2f'):>10} {_fmt(r['vertex_z'], '.2f'):>10} "
                f"{_fmt(r.get('precut_energy_MeV'), '.2f'):>14} {r.get('precut_n_points', 0):>14} "
                f"{_fmt(r['mc_total_energy_MeV'], '.1f'):>9} {_fmt(r['mc_edep_MeV'], '.1f'):>9}"
                f"  {r.get('removal_reason') or 'n/a'}\n")

        by_category = {}
        for r in removed:
            by_category.setdefault(r.get('removal_category') or 'unknown', []).append(r)

        n_surviving = n_total - len(removed)
        f.write(f"\n{'='*140}\n")
        f.write("REMOVAL SUMMARY BY REASON\n")
        f.write(f"{'='*140}\n")
        f.write(f"{'reason':<28} {'removed':>9} {'% of removed':>13} {'% of all':>10} "
                f"{'vertex_in':>10} {'vertex_out':>11}\n")
        f.write(f"{'-'*140}\n")
        for category, group in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
            n_in = sum(1 for r in group if r['vertex_in_volume'] is True)
            f.write(f"{category:<28} {len(group):>9} "
                    f"{100.0 * len(group) / len(removed):>12.1f}% {100.0 * len(group) / n_total:>9.1f}% "
                    f"{n_in:>10} {len(group) - n_in:>11}\n")
        f.write(f"{'-'*140}\n")
        n_in_all = sum(1 for r in removed if r['vertex_in_volume'] is True)
        f.write(f"{'TOTAL REMOVED':<28} {len(removed):>9} {100.0:>12.1f}% "
                f"{100.0 * len(removed) / n_total:>9.1f}% {n_in_all:>10} {len(removed) - n_in_all:>11}\n\n")
        f.write(f"True neutrino interactions in mc.json: {n_total}\n")
        f.write(f"  removed by the true selections:     {len(removed)}\n")
        f.write(f"  surviving (and therefore plotted):  {n_surviving}\n\n")
        f.write("Cross-check against the cluster-level counts: interactions minus 'no true deposits'\n")
        f.write("is the neutrino cluster count BEFORE any cut, and subtracting 'below energy cut'\n")
        f.write("leaves the count after it -- the same two numbers SelectionAnalysis reports.\n")

        # What lowering the threshold would recover -- the actionable half of
        # "understand what we can do for those neutrinos".
        energy_removed = [r for r in removed
                          if r.get('removal_category') == 'below energy cut'
                          and r.get('precut_energy_MeV') is not None]
        if energy_removed:
            energies = sorted(r['precut_energy_MeV'] for r in energy_removed)
            f.write(f"\n{'='*140}\n")
            f.write("DEPOSITED ENERGY OF THE INTERACTIONS LOST TO THE ENERGY CUT\n")
            f.write("(how many would come back if the threshold moved -- deposited energy from the sed points)\n")
            f.write(f"{'='*140}\n")
            f.write(f"{'energy range [MeV]':<22} {'count':>7} {'cumulative':>12}   (cumulative = recovered if the cut moved to the upper edge)\n")
            edges = [0, 10, 25, 50, 75, 100]
            cumulative = 0
            for low, high in zip(edges[:-1], edges[1:]):
                n = sum(1 for e in energies if low <= e < high)
                cumulative += n
                f.write(f"{f'{low} - {high}':<22} {n:>7} {cumulative:>12}\n")
            f.write(f"\nmin {energies[0]:.1f} MeV, median {energies[len(energies)//2]:.1f} MeV, "
                    f"max {energies[-1]:.1f} MeV, over {len(energies)} interactions\n")

    return out_path


def _split_names(output_dir, filename):
    """
    (combined, neutrino-only, cosmic-only) paths for a table written per cluster type:
    completeness_info.txt -> completeness_info.txt, completeness_info_neutrino.txt,
    completeness_info_cosmic.txt, in that directory.

    Derived from the filename's stem rather than hard-coded so a caller passing a custom
    filename still gets a matching pair of split tables next to it.
    """
    stem, suffix = Path(filename).stem, Path(filename).suffix
    return (output_dir / filename,
            output_dir / f"{stem}_neutrino{suffix}",
            output_dir / f"{stem}_cosmic{suffix}")


def write_completeness_info(metadata_list, output_dir, filename="completeness_info.txt"):
    """
    Write the imaginglevel completeness_info.txt: one row per TRUE cluster, with its
    completeness summed over all its reco matches -- the same grouping
    DrawCompletenessVsTrueEnergyPerEvent/PerFile/PerJob plot, so the numbers behind
    those plots can be read cluster by cluster instead of off a plot.

    Level-agnostic -- pass one event's metadata for the event-level copy, a file's or
    the job's for the aggregated copy. The 'event' column is what makes the aggregated
    copies traceable back to an event, and is present at every level so the formats stay
    identical (the event-level file is a slice of the job-level one, byte for byte).

    num_reco_matches is 0 for a true cluster that matched nothing: add_metadata_true_clusters
    excludes EvaluateCompleteness's 8888 "no match" sentinel row from the count, so an
    unmatched cluster reads completeness 0.0000 / 0 matches rather than 0.0000 / 1.

    Parameters:
    - metadata_list: List of dicts from add_metadata_true_clusters(), each with 'event',
        'true_cluster_id', 'cluster_type', 'cluster_category', 'total_true_energy',
        'total_completeness', 'num_reco_matches'
    - output_dir: Directory to write into (created if missing)
    - filename: Output file name for the combined table; the neutrino-only and
        cosmic-only tables take its stem plus _neutrino / _cosmic

    Three files are written side by side -- the combined table, plus one per cluster
    type (see _split_names), since the two populations are read for different questions:
    the neutrino rows are the physics signal, the cosmic rows the background the
    reconstruction also has to get right.

    All are written even when there is nothing to report -- a header-only table says "no
    clusters of this type here", whereas a missing file is indistinguishable from a
    crashed job.

    Returns:
        Path of the combined table
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write(path, entries):
        with open(path, "w") as f:
            f.write(f"{'event':<20} {'true_cluster_id':>16} {'cluster_type':<10} {'cluster_category':<16} "
                    f"{'total_true_energy_MeV':>22} {'total_completeness':>16} {'num_reco_matches':>16}\n")
            # Grouped by event, then energy-ranked inside it -- the event-level file has
            # one event so this is exactly its old ordering.
            for m in sorted(entries, key=lambda m: (str(m['event']), -m['total_true_energy'])):
                f.write(f"{str(m['event']):<20} {m['true_cluster_id']:>16.3f} {m['cluster_type']:<10} {m['cluster_category']:<16} "
                        f"{m['total_true_energy']:>22.3f} {m['total_completeness']:>16.4f} {m['num_reco_matches']:>16}\n")

    all_path, neutrino_path, cosmic_path = _split_names(output_dir, filename)
    _write(all_path, metadata_list)
    _write(neutrino_path, [m for m in metadata_list if m['cluster_type'] == 'neutrino'])
    _write(cosmic_path, [m for m in metadata_list if m['cluster_type'] == 'cosmic'])

    return all_path


def write_pair_completeness_info(pair_metadata_list, output_dir, all_true_metadata_list=None,
                               filename="completeness_info.txt"):
    """
    Write the clusteringlevel completeness_info.txt: one row per 1-to-1 matched true-reco
    pair, carrying completeness AND purity together -- the population
    DrawClusterCompletenessVsTrueEnergy* / DrawCompletenessVsTrueEnergy_MatchedPairs_* plot.

    all_true_metadata_list controls which of the two clusteringlevel directories this is:
      - given (add_metadata_true_clusters output): true clusters that matched no reco
        cluster are appended with reco_cluster_id N/A, completeness 0 and matched=no --
        the "including unmatched true clusters" table, matching the plots that take
        all_true_metadata_list.
      - None: matched pairs only, the "..._true_reco_pairs_only" table.

    Level-agnostic, same as write_completeness_info -- see its note on the 'event' column.

    Parameters:
    - pair_metadata_list: List of dicts from add_metadata_true_reco_pair_cluster()
    - output_dir: Directory to write into (created if missing)
    - all_true_metadata_list: Optional list from add_metadata_true_clusters(); when given,
        its true clusters absent from pair_metadata_list are written as unmatched rows
    - filename: Output file name for the combined table; the neutrino-only and
        cosmic-only tables take its stem plus _neutrino / _cosmic

    Writes the combined table plus one per cluster type, same as write_completeness_info.

    Returns:
        Path of the combined table
    """
    entries = list(pair_metadata_list)

    if all_true_metadata_list:
        # Matched on (event, true_cluster_id) rather than the id alone: the same
        # true_cluster_id (a rounded average X) recurs across events.
        matched_keys = {(m['event'], m['true_cluster_id']) for m in pair_metadata_list}
        entries += [
            {'event': m['event'], 'true_cluster_id': m['true_cluster_id'], 'reco_cluster_id': None,
             'cluster_type': m['cluster_type'], 'cluster_category': m['cluster_category'],
             'total_true_energy': m['total_true_energy'], 'completeness': 0.0,
             'purity': None, 'total_reco_charge': None}
            for m in all_true_metadata_list
            if (m['event'], m['true_cluster_id']) not in matched_keys
        ]

    # Written even when empty -- see write_completeness_info's note.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write(path, rows):
        with open(path, "w") as f:
            f.write(f"{'event':<20} {'true_cluster_id':>16} {'reco_cluster_id':>16} {'cluster_type':<10} {'cluster_category':<16} "
                    f"{'total_true_energy_MeV':>22} {'completeness':>10} {'purity':>10} {'total_reco_charge':>18} {'matched':>8}\n")
            for m in sorted(rows, key=lambda m: (str(m['event']), -m['total_true_energy'])):
                reco_id_str = f"{m['reco_cluster_id']:.3f}" if m['reco_cluster_id'] is not None else "N/A"
                purity_str  = f"{m['purity']:.4f}" if m['purity'] is not None else "N/A"
                charge_str  = f"{m['total_reco_charge']:.1f}" if m['total_reco_charge'] is not None else "N/A"
                matched_str = "yes" if m['reco_cluster_id'] is not None else "no"
                f.write(f"{str(m['event']):<20} {m['true_cluster_id']:>16.3f} {reco_id_str:>16} {m['cluster_type']:<10} {m['cluster_category']:<16} "
                        f"{m['total_true_energy']:>22.3f} {m['completeness']:>10.4f} {purity_str:>10} {charge_str:>18} {matched_str:>8}\n")

    all_path, neutrino_path, cosmic_path = _split_names(output_dir, filename)
    _write(all_path, entries)
    _write(neutrino_path, [m for m in entries if m['cluster_type'] == 'neutrino'])
    _write(cosmic_path, [m for m in entries if m['cluster_type'] == 'cosmic'])

    return all_path


def write_purity_info(purity_results, output_dir, filename="purity_info.txt"):
    """
    Write purity_info.txt: raw EvaluatePurity output, one row per RECO cluster -- matched
    or not. Unmatched reco clusters carry the true_cluster_id=8888 / purity=-0.1 sentinel
    and are reported as matched=no; they have no matched_reco_points/total_reco_points, so
    those columns read N/A for them.

    Level-agnostic, same as write_completeness_info -- see its note on the 'event' column.

    Parameters:
    - purity_results: List of dicts from EvaluatePurity(), each with 'event',
        'reco_cluster_id', 'true_cluster_id', 'purity', 'total_reco_cluster_charge'
    - output_dir: Directory to write into (created if missing)
    - filename: Output file name

    Written even when empty -- see write_completeness_info's note.

    Returns:
        Path written
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    with open(out_path, "w") as f:
        f.write(f"{'event':<20} {'reco_cluster_id':>16} {'true_cluster_id':>16} {'purity':>10} {'total_reco_charge':>18} "
                f"{'matched_reco_points':>20} {'total_reco_points':>18} {'matched':>8}\n")
        for p in sorted(purity_results, key=lambda p: (str(p['event']), -p['total_reco_cluster_charge'])):
            matched = p['true_cluster_id'] != 8888
            mrp = p.get('matched_reco_points', 'N/A')
            trp = p.get('total_reco_points', 'N/A')
            f.write(f"{str(p['event']):<20} {p['reco_cluster_id']:>16.3f} {p['true_cluster_id']:>16.3f} {p['purity']:>10.4f} "
                    f"{p['total_reco_cluster_charge']:>18.1f} {str(mrp):>20} {str(trp):>18} {('yes' if matched else 'no'):>8}\n")

    return out_path


_EXTRA_RECO_CATEGORY_ORDER = ['fragment_of_neutrino', 'matched_cosmic_only', 'no_true_overlap']


def write_extra_reco_info(categorized_rows, output_dir, filename="extra_reco_info.txt"):
    """
    Write extra_reco_info.txt: one row per "extra" reco cluster -- every row from
    metadata.categorize_extra_reco_clusters() EXCEPT category=='matched_winner'
    (the reco cluster that IS the 1-to-1 match of a true neutrino cluster, i.e.
    not "extra"). See categorize_extra_reco_clusters' docstring for what each
    category means.

    Level-agnostic, same as the other writers in this module -- pass one event's
    rows for the event-level copy, or a whole file's/job's for the aggregated one.

    Rows are grouped by category (fragment_of_neutrino, matched_cosmic_only,
    no_true_overlap, in that order) and sorted by ascending min_dist within
    no_true_overlap -- the near-miss candidates (small min_dist, especially a
    small dx = drift/X direction, a possible charge-light X-mis-assignment) sort
    to the top of that block.

    Parameters:
    - categorized_rows: List of dicts from metadata.categorize_extra_reco_clusters()
    - output_dir: Directory to write into (created if missing)
    - filename: Output file name

    Returns:
        Path written, or None if there was nothing to write
    """
    extra_rows = [r for r in categorized_rows if r['category'] != 'matched_winner']
    if not extra_rows:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    def _fmt(value, spec='.2f'):
        return format(value, spec) if value is not None else "n/a"

    def _sort_key(r):
        cat_rank = _EXTRA_RECO_CATEGORY_ORDER.index(r['category'])
        return (cat_rank, r['min_dist'] if r['min_dist'] is not None else -1)

    with open(out_path, "w") as f:
        f.write(f"{'='*170}\n")
        f.write("EXTRA (NON-WINNER) RECO CLUSTERS\n")
        f.write(f"{'='*170}\n")
        f.write("fragment_of_neutrino : overlaps a true neutrino cluster (purity>0) but a different reco\n")
        f.write("                       cluster won that true cluster's 1-to-1 match (MatchTrueToReco1to1)\n")
        f.write("matched_cosmic_only  : overlaps only cosmic true cluster(s), never a neutrino\n")
        f.write("no_true_overlap      : zero spatial overlap with any true cluster (EvaluatePurity's 8888\n")
        f.write("                       sentinel); nearest_true_* / min_dist / dx,dy,dz come from a KDTree\n")
        f.write("                       search against every true cluster in the event -- small min_dist\n")
        f.write("                       (esp. small dx) is an X-mis-assignment candidate, large min_dist\n")
        f.write("                       suggests a genuinely spurious/noise cluster\n")
        f.write(f"{'='*170}\n\n")

        f.write(f"{'file':<8} {'event':<14} {'reco_id':>12} {'category':<22} {'charge':>12} "
                f"{'matched_true_id':>16} {'purity':>8} {'n_true_matches':>15} "
                f"{'nearest_true_id':>16} {'nearest_is_nu':>14} {'min_dist_cm':>12} {'mean_nn_cm':>11} "
                f"{'dx_cm':>8} {'dy_cm':>8} {'dz_cm':>8}\n")
        for r in sorted(extra_rows, key=_sort_key):
            f.write(f"{r['file_name']:<8} {str(r['event']):<14} {r['reco_cluster_id']:>12.3f} {r['category']:<22} "
                    f"{r['total_reco_charge']:>12.1f} {_fmt(r['matched_true_cluster_id'], '.0f'):>16} "
                    f"{_fmt(r['purity'], '.4f'):>8} {r['num_true_matches']:>15} "
                    f"{_fmt(r['nearest_true_cluster_id'], '.0f'):>16} {str(r['nearest_true_is_neutrino']):>14} "
                    f"{_fmt(r['min_dist']):>12} {_fmt(r['mean_nn_dist']):>11} "
                    f"{_fmt(r['dx']):>8} {_fmt(r['dy']):>8} {_fmt(r['dz']):>8}\n")

        f.write(f"\n{'='*170}\n")
        f.write("SUMMARY BY CATEGORY\n")
        f.write(f"{'='*170}\n")
        n_winner = sum(1 for r in categorized_rows if r['category'] == 'matched_winner')
        f.write(f"{'matched_winner (not extra)':<30} {n_winner:>8}\n")
        for cat in _EXTRA_RECO_CATEGORY_ORDER:
            n = sum(1 for r in extra_rows if r['category'] == cat)
            f.write(f"{cat:<30} {n:>8}\n")
        f.write(f"{'-'*40}\n")
        f.write(f"{'TOTAL reco clusters':<30} {len(categorized_rows):>8}\n")

    return out_path


_UNMATCHED_TRUE_NU_CATEGORY_ORDER = ['wrong_charge_light_matching', 'removed_by_cosmic_tagger',
                                      'reco_outside_beam_window',
                                      'broken_or_sparse_reco',
                                      'no_reco_overlap', 'unexplained']


def write_unmatched_true_neutrino_info(neutrino_rows, output_dir, filename="unmatched_true_neutrino_info.txt"):
    """
    Write unmatched_true_neutrino_info.txt: one row per true neutrino cluster that
    found NO 1-to-1 reco match -- every row from
    metadata.categorize_unmatched_true_neutrinos() EXCEPT category=='matched'.
    See that function's docstring for what each category means.

    The mirror of write_extra_reco_info above, and level-agnostic in the same way
    as every other writer here: pass one event's rows for the event-level copy, or
    a whole file's/job's for the aggregated one.

    Rows are grouped by category in _UNMATCHED_TRUE_NU_CATEGORY_ORDER. Within
    reco_outside_beam_window and wrong_charge_light_matching they sort by
    |flash offset| so the near misses come before the gross ones; within
    no_reco_overlap they sort by ascending min_dist.

    Parameters:
    - neutrino_rows: List of dicts from metadata.categorize_unmatched_true_neutrinos()
    - output_dir: Directory to write into (created if missing)
    - filename: Output file name

    Returns:
        Path written, or None if there was nothing to write
    """
    unmatched_rows = [r for r in neutrino_rows if r['category'] != 'matched']
    if not unmatched_rows:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    def _fmt(value, spec='.2f'):
        return format(value, spec) if value is not None else "n/a"

    def _sort_key(r):
        cat_rank = _UNMATCHED_TRUE_NU_CATEGORY_ORDER.index(r['category']) \
            if r['category'] in _UNMATCHED_TRUE_NU_CATEGORY_ORDER else len(_UNMATCHED_TRUE_NU_CATEGORY_ORDER)
        if r['category'] in ('reco_outside_beam_window', 'wrong_charge_light_matching'):
            return (cat_rank, abs(r['winner_flash_offset_us']) if r['winner_flash_offset_us'] is not None else 1e9)
        return (cat_rank, r['min_dist'] if r['min_dist'] is not None else -1)

    with open(out_path, "w") as f:
        f.write(f"{'='*205}\n")
        f.write("TRUE NEUTRINO CLUSTERS WITH NO RECO MATCH\n")
        f.write(f"{'='*205}\n")
        f.write("NOTE: img_ovl is measured RELAXED (img-global is sparser than clustering-global); strict_ovl uses\n")
        f.write("      the same >5-neighbour bar as MatchTrueToReco1to1. A category claiming a reconstruction was\n")
        f.write("      lost needs img_ovl (relaxed) AND img_pur above 10% (metadata.IMG_MATCH_MIN_*).\n")
        f.write("img_ovl / img_pur           : best RELAXED overlap (and its purity) of this neutrino with the\n")
        f.write("                             IMG-GLOBAL reco -- BEFORE charge-light matching set the drift coordinate.\n")
        f.write("                             img_match = both above 10%. img_match + a clustering cluster that lines up\n")
        f.write("                             in YZ but not 3D = a charge-light X-shift; img_match + NOTHING lining up =\n")
        f.write("                             the cluster was lost at the clustering stage, not shifted (-> no_reco_overlap).\n")
        f.write("wrong_charge_light_matching : the neutrino WAS imaged (img_match) AND a clustering reco cluster of it\n")
        f.write("                             exists, but charge-light matching set its drift (X) coordinate wrong.\n")
        f.write("                             cl_reason: 'no flash attached' / 'wrong flash (out of window)' (a cluster\n")
        f.write("                             with partial 3D overlap, dropped by the beam-window ID filter -- flash_off_us\n")
        f.write("                             is how far outside), or 'drift shift (YZ aligns, X off)' (NO 3D overlap left,\n")
        f.write("                             but a cluster still matches in YZ -- yz_ovl/yz_recofrac clear 10%, yz_dx is\n")
        f.write("                             the drift offset). Merges the old reco_no_flash_match, no_reco_overlap_x_shift\n")
        f.write("                             and the wrong-flash rows of reco_outside_beam_window.\n")
        f.write("removed_by_cosmic_tagger    : a reco cluster overlaps this neutrino well enough to have matched it AND\n")
        f.write("                             its flash is INSIDE the beam window, but the cosmic tagger cut removed it.\n")
        f.write("reco_outside_beam_window    : defensive residual -- clustering reconstructed something imaging did not\n")
        f.write("                             (img_match False), flash outside the window. Almost always empty.\n")
        f.write("broken_or_sparse_reco      : reco points DO sit on this neutrino (relaxed_ovl>0) but no single reco\n")
        f.write("                             cluster is dense enough to reach completeness>0 -- fragmented/scattered.\n")
        f.write("no_reco_overlap            : no 3D clustering overlap AND no YZ alignment -- there is no clustering reco\n")
        f.write("                             cluster for this neutrino at all. img_ovl/img_pur say whether IMAGING had it\n")
        f.write("                             (imaged, lost at the clustering/charge-light stage) or not (never anywhere).\n")
        f.write("unexplained                : should never appear -- a SELECTED reco cluster overlaps well enough to\n")
        f.write("                             match yet no pair formed; means this script and the notebook have drifted\n")
        f.write("\n")
        f.write("channel: numu_CC / nue_CC / NC from the interacting flavor plus the first list of daughters\n")
        f.write("        (metadata.classify_neutrino_interaction). The in-volume neutrinos are also written\n")
        f.write("        split by it, under by_vertex_volume/in_volume/by_interaction_channel/\n")
        f.write("in_vol: whether this interaction's mc.json VERTEX sits inside the wire-readout sensitive box.\n")
        f.write("        An out-of-volume interaction only ever deposits the part of itself that leaked into the\n")
        f.write("        active volume, so a failure category reads differently there -- which is why the plots\n")
        f.write("        are also written split, under by_vertex_volume/{in_volume,out_volume}/\n")
        f.write("linearity: PCA lambda1/sum(lambda) of the TRUE cluster (1.0 = a clean line, low = scattered/multi-prong)\n")
        f.write("strict_ovl / relaxed_ovl: best energy-weighted overlap over the pre-cut reco set, with / without the\n")
        f.write("                          min-reco-points-per-true-point neighbor requirement\n")
        f.write("ovl_reco_id: the pre-cut reco cluster behind that best overlap -- the one that WOULD have matched\n")
        f.write("             (strict_ovl>0), else the one carrying the most of this neutrino's charge. Same reco\n")
        f.write("             cluster IDs as the event plots and the extra-reco investigation, so it cross-references\n")
        f.write(f"{'='*205}\n\n")

        f.write(f"{'file':<8} {'event':<14} {'true_id':>9} {'in_vol':>7} {'channel':<9} {'category':<27} {'n_pts':>7} {'energy_MeV':>11} "
                f"{'linearity':>10} {'ext_x':>7} {'ext_y':>7} {'ext_z':>7} "
                f"{'strict_ovl':>11} {'relaxed_ovl':>12} {'img_ovl':>9} {'img_pur':>9} {'ovl_reco_id':>12} {'n_ovl_reco':>11} {'n_ovl_inbeam':>13} "
                f"{'flash_us':>9} {'flash_off_us':>13} {'nearest_reco':>13} {'min_dist_cm':>12} {'mean_nn_cm':>11} "
                f"{'dx_cm':>8} {'dy_cm':>8} {'dz_cm':>8} {'yz_reco_id':>11} {'yz_ovl':>8} {'yz_recofrac':>12} {'yz_dx_cm':>9} {'cl_reason':<28}\n")
        for r in sorted(unmatched_rows, key=_sort_key):
            ovl_reco_id = r['best_strict_reco_cluster_id'] if r['best_strict_overlap'] > 0 \
                else r['best_relaxed_reco_cluster_id']
            f.write(f"{r['file_name']:<8} {str(r['event']):<14} {r['true_cluster_id']:>9.0f} "
                    f"{str(r.get('vertex_in_volume', 'n/a')):>7} {str(r.get('interaction_channel') or 'n/a'):<9} "
                    f"{r['category']:<27} "
                    f"{r['n_true_points']:>7} {r['total_true_energy']:>11.1f} {r['linearity']:>10.4f} "
                    f"{r['extent_x']:>7.1f} {r['extent_y']:>7.1f} {r['extent_z']:>7.1f} "
                    f"{r['best_strict_overlap']:>11.4f} {r['best_relaxed_overlap']:>12.4f} "
                    f"{r.get('img_best_completeness', 0.0):>9.4f} {_fmt(r.get('img_best_purity'), '.4f'):>9} "
                    f"{_fmt(ovl_reco_id, '.3f'):>12} "
                    f"{r['n_overlapping_reco_clusters']:>11} {r['n_overlapping_in_beam_window']:>13} "
                    f"{_fmt(r['winner_flash_time'], '.4f'):>9} {_fmt(r['winner_flash_offset_us'], '.4f'):>13} "
                    f"{_fmt(r['nearest_reco_cluster_id'], '.3f'):>13} "
                    f"{_fmt(r['min_dist']):>12} {_fmt(r['mean_nn_dist']):>11} "
                    f"{_fmt(r['dx']):>8} {_fmt(r['dy']):>8} {_fmt(r['dz']):>8} "
                    f"{_fmt(r.get('yz_best_reco_cluster_id'), '.3f'):>11} "
                    f"{_fmt(r.get('yz_overlap'), '.4f'):>8} {_fmt(r.get('yz_reco_frac'), '.4f'):>12} "
                    f"{_fmt(r.get('yz_dx')):>9} {str(r.get('charge_light_reason') or ''):<28}\n")

        f.write(f"\n{'='*205}\n")
        f.write("SUMMARY BY CATEGORY\n")
        f.write(f"{'='*205}\n")
        # Split by vertex volume beside the total: the same table the by_vertex_volume/
        # copies show one column of, kept here so the whole picture is readable in
        # one file too. 'n/a' rows (no vertex flag) fall in neither column.
        def _n(rows, predicate=None):
            return sum(1 for r in rows if predicate is None or predicate(r))

        in_vol  = lambda r: r.get('vertex_in_volume') is True
        out_vol = lambda r: r.get('vertex_in_volume') is False

        f.write(f"{'category':<34} {'all':>8} {'in volume':>11} {'out volume':>12}\n")
        matched_rows = [r for r in neutrino_rows if r['category'] == 'matched']
        f.write(f"{'matched (found a reco pair)':<34} {_n(matched_rows):>8} "
                f"{_n(matched_rows, in_vol):>11} {_n(matched_rows, out_vol):>12}\n")
        for cat in _UNMATCHED_TRUE_NU_CATEGORY_ORDER:
            cat_rows = [r for r in unmatched_rows if r['category'] == cat]
            f.write(f"{cat:<34} {_n(cat_rows):>8} "
                    f"{_n(cat_rows, in_vol):>11} {_n(cat_rows, out_vol):>12}\n")
        f.write(f"{'-'*67}\n")
        f.write(f"{'TOTAL true neutrino clusters':<34} {len(neutrino_rows):>8} "
                f"{_n(neutrino_rows, in_vol):>11} {_n(neutrino_rows, out_vol):>12}\n")

    return out_path
