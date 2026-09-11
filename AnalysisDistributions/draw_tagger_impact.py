"""
WHAT THE COSMIC TAGGER CUT DID TO THE NEUTRINOS -- driven by

    AnalysisDistributions/Draw_InVolumeSignal_Removed_Before_CosmicTagger.ipynb
    AnalysisDistributions/Draw_InVolumeSignal_Removed_After_CosmicTagger.ipynb
    AnalysisDistributions/Draw_InVolumeSignal_Survived_After_CosmicTagger.ipynb
    AnalysisDistributions/Draw_OutOfVolumeNeutrinos_Survived_After_CosmicTagger.ipynb
    AnalysisDistributions/Draw_OutOfVolumeNeutrinos_Removed_After_CosmicTagger.ipynb

FIVE populations. Four of them are about reco clusters that SURVIVED the
beam-window cut; the fifth asks what that cut itself already threw away, and so
is matched against the PRE-cut reco set:

    BEAM-WINDOW REMOVED  matched an IN-volume neutrino, and the BEAM-WINDOW cut
                         removed it before the tagger ever ran -- signal lost one
                         stage earlier, and invisible to the other four

    IN-VOLUME REMOVED    matched an IN-volume neutrino, and the tagger cut
                         REMOVED it -- signal the selection lost
    IN-VOLUME SURVIVED   matched an IN-volume neutrino, and the tagger cut
                         KEPT it -- signal the tagger correctly left alone
    OUT-OF-VOL SURVIVED  matched an OUT-of-volume neutrino, and the tagger cut
                         KEPT it -- background the selection still carries
    OUT-OF-VOL REMOVED   matched an OUT-of-volume neutrino, and the tagger cut
                         REMOVED it -- background the tagger correctly rejected

They are the four corners of the (in volume?, removed?) grid, and they are meant
to be read together: (in, removed) is the cost and (out, kept) the leftover,
while (out, removed) and (in, kept) are the tagger working as intended. The four
counts together give the rejection and retention rates -- no one of them means
much on its own.

THE FIGURES put the TRUE cluster on the TOP row and the RECO cluster BELOW it,
the same order Contamination_Clusters and the completeness/purity populations
use. The two rows share axes per column, so what the cut took is legible as the
part of the lower row that has no counterpart above it. A BLACK STAR on the true
row's three panels is the interaction vertex from mc.json, which is what makes an
out-of-volume interaction readable as one: the vertex sits outside the active
volume while its deposits are inside it, and only the marker shows that.

BEE LINKS ARE PER CATEGORY. One file per interaction channel (numu_CC, nue_CC,
NC), written one directory above the chunk directories, so a channel can be
worked through on its own without filtering a combined list. A summary beside
them carries the counts.

WHY A MATCH MEANS WHAT IT MEANS. The pairing is match_reco_to_true_neutrino with
the pipeline's own radii and MIN_MATCH_PURITY, run on the PRE-cut beam-window
clusters -- the removed ones are, by construction, absent from the post-cut set,
so pairing after the cut would find nothing to report. The bar is loose (5%
purity), so a "matched" cluster can hold a fragment rather than a whole
interaction; the per-figure completeness and purity are what separate the two,
and the summary reports the completeness spread for exactly that reason.
"""

from pathlib import Path

import numpy as np

from draw_saved_clusters import _draw_row_panels, _TRUE_STYLE, _RECO_STYLE
from draw_contamination_clusters import _id_text, bee_event_url, split_event_key


IN_VOLUME_SIGNAL_REMOVED_DIR_NAME = 'InVolumeSignalRemoved'
IN_VOLUME_SIGNAL_SURVIVED_DIR_NAME = 'InVolumeSignalSurvived'
# The stage BEFORE the tagger: signal the BEAM-WINDOW cut itself removed.
IN_VOLUME_SIGNAL_REMOVED_BEFORE_TAGGER_DIR_NAME = 'InVolumeSignalRemovedBeforeCosmicTagger'
OUT_OF_VOLUME_SURVIVED_DIR_NAME = 'OutOfVolumeNeutrinosSurvived'
OUT_OF_VOLUME_REMOVED_DIR_NAME = 'OutOfVolumeNeutrinosRemoved'

# The interaction channels, in a fixed order so every summary and every listing
# reads the same way regardless of what the sample happened to contain.
CHANNELS = ('numu_CC', 'nue_CC', 'NC')


def neutrino_vertex_lookup(vertex_records):
    """
    {true cluster id: {'channel', 'in_volume', 'precut_energy_MeV'}} from this
    event's build_neutrino_vertex_records output.

    Vertices with no true cluster are skipped: they cannot be matched to a reco
    cluster, so they cannot appear in either population.
    """
    lookup = {}
    for record in vertex_records or []:
        cluster_id = record.get('cluster_id')
        if cluster_id is None:
            continue
        # xyz is the interaction vertex from mc.json's root node, in the same cm
        # frame as the true points -- None when mc.json carried no start_xyz, in
        # which case the true panels simply get no marker.
        vx, vy, vz = (record.get('vertex_x'), record.get('vertex_y'),
                      record.get('vertex_z'))
        lookup[cluster_id] = {
            'channel':           record.get('interaction_channel'),
            'in_volume':         record.get('vertex_in_volume'),
            'precut_energy_MeV': record.get('precut_energy_MeV'),
            'xyz':               None if None in (vx, vy, vz) else (vx, vy, vz),
        }
    return lookup


def draw_reco_true_pair_views(reco_points, true_points, output_root, event_key,
                              cluster_id, true_cluster_id, legend_lines,
                              dir_name, title, name_prefix, channel,
                              bee_url=None, vertex_xyz=None):
    """
    One reco-true pair with the TRUE cluster on the TOP row and the RECO cluster
    BELOW, matching every other pair population in this codebase.

    vertex_xyz, when given, marks the true interaction vertex on the true row's
    XZ, YZ and XY panels with a black star. It goes on the TRUE row only: it is a
    truth-level quantity, and putting it under the reco cluster as well would
    read as something the reconstruction produced.

    Written to <dir_name>/<chunk>/<channel>/, so each chunk separates its
    interaction channels: the channels are looked at independently -- there are
    six nue CC in the whole sample against hundreds of numu CC -- and a flat chunk
    directory buries the rare one among the common.

    Returns the path written, or None when either cloud is empty.
    """
    if reco_points is None or not len(reco_points):
        return None
    if true_points is None or not len(true_points):
        return None

    chunk, event = split_event_key(event_key)
    name = (f"{name_prefix}_{chunk}_event{event}"
            f"_recoID{_id_text(cluster_id)}_trueID{_id_text(true_cluster_id)}.png")
    footer_note = ('bee-display', bee_url) if bee_url else None
    return _draw_row_panels(
        title,
        [("TRUE cluster", [(true_points, _TRUE_STYLE)], vertex_xyz),
         ("RECO cluster", [(reco_points, _RECO_STYLE)])],
        (Path(output_root) / dir_name / (chunk or 'unknown_chunk')
         / (channel or 'unknown_channel') / name),
        legend_lines,
        footer_note=footer_note)


def _entry(event_key, chunk, event, cluster_id, match, vertex, true_points,
           reco_energy, tagged_detail, path, bee_url):
    """One row for the listings and the summary."""
    return {
        'path':            path,
        'event_key':       event_key,
        'chunk':           chunk,
        'event':           event,
        'reco_cluster_id': cluster_id,
        'true_cluster_id': match['true_cluster_id'],
        'channel':         vertex.get('channel'),
        'in_volume':       vertex.get('in_volume'),
        'purity':          match.get('purity'),
        'completeness':    match.get('completeness'),
        'true_energy_mev': (float(np.asarray(true_points)[:, 5].sum())
                            if true_points is not None and len(true_points) else None),
        'reco_energy_mev': reco_energy,
        'tagged_directly': (tagged_detail or {}).get('tagged_directly'),
        'taggers':         (tagged_detail or {}).get('taggers', []),
        'bee_url':         bee_url,
    }


def save_event_tagger_impact(neutrino_matches, tagged_by_cluster, clusters_reco,
                             clusters_true, vertex_records, output_root, event_key,
                             want_removed, want_in_volume,
                             dir_name, title, name_prefix,
                             reco_var_records=None, id_field='reco_cluster_id',
                             bee_links=None, min_completeness=None):
    """
    Every matched pair in ONE event that satisfies (removed?, in volume?).

    - want_removed=True  : the tagger cut removed this cluster
      want_removed=False : it survived
    - want_in_volume     : the matched neutrino's vertex is / is not in volume

    The three notebooks are (True, True) -- in-volume signal removed -- (False, False) --
    surviving out-of-volume neutrinos -- and (True, False) -- out-of-volume
    neutrinos the tagger removed.

    neutrino_matches is match_reco_to_true_neutrino's output for the PRE-cut
    beam-window clusters; tagged_by_cluster is tag_reco_clusters' output for the
    same set.

    reco_var_records must be the CATEGORISED records (categorize_reco_clusters),
    not the raw ones from build_reco_cluster_variable_records: only the former
    carry reco_energy_mev, and both key the cluster as 'reco_cluster_id'. Passing
    the raw records silently gives every figure a reco energy of zero.

    min_completeness, when given, drops matches holding less than that fraction of
    their true neutrino. The tagger populations do not need it: they pair against
    the beam-window set, roughly one cluster per event, where a 5% purity match is
    already meaningful. A caller pairing against the PRE-cut set does -- there are
    ~13 reco clusters per event there, a neutrino incidentally overlaps several,
    and without a floor the population fills with 1-2% fragments that say nothing
    about whether the interaction was really lost (measured on chunk0: 17 of 20
    entries fell below 20% completeness).
    """
    chunk, event = split_event_key(event_key)
    bee_url = bee_event_url((bee_links or {}).get(chunk), event) if event else None
    vertices = neutrino_vertex_lookup(vertex_records)
    energy_by_id = {r.get(id_field): r.get('reco_energy_mev')
                    for r in reco_var_records or []}

    entries = []
    for (_, cluster_id), match in sorted(neutrino_matches.items(), key=lambda kv: kv[0][1]):
        was_removed = cluster_id in (tagged_by_cluster or {})
        if was_removed != want_removed:
            continue
        vertex = vertices.get(match['true_cluster_id'])
        if vertex is None or vertex.get('in_volume') is not want_in_volume:
            continue
        # A completeness floor, for callers pairing against a set where the 5%
        # purity bar is too loose on its own -- see min_completeness above.
        if (min_completeness is not None
                and (match.get('completeness') or 0.0) < min_completeness):
            continue

        reco_points = (clusters_reco or {}).get(cluster_id)
        true_points = (clusters_true or {}).get(match['true_cluster_id'])
        if reco_points is None or true_points is None:
            continue

        tagged_detail = (tagged_by_cluster or {}).get(cluster_id)
        true_energy = float(np.asarray(true_points)[:, 5].sum()) if len(true_points) else 0.0
        reco_energy = energy_by_id.get(cluster_id)

        legend_lines = [
            f"event {event_key}",
            f"{vertex.get('channel')}",
            'in-volume' if vertex.get('in_volume') else 'OUT-of-volume',
            f"purity {(match.get('purity') or 0):.3f}",
            f"completeness {(match.get('completeness') or 0):.3f}",
            f"true E {true_energy:.0f} MeV",
            f"reco E {reco_energy:.0f} MeV" if reco_energy is not None else "reco E n/a",
            f"true id {match['true_cluster_id']:.0f}",
            f"reco id {cluster_id:.3f}",
        ]
        if vertex.get('xyz'):
            vx, vy, vz = vertex['xyz']
            legend_lines.append(f"vertex ({vx:.1f}, {vy:.1f}, {vz:.1f}) cm")
        if tagged_detail is not None:
            # 'reason' lets a caller that is not the cosmic tagger say what
            # removed the cluster -- the beam-window population passes a flash
            # time. Without it, the tagger wording is used.
            if tagged_detail.get('reason'):
                legend_lines.append(tagged_detail['reason'])
            elif tagged_detail.get('tagged_directly'):
                legend_lines.append("REMOVED: tagged by "
                                    + ", ".join(tagged_detail['taggers']))
            else:
                legend_lines.append(
                    "REMOVED: tagged by propagation across the beam window")

        path = draw_reco_true_pair_views(
            reco_points, true_points, output_root, event_key, cluster_id,
            match['true_cluster_id'], legend_lines, dir_name, title, name_prefix,
            vertex.get('channel'), bee_url=bee_url, vertex_xyz=vertex.get('xyz'))
        if path is None:
            continue
        entries.append(_entry(event_key, chunk, event, cluster_id, match, vertex,
                              true_points, reco_energy, tagged_detail, path, bee_url))
    return entries


def _relative_path(entry):
    """
    'chunk0/numu_CC/invol_signal_removed_....png' -- the path relative to the directory
    the bee-link files and the summary sit in, which is two levels above the
    figure now that each chunk splits by channel.
    """
    path = Path(entry['path'])
    return f"{path.parent.parent.name}/{path.parent.name}/{path.name}"


def write_bee_links_by_category(entries, output_root, dir_name,
                                prefix='bee_links', bee_set_url=None):
    """
    One BEE link file per interaction channel, written one directory ABOVE the
    chunk directories so a channel can be worked through on its own.

    A channel with no entries still gets a file, saying so -- an absent file and
    an empty population look the same otherwise, and only one of them means the
    job did what was asked.

    bee_set_url, when given, is the ONE BEE set holding every event of this
    population (build_bee_set_from_links.build_population_bee_set); it is written
    at the top of every channel file, and the per-row urls are events within it.

    Returns {channel: path}.
    """
    output_root = Path(output_root) / dir_name
    output_root.mkdir(parents=True, exist_ok=True)

    written = {}
    for channel in CHANNELS:
        rows = [e for e in entries or [] if e.get('channel') == channel]
        rows.sort(key=lambda e: (str(e['chunk']), int(e['event'] or 0)))
        lines = [f"# BEE event display -- {channel}",
                 "# One line per figure. The same URL is printed on the figure itself,",
                 "# where it cannot be clicked: PNG has no hyperlinks."]
        if bee_set_url:
            lines.append(f"# BEE SET (all channels of this population, one upload): {bee_set_url}")
        lines.append("")
        if not rows:
            lines.append(f"# (no {channel} interactions in this population)")
        for entry in rows:
            if entry.get('bee_url'):
                lines.append(f"{_relative_path(entry)}  {entry['bee_url']}")
            else:
                lines.append(f"{_relative_path(entry)}  (no BEE set for {entry['chunk']})")
        path = output_root / f"{prefix}_{channel}.txt"
        path.write_text("\n".join(lines) + "\n")
        written[channel] = path
    return written


def write_impact_summary(entries, output_root, dir_name, filename,
                         headline, explanation,
                         n_matched_total=None, n_beam_window_clusters=None,
                         n_events=None):
    """
    The counts, per interaction channel, plus a completeness breakdown.

    The breakdown matters because the match bar is loose: "12 numu CC removed"
    reads as twelve lost interactions, but some are clusters holding a fragment.
    Splitting by how much of the neutrino the cluster actually held is what makes
    the number honest.
    """
    output_root = Path(output_root) / dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    entries = entries or []

    bands = [('completeness > 80%   (the interaction is essentially in there)',
              lambda c: c > 0.8),
             ('50-80%               (most of it)', lambda c: 0.5 < c <= 0.8),
             ('20-50%               (a substantial part)', lambda c: 0.2 < c <= 0.5),
             ('<= 20%               (a fragment)', lambda c: c <= 0.2)]

    lines = []
    lines.append("=" * 88)
    lines.append(headline)
    lines.append("=" * 88)
    lines.append("")
    lines.extend(explanation)
    lines.append("")
    if n_events is not None:
        lines.append(f"{n_events} event(s) processed.")
    if n_beam_window_clusters:
        lines.append(f"{n_beam_window_clusters} reco cluster(s) survived the beam-window cut.")
    if n_matched_total is not None:
        lines.append(f"{n_matched_total} of them matched a true neutrino.")
    lines.append("")
    lines.append("-" * 88)
    lines.append(f"  {'channel':<12s}{'clusters':>10s}{'distinct interactions':>24s}"
                 f"{'true E sum (MeV)':>20s}")
    lines.append("-" * 88)
    total_clusters, total_interactions = 0, set()
    for channel in CHANNELS:
        rows = [e for e in entries if e.get('channel') == channel]
        interactions = {(e['event_key'], e['true_cluster_id']) for e in rows}
        energy = sum(e['true_energy_mev'] or 0 for e in rows)
        total_clusters += len(rows)
        total_interactions |= interactions
        lines.append(f"  {channel:<12s}{len(rows):>10d}{len(interactions):>24d}{energy:>20.0f}")
    lines.append("-" * 88)
    lines.append(f"  {'TOTAL':<12s}{total_clusters:>10d}{len(total_interactions):>24d}")
    lines.append("")
    lines.append("A cluster and an interaction are not the same count: two reco clusters can")
    lines.append("match the same true neutrino, so the second column is the number of")
    lines.append("INTERACTIONS involved and is the one to quote.")
    lines.append("")
    lines.append("How much of the neutrino each cluster actually held:")
    lines.append("")
    for label, test in bands:
        n = sum(1 for e in entries if test(e.get('completeness') or 0))
        lines.append(f"    {label:<58s} {n:4d}")
    lines.append("")
    if entries:
        lines.append("-" * 88)
        lines.append(f"  {'event':<14s}{'channel':>9s}{'purity':>9s}{'compl':>8s}"
                     f"{'true E':>9s}{'reco E':>9s}{'reco id':>11s}{'true id':>10s}")
        lines.append("-" * 88)
        for entry in sorted(entries, key=lambda e: -(e.get('completeness') or 0)):
            lines.append(
                f"  {str(entry['event_key']):<14s}{str(entry['channel']):>9s}"
                f"{(entry['purity'] or 0):>9.3f}{(entry['completeness'] or 0):>8.3f}"
                f"{(entry['true_energy_mev'] or 0):>9.0f}{(entry['reco_energy_mev'] or 0):>9.0f}"
                f"{_id_text(entry['reco_cluster_id']):>11s}"
                f"{_id_text(entry['true_cluster_id']):>10s}")
    lines.append("=" * 88)

    path = output_root / filename
    path.write_text("\n".join(lines) + "\n")
    return path
