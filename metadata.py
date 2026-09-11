import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

from variable_pca_linearity import calculate_pca_linearity
from DrawRecoTrueFlashes import BEAM_WINDOW_MIN_US, BEAM_WINDOW_MAX_US

def add_metadata_true_clusters(completeness_results, cluster_category_results, file_name, event, apa, view, event_key=None):
    """
    Create metadata for each true cluster.

    Args:
        completeness_results: List of completeness result dictionaries from EvaluateCompleteness
        cluster_category_results: Dictionary mapping cluster IDs to category info (is_neutrino, track_type)
        file_name: Name of the input file (e.g., "file1")
        event: Event number
        apa: APA number (e.g., "APA0")
        view: View type (e.g., "2view", "3view")
        event_key: Full event key like "file1_0" (if None, will be constructed from file_name and event)

    Returns:
        List of metadata dictionaries, one per unique true cluster
    """
    # Construct full event_key if not provided
    if event_key is None:
        event_key = f"{file_name}_{event}"
    if not completeness_results:
        return []

    # Group completeness data by true cluster
    true_cluster_data = {}
    for eff in completeness_results:
        true_cid = eff['true_cluster_id']

        if true_cid not in true_cluster_data:
            true_cluster_data[true_cid] = {
                'total_completeness': 0,
                'reco_match_count': 0,
                'total_true_energy': eff.get('total_true_cluster_energy', 0)
            }

        true_cluster_data[true_cid]['total_completeness'] += eff['completeness_energy_weighted']
        # The 8888 row is EvaluateCompleteness's "this true cluster matched nothing"
        # sentinel, not a reco cluster, so it must not count towards num_reco_matches --
        # an unmatched true cluster has 0 matches, not 1. (The multiplicity bar chart and
        # non_one_match_*.txt already corrected for this via total_completeness <= 0; the
        # count is now correct at the source, so those corrections simply agree with it.)
        if eff['reco_cluster_id'] != 8888:
            true_cluster_data[true_cid]['reco_match_count'] += 1

    # Create metadata entries for each true cluster
    metadata_list = []

    for true_cid, cluster_info in true_cluster_data.items():
        # Get category information
        category_info = cluster_category_results.get(true_cid, {})
        is_neutrino = category_info.get('is_neutrino', False)
        track_type = category_info.get('track_type', 'normal')

        # Determine cluster type
        cluster_type = 'neutrino' if is_neutrino else 'cosmic'

        # Create metadata dictionary
        metadata = {
            'file_name': file_name,
            'event': event_key,  # Store the full event_key (e.g., "file1_0") for proper matching
            'event_num': event,  # Also store the event number for reference
            'apa': apa,
            'view': view,
            'true_cluster_id': true_cid,
            'cluster_type': cluster_type,  # neutrino or cosmic
            'cluster_category': track_type,  # isochronous, prolonged, normal (only for cosmic)
            'total_completeness': cluster_info['total_completeness'],
            'num_reco_matches': cluster_info['reco_match_count'],
            'total_true_energy': cluster_info['total_true_energy']
        }

        metadata_list.append(metadata)

    return metadata_list


def add_metadata_true_reco_pair_cluster(matched_pairs, cluster_category_results, file_name, event, apa, view, event_key=None):
    """
    Create metadata for each matched true-reco cluster pair (1-to-1 matching).

    Args:
        matched_pairs: List of matched pair dictionaries from MatchTrueToReco1to1, each
            containing true_cluster_id, reco_cluster_id, completeness_energy_weighted, purity,
            total_true_cluster_energy, and total_reco_cluster_charge
        cluster_category_results: Dictionary mapping cluster IDs to category info (is_neutrino, track_type)
        file_name: Name of the input file (e.g., "file1")
        event: Event number
        apa: APA number (e.g., "APA0")
        view: View type (e.g., "2view", "3view")
        event_key: Full event key like "file1_0" (if None, will be constructed from file_name and event)

    Returns:
        List of metadata dictionaries, one per matched true-reco pair
    """
    # Construct full event_key if not provided
    if event_key is None:
        event_key = f"{file_name}_{event}"
    if not matched_pairs:
        return []

    metadata_list = []

    for pair in matched_pairs:
        true_cid = pair['true_cluster_id']
        reco_cid = pair['reco_cluster_id']

        # Get category information
        category_info = cluster_category_results.get(true_cid, {})
        is_neutrino = category_info.get('is_neutrino', False)
        track_type = category_info.get('track_type', 'normal')

        # Determine cluster type
        cluster_type = 'neutrino' if is_neutrino else 'cosmic'

        # Create metadata dictionary
        metadata = {
            'file_name': file_name,
            'event': event_key,  # Store the full event_key (e.g., "file1_0") for proper matching
            'event_num': event,  # Also store the event number for reference
            'apa': apa,
            'view': view,
            'true_cluster_id': true_cid,
            'reco_cluster_id': reco_cid,
            'cluster_type': cluster_type,  # neutrino or cosmic
            'cluster_category': track_type,  # isochronous, prolonged, normal (only for cosmic)
            'completeness': pair.get('completeness_energy_weighted', 0),
            'purity': pair.get('purity', 0),
            'total_true_energy': pair.get('total_true_cluster_energy', 0),
            'total_reco_charge': pair.get('total_reco_cluster_charge', 0)
        }

        metadata_list.append(metadata)

    return metadata_list


def add_metadata_reco_clusters(purity_results, file_name, event, apa, view, event_key=None):
    """
    Create metadata for each reco cluster. Symmetric counterpart to add_metadata_true_clusters,
    aggregated from purity_results (EvaluatePurity) the same way the true-side function
    aggregates from completeness_results (EvaluateCompleteness).

    Args:
        purity_results: List of purity result dictionaries from EvaluatePurity
        file_name: Name of the input file (e.g., "file1")
        event: Event number
        apa: APA number (e.g., "APA0")
        view: View type (e.g., "2view", "3view")
        event_key: Full event key like "file1_0" (if None, will be constructed from file_name and event)

    Returns:
        List of metadata dictionaries, one per unique reco cluster
    """
    if event_key is None:
        event_key = f"{file_name}_{event}"
    if not purity_results:
        return []

    # Group purity data by reco cluster
    reco_cluster_data = {}
    for pur in purity_results:
        reco_cid = pur['reco_cluster_id']

        if reco_cid not in reco_cluster_data:
            reco_cluster_data[reco_cid] = {
                'total_purity': 0,
                'true_match_count': 0,
                'total_reco_charge': pur.get('total_reco_cluster_charge', 0)
            }

        # The unmatched sentinel (true_cluster_id=8888, purity=-0.1) marks a reco cluster
        # with no true match at all - don't fold it into the purity sum/match count.
        if pur.get('true_cluster_id') != 8888:
            reco_cluster_data[reco_cid]['total_purity'] += pur['purity']
            reco_cluster_data[reco_cid]['true_match_count'] += 1

    # Create metadata entries for each reco cluster
    metadata_list = []

    for reco_cid, cluster_info in reco_cluster_data.items():
        metadata = {
            'file_name': file_name,
            'event': event_key,  # Store the full event_key (e.g., "file1_0") for proper matching
            'event_num': event,  # Also store the event number for reference
            'apa': apa,
            'view': view,
            'reco_cluster_id': reco_cid,
            'total_purity': cluster_info['total_purity'],
            'num_true_matches': cluster_info['true_match_count'],
            'total_reco_charge': cluster_info['total_reco_charge']
        }

        metadata_list.append(metadata)

    return metadata_list


# ============================================================================
# EXTRA (UNMATCHED) RECO CLUSTER INVESTIGATION (additive)
# ============================================================================
# Neutrino true clusters carry cluster_id >= 99990 in the charge-light pipeline
# (see reassign_cluster_ID_true_charge_light in selections.py, 99990 + nu_idx).
# This threshold check is the same pattern already used by
# EnergyReconstruction/draw_energy_reconstruction.py and
# AnalysisDistributions/draw_variables.py to split neutrino vs cosmic
# matched pairs, reused here instead of taking cluster_category_results as an
# extra dependency.
NEUTRINO_CLUSTER_ID_BASE = 99990


def _is_neutrino_true_cluster_id(true_cluster_id):
    return true_cluster_id is not None and true_cluster_id >= NEUTRINO_CLUSTER_ID_BASE


def categorize_extra_reco_clusters(clusters_true, clusters_reco, purity_results, matched_pairs,
                                    file_name, event, apa="Combined", event_key=None):
    """
    Categorize every reco cluster in one event by why it is (or isn't) the 1-to-1
    match of a true neutrino cluster -- built to explain why the number of reco
    clusters surviving the beam-window cut exceeds the number of true neutrino
    clusters (see investigate_extra_reco_clusters.py).

    Categories:
      - matched_winner: this reco cluster IS the MatchTrueToReco1to1 winner for a
        true NEUTRINO cluster. Not "extra" -- this is the intended 1-to-1 match.
      - fragment_of_neutrino: purity > 0 (from purity_results) against a true
        neutrino cluster, but a DIFFERENT reco cluster won that true cluster's
        1-to-1 slot. clusterpairmatching.MatchTrueToReco1to1 keeps only the
        highest-completeness reco per true cluster, so the others read as "extra"
        here even though they genuinely overlap the neutrino.
      - matched_cosmic_only: purity > 0 only against cosmic true cluster(s),
        never a neutrino -- correctly reconstructed in-spill cosmic activity,
        not a bug, but still inflates the total reco-in-beam-window count.
      - no_true_overlap: EvaluatePurity's true_cluster_id=8888 sentinel row --
        zero spatial overlap with any true cluster. nearest_true_cluster_id /
        nearest_true_is_neutrino / min_dist / mean_nn_dist / dx,dy,dz are filled
        in via a KDTree nearest-neighbor search of this reco cluster's points
        against every true cluster in the event (same technique as
        near_miss_investigation.py's find_near_miss_rows, but reco-centric), so
        a small min_dist (especially a small dx, the drift/X direction) reads as
        an X-mis-assignment candidate, while a large offset suggests a
        genuinely spurious/noise cluster. No hard distance threshold is
        imposed -- callers sort by min_dist instead.

    Args:
        clusters_true: dict {true_cluster_id: points} for this event
        clusters_reco: dict {reco_cluster_id: points} for this event
        purity_results: EvaluatePurity() output for this event
        matched_pairs: MatchTrueToReco1to1() output for this event
        file_name, event, apa: same convention as add_metadata_true_clusters
        event_key: full event key like "file1_0" (constructed if None)

    Returns:
        List of dicts, one per reco cluster in clusters_reco.
    """
    if event_key is None:
        event_key = f"{file_name}_{event}"

    winner_reco_ids = {pair['reco_cluster_id'] for pair in matched_pairs
                        if _is_neutrino_true_cluster_id(pair['true_cluster_id'])}

    # reco_cluster_id -> [(true_cluster_id, purity), ...] over every non-sentinel match
    reco_true_matches = {}
    for p in purity_results:
        if p['true_cluster_id'] == 8888:
            continue
        reco_true_matches.setdefault(p['reco_cluster_id'], []).append((p['true_cluster_id'], p['purity']))

    true_trees = {cid: KDTree(np.array(pts)[:, :3]) for cid, pts in clusters_true.items()}

    rows = []
    for reco_cid, reco_points in clusters_reco.items():
        reco_points = np.array(reco_points)
        total_reco_charge = np.sum(reco_points[:, 4])
        matches = reco_true_matches.get(reco_cid, [])

        if reco_cid in winner_reco_ids:
            category = 'matched_winner'
        elif matches:
            category = 'fragment_of_neutrino' if any(_is_neutrino_true_cluster_id(tid) for tid, _ in matches) \
                else 'matched_cosmic_only'
        else:
            category = 'no_true_overlap'

        row = {
            'file_name': file_name,
            'event': event_key,
            'event_num': event,
            'apa': apa,
            'reco_cluster_id': reco_cid,
            'category': category,
            'total_reco_charge': total_reco_charge,
            'matched_true_cluster_id': None,
            'purity': None,
            'num_true_matches': len(matches),
            'nearest_true_cluster_id': None,
            'nearest_true_is_neutrino': None,
            'min_dist': None,
            'mean_nn_dist': None,
            'dx': None, 'dy': None, 'dz': None,
        }

        if matches:
            # Best (highest-purity) true match, kept for reference/lookup even
            # when it's not the 1-to-1 winner.
            best_true_id, best_purity = max(matches, key=lambda t: t[1])
            row['matched_true_cluster_id'] = best_true_id
            row['purity'] = best_purity

        if category == 'no_true_overlap' and true_trees:
            reco_xyz = reco_points[:, :3]
            best_true_id, best_min_dist, best_mean_dist, best_offset = None, np.inf, None, None
            for true_cid, tree in true_trees.items():
                dists, idx = tree.query(reco_xyz)
                d = dists.min()
                if d < best_min_dist:
                    nearest_true_pts = np.array(clusters_true[true_cid])[idx, :3]
                    best_true_id      = true_cid
                    best_min_dist     = d
                    best_mean_dist    = dists.mean()
                    best_offset       = (reco_xyz - nearest_true_pts).mean(axis=0)
            row['nearest_true_cluster_id']  = best_true_id
            row['nearest_true_is_neutrino'] = _is_neutrino_true_cluster_id(best_true_id)
            row['min_dist']     = best_min_dist
            row['mean_nn_dist'] = best_mean_dist
            row['dx'], row['dy'], row['dz'] = best_offset

        rows.append(row)

    return rows


# ============================================================================
# UNMATCHED TRUE NEUTRINO CLUSTER INVESTIGATION (additive)
# ============================================================================
# The mirror image of categorize_extra_reco_clusters above: that one asks "why
# are there MORE selected reco clusters than true neutrinos", this one asks
# "why do FEWER true neutrinos find a reco match than exist" (job-wide: 72 true
# neutrinos, only 61 of them in a 1-to-1 pair). See
# investigate_unmatched_true_neutrinos.py.


def _beam_window_offset_us(flash_time):
    """
    Signed distance from flash_time to the beam window, in us: 0.0 inside,
    negative if the flash is early (before BEAM_WINDOW_MIN_US), positive if late
    (after BEAM_WINDOW_MAX_US). Magnitude is what separates "the neutrino sat
    just outside the spill" from "charge-light matching handed this cluster a
    cosmic's flash tens of us away".
    """
    if flash_time < BEAM_WINDOW_MIN_US:
        return float(flash_time - BEAM_WINDOW_MIN_US)
    if flash_time > BEAM_WINDOW_MAX_US:
        return float(flash_time - BEAM_WINDOW_MAX_US)
    return 0.0


# A "would have matched" verdict claims a specific reco cluster WAS the neutrino
# and a selection took it away. That claim needs the pair to be a real pair.
#
# Measured case that forced these: chunk0 event 39, a full-detector cosmic track
# whose flash sat 502 us outside the beam window, clipping the neutrino at
# completeness 0.0004 and purity 0.007. It cleared the old "strict overlap > 0"
# bar and was reported as reco_outside_beam_window -- i.e. as signal the timing
# cut had cost us -- when nothing of the sort had happened.
#
# Below either threshold the reco cluster is not that neutrino's reconstruction,
# so the neutrino is recorded as no_reco_overlap: there was no reco of it to lose.
MIN_WOULD_HAVE_MATCHED_COMPLETENESS = 0.10
MIN_WOULD_HAVE_MATCHED_PURITY       = 0.10

# The img-level (pre charge-light-matching) reconstruction quality above which a
# neutrino counts as "imaged": sed-sce truth vs img-global reco. img-global and
# its truth share the imaging T0 convention, so a neutrino imaged this well
# overlaps its truth here WHATEVER flash charge-light matching later assigns it
# -- the flash only sets the drift (X) coordinate afterwards. An imaged neutrino
# that then has no clustering-level match is therefore charge-light matching's
# doing, not a reconstruction gap -- WHETHER the clustering cluster was drift-
# shifted right off the truth or merely pushed just outside the beam window (a
# correctly-flashed in-time neutrino would have landed inside it).
#
# img completeness is measured RELAXED (>= 1 reco point per true point, not the
# >5 the strict metric wants): img-global is sparser than clustering-global, so
# a strict bar here would read "not imaged" for genuinely imaged neutrinos and
# hand their failure back to the reconstruction.
IMG_MATCH_MIN_COMPLETENESS = MIN_WOULD_HAVE_MATCHED_COMPLETENESS
IMG_MATCH_MIN_PURITY       = MIN_WOULD_HAVE_MATCHED_PURITY


def _true_reco_overlap_metrics(true_points, reco_points, radius_completeness, min_recopoints_threshold):
    """
    Energy-weighted overlap of one true cluster with one reco cluster, at two
    strictnesses, from a single KDTree pass.

    Returns (strict, relaxed):
      - strict : EXACTLY EvaluateCompleteness's completeness_energy_weighted -- the
        energy fraction of the true cluster whose points have MORE than
        min_recopoints_threshold reco points within radius_completeness. This is
        the quantity MatchTrueToReco1to1 needs to be > 0 for a pair to form, so
        strict == 0 means "this reco cluster cannot match this true cluster".
      - relaxed : the same energy fraction but requiring only >=1 reco point
        within radius_completeness. relaxed > 0 while strict == 0 is the signature
        of a broken/sparse reconstruction -- reco charge IS sitting on the true
        neutrino, just never densely enough to clear the neighbor threshold.
    """
    true_points = np.asarray(true_points)
    true_energies = true_points[:, 5]
    total_true_energy = true_energies.sum()
    if total_true_energy <= 0:
        return 0.0, 0.0

    tree = KDTree(np.asarray(reco_points)[:, :3])
    neighbor_counts = np.fromiter(
        (len(n) for n in tree.query_ball_point(true_points[:, :3], r=radius_completeness)),
        dtype=int, count=len(true_points))

    strict  = true_energies[neighbor_counts > min_recopoints_threshold].sum() / total_true_energy
    relaxed = true_energies[neighbor_counts > 0].sum() / total_true_energy
    return float(strict), float(relaxed)


def _true_reco_yz_overlap_metrics(true_points, reco_points, radius_completeness):
    """
    Same energy-weighted overlap as _true_reco_overlap_metrics' `relaxed`, but
    computed in the YZ PROJECTION ONLY -- X is dropped entirely.

    This is the charge-light X-mis-assignment test. Charge-light matching sets a
    cluster's drift coordinate from its matched flash time and touches nothing
    else, so a wrong flash moves the reco cluster in X while leaving Y and Z
    exactly where they were. A true cluster with ZERO 3D overlap but healthy YZ
    overlap can therefore only be separated from its reco by a displacement along
    X -- which is the fingerprint, with no distance threshold needed to see it.

    Measured in BOTH directions, because a one-sided YZ overlap is easy to fake:
    a long cosmic track crossing the YZ region of a small neutrino blob covers
    ~100% of that blob while only ~1% of the track lies on it. That is a
    coincidental crossing, not a drift-shifted reconstruction of the neutrino.
    Requiring both fractions to be high is the same completeness/purity pairing
    EvaluateCompleteness/EvaluatePurity already use, applied to the YZ projection.

    Returns (yz_overlap, yz_reco_frac, dx_mean):
      - yz_overlap : energy fraction of the TRUE cluster with >=1 reco point
        within radius_completeness in YZ (the completeness-like direction)
      - yz_reco_frac : fraction of the RECO cluster's points lying within
        radius_completeness of a true point in YZ (the purity-like direction --
        this is what a passing cosmic track fails)
      - dx_mean : mean (reco_x - true_x) over the overlapping true points, each
        against its nearest-in-YZ reco point -- how far the reco sits from the
        truth along the drift direction. None when nothing overlaps in YZ.
    """
    true_points = np.asarray(true_points)
    reco_points = np.asarray(reco_points)
    true_energies = true_points[:, 5]
    total_true_energy = true_energies.sum()
    if total_true_energy <= 0:
        return 0.0, 0.0, None

    true_yz = true_points[:, 1:3]
    reco_yz = reco_points[:, 1:3]

    dists, idx = KDTree(reco_yz).query(true_yz)
    within = dists <= radius_completeness
    yz_overlap = true_energies[within].sum() / total_true_energy

    reco_dists, _ = KDTree(true_yz).query(reco_yz)
    yz_reco_frac = float((reco_dists <= radius_completeness).mean())

    if not within.any():
        return float(yz_overlap), yz_reco_frac, None

    dx_mean = float((reco_points[idx[within], 0] - true_points[within, 0]).mean())
    return float(yz_overlap), yz_reco_frac, dx_mean


# Minimum YZ overlap -- required of BOTH directions (true-side and reco-side, see
# _true_reco_yz_overlap_metrics). Once its own category (no_reco_overlap_x_shift),
# now folded into wrong_charge_light_matching via the img-level cross-check;
# kept as the bar above which yz_overlap / yz_reco_frac count as a real
# X-displacement fingerprint rather than a stray projection coincidence, for the
# 'charge_light_reason' text and the .txt evidence.
YZ_ALIGNED_MIN_OVERLAP = 0.1


def _closest_flash_to_window(flash_times):
    """
    (flash_time, signed window offset in us) for the flash NEAREST the beam
    window, or (None, None) if there are none. With several flashes on one
    cluster (cathode crossings, re-merged fragments) the near miss is the
    informative one, not an arbitrary pick.
    """
    if not flash_times:
        return None, None
    offsets = [_beam_window_offset_us(t) for t in flash_times]
    i = int(np.argmin(np.abs(offsets)))
    return flash_times[i], offsets[i]


def _fill_true_neutrino_reco_evidence(row, true_points, clusters_reco_all, radius_completeness):
    """
    KDTree nearest-reco offset + YZ-projection overlap for a true neutrino with
    no clustering strict overlap. Sets row['nearest_reco_cluster_id'],
    row['min_dist'/'mean_nn_dist'/'dx'/'dy'/'dz'], row['yz_*'] and row['yz_aligned']
    (best YZ balance >= YZ_ALIGNED_MIN_OVERLAP -- a reco cluster really lines up
    with the neutrino in projection, so a 3D miss is a pure drift shift, not
    "nothing there"). Picks no category; returns the best YZ balance so the
    caller can branch on it. Shared by no_reco_overlap, broken_or_sparse_reco and
    wrong_charge_light_matching's no-3D-overlap variant.
    """
    if not clusters_reco_all:
        return -1.0
    true_points = np.asarray(true_points)
    true_xyz = true_points[:, :3]
    best_cid, best_min, best_mean, best_offset = None, np.inf, None, None
    for reco_cid, reco_points in clusters_reco_all.items():
        tree = KDTree(np.asarray(reco_points)[:, :3])
        dists, idx = tree.query(true_xyz)
        d = dists.min()
        if d < best_min:
            nearest_reco_pts = np.asarray(reco_points)[idx, :3]
            best_cid, best_min = reco_cid, d
            best_mean   = dists.mean()
            best_offset = (nearest_reco_pts - true_xyz).mean(axis=0)
    if best_cid is not None:
        row['nearest_reco_cluster_id'] = best_cid
        row['min_dist']     = float(best_min)
        row['mean_nn_dist'] = float(best_mean)
        row['dx'], row['dy'], row['dz'] = (float(v) for v in best_offset)

    # Ranked by the WEAKER of the two directions, so a long cosmic track that
    # merely crosses the neutrino's YZ region (high true-side, negligible
    # reco-side) cannot win over a genuinely co-located cluster.
    best_balance = -1.0
    for reco_cid, reco_points in clusters_reco_all.items():
        yz_overlap, yz_reco_frac, yz_dx = _true_reco_yz_overlap_metrics(
            true_points, reco_points, radius_completeness)
        balance = min(yz_overlap, yz_reco_frac)
        if balance > best_balance:
            best_balance = balance
            row['yz_overlap']   = yz_overlap
            row['yz_reco_frac'] = yz_reco_frac
            row['yz_dx']        = yz_dx
            row['yz_best_reco_cluster_id'] = reco_cid
    row['yz_aligned'] = bool(best_balance >= YZ_ALIGNED_MIN_OVERLAP)
    return best_balance


def categorize_unmatched_true_neutrinos(clusters_true, clusters_reco_selected, clusters_reco_all,
                                         reco_provenance, beam_window_real_ids, flash_times_by_real_id,
                                         matched_pairs, file_name, event, apa="Combined", event_key=None,
                                         radius_completeness=2, min_recopoints_threshold=5,
                                         tagger_removed_ids=None,
                                         radius_purity_xz=2, radius_purity_yz=5, radius_purity_xy=5,
                                         clusters_img_true=None, clusters_img_reco=None):
    """
    Categorize every TRUE NEUTRINO cluster in one event by whether it found a
    1-to-1 reco match and, if not, why not.

    The diagnosis works by re-running the overlap test against the FULL
    pre-beam-window-cut reco set (clusters_reco_all) and asking how far up the
    chain the true neutrino got before it dropped out. MatchTrueToReco1to1 does
    argmax-per-true-cluster with no reco-side deduplication, so unlike the reco
    side there is no "lost the 1-to-1 slot to a competitor" failure mode here:
    a true cluster is unmatched if and only if NO selected reco cluster reaches
    completeness_energy_weighted > 0 against it.

    IMG-LEVEL CROSS-CHECK. When the caller passes clusters_img_true /
    clusters_img_reco (sed-sce truth + img-global reco, both PRE charge-light-
    matching), each unmatched neutrino is also tested at that level. img-global
    and its truth share the imaging T0 convention, so a neutrino imaged well
    (img_best_completeness RELAXED and img_best_purity >= IMG_MATCH_MIN_*)
    overlaps its truth here regardless of the flash it is later assigned.
    img_best_completeness / img_best_purity / img_best_reco_cluster_id /
    img_match are on every row.

    An imaged neutrino with no clustering-level match is only
    wrong_charge_light_matching when a clustering reco cluster STILL lines up
    with it in the YZ projection (yz_aligned) -- that is a pure drift shift, the
    flash assignment moving the cluster along X and nothing else. If NOTHING
    lines up, not even in projection, there is simply no clustering cluster for
    the neutrino: that is no_reco_overlap regardless of img_match (the row's
    img_ovl / img_pur still record that imaging had it).

    Categories (a true neutrino gets exactly one, tested in this order):
      - matched: this true neutrino IS in a MatchTrueToReco1to1 pair. Not a
        failure -- carried in the returned rows so one list describes all of
        them, same as categorize_extra_reco_clusters' 'matched_winner'.

      - wrong_charge_light_matching: the neutrino WAS imaged (img_match) and a
        clustering reco cluster of it exists, but charge-light matching's flash
        assignment set its drift (X) coordinate wrong. charge_light_reason:
          'no flash attached'           -- a cluster with partial 3D overlap
                                           survived but charge-light bridged no
                                           flash, so the beam-window ID filter
                                           dropped it
          'wrong flash (out of window)' -- same, but a flash outside the window
                                           (for an imaged in-time neutrino that
                                           means the flash is wrong). winner_
                                           flash_* report it and its offset
          'drift shift (YZ aligns, X off)' -- NO 3D overlap left, but a clustering
                                           reco cluster still matches in YZ
                                           (yz_overlap / yz_reco_frac clear
                                           YZ_ALIGNED_MIN_OVERLAP); yz_dx is how
                                           far it moved along drift
        Merges what used to be reco_no_flash_match, no_reco_overlap_x_shift and
        the wrong-flash rows of reco_outside_beam_window.
      - removed_by_cosmic_tagger: a reco cluster in the FULL set reaches
        completeness > 0 against this true neutrino AND its flash is inside the
        beam window, but selections.apply_cosmic_tagger_cut removed it. The
        reconstruction and its timing were both fine; the tagger judged it cosmic.
        Only populated when the caller passes tagger_removed_ids. Tested BEFORE
        wrong_charge_light_matching so a tagger removal is never blamed on the
        flash.
      - reco_outside_beam_window: defensive residual -- a reco cluster in the
        FULL set WOULD have matched (completeness/purity above
        MIN_WOULD_HAVE_MATCHED_*) with a flash outside the window, yet the
        neutrino was NOT imaged. Clustering reconstructed something imaging did
        not, so the flash cannot be blamed with the img cross-check; almost
        always empty (clustering-global is built from img-global).
      - broken_or_sparse_reco: no reco cluster reaches completeness > 0 at
        clustering level, yet reco points DO sit on the true neutrino
        (best_relaxed_overlap > 0). Fragmented or too sparse to clear the
        neighbour threshold -- the "broken neutrino" case.
      - no_reco_overlap: no 3D clustering overlap AND no YZ alignment -- there is
        no clustering reco cluster for this neutrino at all. img_ovl / img_pur on
        the row say whether IMAGING had it (imaged, then lost at the clustering /
        charge-light stage) or not (never reconstructed anywhere). nearest_reco_*
        / min_dist / dx,dy,dz and yz_* are filled as evidence.
      - unexplained: defensive only. A reco cluster that IS in the selected set
        reaches completeness > 0 yet no pair formed -- impossible given the
        matching code above, so it would signal that this script and the
        notebook pipeline have drifted apart rather than a physics effect.

    Args:
        clusters_true: dict {true_cluster_id: points} for this event
        clusters_reco_selected: dict {reco_cluster_id: points}, AFTER the
            beam-window cut -- the set MatchTrueToReco1to1 actually saw
        clusters_reco_all: dict {reco_cluster_id: points}, the same event's reco
            clusters BEFORE the beam-window cut (superset of the above, sharing
            its cluster IDs -- see _group_reco_with_provenance in the driver)
        reco_provenance: dict {reco_cluster_id: [clustering real_cluster_id, ...]}
            linking each grouped reco cluster back to the raw IDs the flash
            records and the beam-window cut are keyed on
        beam_window_real_ids: set of clustering real_cluster_ids that passed the
            beam-window cut
        flash_times_by_real_id: dict {clustering real_cluster_id: [flash_time_us, ...]}
            from build_img_cluster_flash_metadata; a real_cluster_id absent here
            had no flash attached by charge-light matching
        matched_pairs: MatchTrueToReco1to1() output for this event
        file_name, event, apa: same convention as add_metadata_true_clusters
        event_key: full event key like "file1_0" (constructed if None)
        radius_completeness, min_recopoints_threshold: must be the SAME values the
            driver passed to EvaluateCompleteness, or the strict overlap recomputed
            here won't reproduce the matching it is trying to explain

    Returns:
        List of dicts, one per TRUE NEUTRINO cluster in clusters_true (cosmic
        true clusters are skipped entirely).
    """
    if event_key is None:
        event_key = f"{file_name}_{event}"

    matched_true_ids = {pair['true_cluster_id'] for pair in matched_pairs}
    beam_window_real_ids = set(beam_window_real_ids)

    rows = []
    for true_cid, true_points in clusters_true.items():
        if not _is_neutrino_true_cluster_id(true_cid):
            continue

        true_points = np.asarray(true_points)
        extent = true_points[:, :3].max(axis=0) - true_points[:, :3].min(axis=0)

        row = {
            'file_name': file_name,
            'event': event_key,
            'event_num': event,
            'apa': apa,
            'true_cluster_id': true_cid,
            'category': None,
            'n_true_points': len(true_points),
            'total_true_energy': float(true_points[:, 5].sum()),
            'linearity': calculate_pca_linearity(true_points),
            'extent_x': float(extent[0]), 'extent_y': float(extent[1]), 'extent_z': float(extent[2]),
            'matched_reco_cluster_id': None,
            'completeness': None,
            # Best overlap found anywhere in the PRE-cut reco set, at both strictnesses.
            'best_strict_reco_cluster_id': None,
            'best_strict_overlap': 0.0,
            'best_strict_purity': None,
            'best_relaxed_reco_cluster_id': None,
            'best_relaxed_overlap': 0.0,
            'n_overlapping_reco_clusters': 0,
            'n_overlapping_in_beam_window': 0,
            # IMG-LEVEL (pre charge-light) reconstruction of this same neutrino.
            'img_best_reco_cluster_id': None,
            'img_best_completeness': 0.0,
            'img_best_purity': None,
            'img_match': False,
            # Filled for wrong_charge_light_matching / reco_outside_beam_window.
            'winner_in_beam_window': None,
            'winner_flash_time': None,
            'winner_flash_offset_us': None,
            'charge_light_reason': None,
            # Filled for no_reco_overlap / wrong_charge_light_matching.
            'nearest_reco_cluster_id': None,
            'min_dist': None,
            'mean_nn_dist': None,
            'dx': None, 'dy': None, 'dz': None,
            # YZ-projection (charge-light X-mis-assignment) test, same rows.
            'yz_best_reco_cluster_id': None,
            'yz_overlap': None,
            'yz_reco_frac': None,
            'yz_dx': None,
            'yz_aligned': None,
        }

        if true_cid in matched_true_ids:
            pair = next(p for p in matched_pairs if p['true_cluster_id'] == true_cid)
            row['category'] = 'matched'
            row['matched_reco_cluster_id'] = pair['reco_cluster_id']
            row['completeness'] = pair['completeness_energy_weighted']
            rows.append(row)
            continue

        # --- Unmatched: re-test against every PRE-cut reco cluster ---
        for reco_cid, reco_points in clusters_reco_all.items():
            strict, relaxed = _true_reco_overlap_metrics(true_points, reco_points,
                                                          radius_completeness, min_recopoints_threshold)
            if relaxed > 0:
                row['n_overlapping_reco_clusters'] += 1
                if any(rid in beam_window_real_ids for rid in reco_provenance.get(reco_cid, [])):
                    row['n_overlapping_in_beam_window'] += 1
            if strict > row['best_strict_overlap']:
                row['best_strict_overlap'] = strict
                row['best_strict_reco_cluster_id'] = reco_cid
            if relaxed > row['best_relaxed_overlap']:
                row['best_relaxed_overlap'] = relaxed
                row['best_relaxed_reco_cluster_id'] = reco_cid

        # PURITY of the best-overlapping cluster, with the pipeline's own
        # EvaluatePurity so the number means what purity means everywhere else.
        # Needed here and not only for display: it is half the test below.
        if row['best_strict_overlap'] > 0 and row['best_strict_reco_cluster_id'] in clusters_reco_all:
            from completeness_purity_estimate import EvaluatePurity
            winner_cid = row['best_strict_reco_cluster_id']
            for rec in EvaluatePurity({true_cid: true_points},
                                      {winner_cid: clusters_reco_all[winner_cid]}, event_key,
                                      radius_purity_xz, radius_purity_yz, radius_purity_xy):
                if rec.get('purity') is not None:
                    row['best_strict_purity'] = float(rec['purity'])
                    break

        # Does the winner actually look like this neutrino's reconstruction? See
        # MIN_WOULD_HAVE_MATCHED_* for why a bare overlap > 0 is not enough.
        would_have_matched = (
            row['best_strict_overlap'] >= MIN_WOULD_HAVE_MATCHED_COMPLETENESS
            and row.get('best_strict_purity') is not None
            and row['best_strict_purity'] >= MIN_WOULD_HAVE_MATCHED_PURITY)

        # --- IMG-LEVEL cross-check: was this neutrino reconstructed BEFORE
        # charge-light matching set the drift coordinate? sed-sce truth vs
        # img-global reco, keyed by the same 99990+nu_idx. img_best_completeness
        # is RELAXED (img-global is sparser) -- see IMG_MATCH_MIN_* and the
        # docstring's IMG-LEVEL CROSS-CHECK section.
        img_true_pts = (clusters_img_true or {}).get(true_cid)
        if img_true_pts is not None and len(img_true_pts) and clusters_img_reco:
            img_true_pts = np.asarray(img_true_pts)
            for img_reco_cid, img_reco_pts in clusters_img_reco.items():
                _s, relaxed = _true_reco_overlap_metrics(img_true_pts, img_reco_pts,
                                                         radius_completeness, min_recopoints_threshold)
                if relaxed > row['img_best_completeness']:
                    row['img_best_completeness'] = relaxed
                    row['img_best_reco_cluster_id'] = img_reco_cid
            if row['img_best_completeness'] > 0 and row['img_best_reco_cluster_id'] in clusters_img_reco:
                from completeness_purity_estimate import EvaluatePurity
                icid = row['img_best_reco_cluster_id']
                for rec in EvaluatePurity({true_cid: img_true_pts},
                                          {icid: clusters_img_reco[icid]}, event_key,
                                          radius_purity_xz, radius_purity_yz, radius_purity_xy):
                    if rec.get('purity') is not None:
                        row['img_best_purity'] = float(rec['purity'])
                        break
        row['img_match'] = (row['img_best_completeness'] >= IMG_MATCH_MIN_COMPLETENESS
                            and row['img_best_purity'] is not None
                            and row['img_best_purity'] >= IMG_MATCH_MIN_PURITY)

        # A partial 3D clustering overlap that "would have matched" is the only
        # case that does NOT need the nearest-reco / YZ evidence -- the winner
        # cluster itself IS the evidence. Everything else has no clustering reco
        # sitting on the neutrino in 3D, so fill the evidence once here and let
        # the category logic branch on yz_aligned (a reco cluster lines up in
        # YZ = drift shift) vs nothing there.
        first_branch = row['best_strict_overlap'] > 0 and would_have_matched
        if not first_branch:
            _fill_true_neutrino_reco_evidence(row, true_points, clusters_reco_all, radius_completeness)

        if first_branch:
            winner_cid    = row['best_strict_reco_cluster_id']
            winner_reals  = reco_provenance.get(winner_cid, [])
            winner_flashes = [t for rid in winner_reals for t in flash_times_by_real_id.get(rid, [])]
            row['winner_in_beam_window'] = winner_cid in clusters_reco_selected

            if row['winner_in_beam_window']:
                row['category'] = 'unexplained'
            elif winner_cid in (tagger_removed_ids or ()):
                # In the window, but the cosmic tagger cut took it. Tested BEFORE
                # wrong_charge_light_matching so a tagger removal is never blamed
                # on the flash: this cluster HAS an in-window flash.
                row['category'] = 'removed_by_cosmic_tagger'
                row['winner_flash_time'], row['winner_flash_offset_us'] = \
                    _closest_flash_to_window(winner_flashes)
            elif row['img_match']:
                # Imaged AND partly clustering-reconstructed (some 3D overlap
                # survives), but the selection lost it via the flash: no flash
                # bridged, or a flash outside the window -- which for an imaged
                # in-time neutrino means the flash is wrong.
                row['category'] = 'wrong_charge_light_matching'
                row['charge_light_reason'] = ('no flash attached' if not winner_flashes
                                              else 'wrong flash (out of window)')
                row['winner_flash_time'], row['winner_flash_offset_us'] = \
                    _closest_flash_to_window(winner_flashes)
            else:
                # Clustering reconstructed something imaging did not -- the flash
                # cannot be blamed with the img cross-check. Defensive residual;
                # almost always empty (clustering-global is built from img-global).
                row['category'] = 'reco_outside_beam_window'
                row['winner_flash_time'], row['winner_flash_offset_us'] = \
                    _closest_flash_to_window(winner_flashes)

        elif row['img_match'] and row.get('yz_aligned'):
            # Imaged, no 3D clustering overlap, but a clustering reco cluster
            # STILL lines up with the neutrino in the YZ projection. Charge-light
            # matching sets the drift (X) coordinate from the flash time and
            # touches nothing else, so YZ-aligned-but-3D-missed is a pure X shift
            # -- a wrong flash. yz_dx is how far along the drift direction.
            row['category'] = 'wrong_charge_light_matching'
            row['charge_light_reason'] = 'drift shift (YZ aligns, X off)'

        elif row['best_relaxed_overlap'] > 0 and row['best_strict_overlap'] == 0:
            # Reco points sit on the neutrino but no single cluster is dense
            # enough -- fragmented/sparse reconstruction.
            row['category'] = 'broken_or_sparse_reco'

        else:
            # No 3D clustering overlap AND no YZ alignment -- there is simply no
            # clustering reco cluster for this neutrino, drift-shifted or not.
            # img_ovl / img_pur on the row still say which sub-case it is:
            #   img_match True  -> imaged, then lost entirely at the clustering-
            #                      global / charge-light stage (not a drift shift:
            #                      nothing lines up even in projection)
            #   img_match False -> never reconstructed at any level
            # A reco cluster that lines up ONLY in YZ but was never imaged is a
            # coincidental cosmic crossing and lands here too -- the img
            # cross-check plus the yz_aligned gate is what keeps it out of
            # wrong_charge_light_matching.
            row['category'] = 'no_reco_overlap'

        rows.append(row)

    return rows


def add_single_metadata(metadata_list, field_name, value_lookup,
                          key_fields=('file_name', 'event', 'apa', 'true_cluster_id'), default=None):
    """
    Attach one additional field to every entry of an existing metadata list, looked up by key.

    Lets you extend metadata already built by add_metadata_true_clusters (or any other
    list of per-cluster dicts) with a new per-cluster quantity without rebuilding the whole
    list. For example, adding PCA linearity to true-cluster metadata:

        linearity_lookup = {
            (file_name, event_key, apa, true_cluster_id): linearity_value,
            ...
        }
        add_single_metadata(true_metadata_list, 'linearity', linearity_lookup)

    Args:
        metadata_list: List of metadata dictionaries (modified in place).
        field_name: Name of the new field to add to each dictionary.
        value_lookup: Dict mapping a key_fields tuple to the value for that cluster.
        key_fields: Dictionary keys used to build the lookup key, in order (default matches
            the schema produced by add_metadata_true_clusters).
        default: Value to assign when a metadata entry has no matching key in value_lookup.

    Returns:
        The same metadata_list, with field_name added to every entry.
    """
    for metadata in metadata_list:
        key = tuple(metadata[k] for k in key_fields)
        metadata[field_name] = value_lookup.get(key, default)

    return metadata_list


def aggregate_metadata(metadata_list):
    """
    Aggregate metadata entries across multiple events/files.
    Useful for file-level and job-level analysis.

    Args:
        metadata_list: List of metadata dictionaries from multiple calls to add_metadata_true_clusters

    Returns:
        Aggregated metadata dictionary with summary statistics
    """
    if not metadata_list:
        return {}

    # Group by cluster type and category
    stats = {
        'total_clusters': len(metadata_list),
        'by_type': {
            'neutrino': 0,
            'cosmic': 0
        },
        'by_category': {
            'neutrino': 0,
            'isochronous': 0,
            'prolonged': 0,
            'normal': 0
        },
        'completeness_stats': {
            'mean': np.mean([m['total_completeness'] for m in metadata_list]),
            'median': np.median([m['total_completeness'] for m in metadata_list]),
            'min': np.min([m['total_completeness'] for m in metadata_list]),
            'max': np.max([m['total_completeness'] for m in metadata_list])
        },
        'reco_matches_stats': {
            'mean': np.mean([m['num_reco_matches'] for m in metadata_list]),
            'median': np.median([m['num_reco_matches'] for m in metadata_list]),
            'min': np.min([m['num_reco_matches'] for m in metadata_list]),
            'max': np.max([m['num_reco_matches'] for m in metadata_list])
        }
    }

    # Count by type and category
    for metadata in metadata_list:
        cluster_type = metadata['cluster_type']
        category = metadata['cluster_category']

        stats['by_type'][cluster_type] += 1
        stats['by_category'][category] += 1

    return stats


def print_metadata(metadata_list):
    """
    Print metadata in a formatted table, showing every field present across the metadata
    dictionaries (not a fixed subset) so fields attached later via add_single_metadata
    (e.g. linearity) show up automatically without needing this function updated.

    Args:
        metadata_list: List of metadata dictionaries
    """
    if not metadata_list:
        print("No metadata to display")
        return

    # Union of keys across all entries, in first-seen order (handles entries where a
    # field was only attached to some rows).
    columns = []
    seen = set()
    for metadata in metadata_list:
        for key in metadata:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        if value is None:
            return "N/A"
        return str(value)

    rows = [[format_value(metadata.get(col, "N/A")) for col in columns] for metadata in metadata_list]
    widths = [max(len(col), *(len(row[i]) for row in rows)) + 2 for i, col in enumerate(columns)]
    total_width = sum(widths)

    print("\n" + "="*total_width)
    print("".join(col.ljust(widths[i]) for i, col in enumerate(columns)))
    print("="*total_width)

    for row in rows:
        print("".join(value.ljust(widths[i]) for i, value in enumerate(row)))

    print("="*total_width + "\n")


# ============================================================================
# CHARGE-LIGHT MATCHING FORMAT (additive; existing functions above are untouched)
# ============================================================================

def build_cluster_flash_metadata(op_data, file_name, event, apa, event_key=None):
    """
    Build per-img-global-cluster-ID optical flash records from the per-flash
    arrays returned by read_op_json() (readfiles.py). op_cluster_ids[i] lists
    the img-global reco cluster IDs matched to flash index i (empty if none);
    every cluster ID in that bracket shares flash i's time (op_t[i]) and APA
    (apa[i]). Verified against the test data: no cluster ID appears in more
    than one flash's bracket, so each cluster ID produces exactly one record.
    Intended for future work (cathode-crossing matching, x-drift-distance
    correction) -- not used for that yet, but feeds DrawRecoTrueFlashes.draw_flashes().

    Args:
        op_data: Dict from read_op_json()
        file_name: Name of the input file (e.g., "file0")
        event: Event number
        apa: Processing-level APA label (e.g. "Combined") -- NOT the same as
            each flash's own detector APA assignment, stored separately below
            as 'flash_apa'
        event_key: Full event key like "file0_9" (if None, constructed from file_name and event)

    Returns a list of dicts (one per matched cluster ID, same shape convention
    as add_metadata_true_clusters/add_metadata_true_reco_pair_cluster) so
    records from multiple events can be concatenated for file/job-level
    aggregation -- see DrawRecoTrueFlashes.draw_flashes().
    """
    if event_key is None:
        event_key = f"{file_name}_{event}"

    records = []
    for flash_index, cluster_ids in enumerate(op_data['op_cluster_ids']):
        if not cluster_ids:
            continue
        flash_time = op_data['op_t'][flash_index]
        flash_apa  = op_data['apa'][flash_index]
        for cluster_id in cluster_ids:
            records.append({
                'file_name': file_name,
                'event': event_key,
                'event_num': event,
                'apa': apa,
                'reco_cluster_id': cluster_id,
                'flash_index': flash_index,
                'flash_time': flash_time,
                'flash_apa': flash_apa,
            })
    return records


CATHODE_CROSSING_TIME_DIFF_MAX_US = 0.02  # 20 ns


def build_img_cluster_flash_metadata(img_data, clustering_data, cluster_flash_records, file_name, event, apa, event_key=None):
    """
    Bridge op.json flash info (attached to img-global cluster IDs by
    build_cluster_flash_metadata) onto clustering-global cluster IDs, even
    though img-global and clustering-global use unrelated cluster_id
    numbering (clustering-global is built AFTER charge-light matching --
    points get re-clustered, possibly merged across the cathode -- so
    cluster_id can't be used to connect the two files directly). This
    function only READS img_data/clustering_data to build an association
    table -- it never modifies or merges the underlying point data in either
    file; any merging visible in clustering-global's clusters already
    happened upstream, before either file was read here.

    The join key is each point's charge ('q'): verified against test data
    that every clustering-global q value has a matching q value somewhere in
    img-global (point order/clustering differs completely between the two
    files, so this must be a value match, not an index match -- only ~0.1%
    of points happen to share position at the same index). img-global q
    values are ~99.2% unique; the rare duplicates are dropped as ambiguous
    rather than guessed at.

    A clustering cluster can match multiple img clusters. Two cases:
      - Same flash (identical flash_time): fragmented img-level sub-clusters
        re-merged into one clustering cluster -- kept as separate records,
        nothing to resolve (their times already agree exactly).
      - Different flashes, close in time (within CATHODE_CROSSING_TIME_DIFF_MAX_US)
        AND from different APAs: this is a cathode-crossing track -- the same
        physical light burst reconstructed independently by each side's
        optical system. These are merged into ONE record with the averaged
        flash_time (verified against test data: e.g. 1.09731/1.09872 us,
        APA 1/0 -> averaged to 1.098015 us). Close-in-time flashes from the
        SAME APA are NOT merged -- that's not the cathode-crossing signature,
        just coincidence (or, as with same-time entries, already handled above).
      - Otherwise (genuinely different, unrelated flashes): kept as separate records.

    Args:
        img_data: Tuple from readfiles.read_img_global_from_json() --
            (x, y, z, cluster_id, q, real_cluster_id)
        clustering_data: Tuple from readfiles.read_cluster_global_from_json() --
            (x, y, z, cluster_id, q, real_cluster_id)
        cluster_flash_records: Output of build_cluster_flash_metadata() for
            this same event (img-global-cluster-ID-keyed flash records)
        file_name: Name of the input file (e.g., "file0")
        event: Event number
        apa: Processing-level APA label (e.g. "Combined")
        event_key: Full event key like "file0_9" (if None, constructed from file_name and event)

    Returns a list of dicts, one per resolved (clustering_cluster_id, flash)
    match -- clustering clusters with no flash-matched img cluster are simply
    absent, same convention as build_cluster_flash_metadata. Each record has
    'img_cluster_id' (representative, first-matched) and 'img_cluster_ids'
    (full list -- length 2 for a cathode-crossing merge), 'flash_apa' and
    'is_cathode_crossing' (bool).
    """
    if event_key is None:
        event_key = f"{file_name}_{event}"

    _, _, _, img_cluster_id, img_q, _ = img_data
    # clustering-global's own 'cluster_id' is a COARSER grouping than
    # 'real_cluster_id' -- it can merge multiple physically distinct tracks
    # that only 'real_cluster_id' keeps separate (confirmed against real data:
    # a single cluster_id spanning two disjoint Y ranges that split cleanly
    # into two real_cluster_id values, each matching a different true
    # cluster). Group/key by real_cluster_id here so clustering_cluster_id in
    # the output records reflects the physically correct clusters.
    _, _, _, _, clu_q, clu_real_cluster_id = clustering_data

    # img cluster_id -> list of flash records (a cluster can only appear in
    # one flash's bracket per build_cluster_flash_metadata, but keep this
    # general in case that changes).
    img_flash_lookup = {}
    for record in cluster_flash_records:
        img_flash_lookup.setdefault(record['reco_cluster_id'], []).append(record)

    # q value -> img cluster_id, dropping ambiguous (duplicate, conflicting) q values.
    q_to_img_cluster = {}
    ambiguous_q = set()
    for q, cid in zip(img_q, img_cluster_id):
        if q in q_to_img_cluster and q_to_img_cluster[q] != cid:
            ambiguous_q.add(q)
        else:
            q_to_img_cluster[q] = cid
    for q in ambiguous_q:
        del q_to_img_cluster[q]

    # Tally, per clustering cluster, how many of its points matched into each img cluster.
    clu_to_img_counts = {}
    for q, clu_cid in zip(clu_q, clu_real_cluster_id):
        img_cid = q_to_img_cluster.get(q)
        if img_cid is None:
            continue
        counts = clu_to_img_counts.setdefault(clu_cid, {})
        counts[img_cid] = counts.get(img_cid, 0) + 1

    records = []
    for clu_cid, img_counts in clu_to_img_counts.items():
        # One entry per (img_cluster_id, flash) match for this clustering cluster.
        entries = []
        for img_cid, n_matched_points in img_counts.items():
            for flash_record in img_flash_lookup.get(img_cid, []):
                entries.append({
                    'img_cluster_id': img_cid,
                    'n_matched_points': n_matched_points,
                    'flash_index': flash_record['flash_index'],
                    'flash_time': flash_record['flash_time'],
                    'flash_apa': flash_record['flash_apa'],
                })
        if not entries:
            continue

        # Greedy adjacent merge (by time) of entries within
        # CATHODE_CROSSING_TIME_DIFF_MAX_US of the previous entry AND from a
        # different APA. Entries with identical time (fragmented same-flash
        # matches, e.g. img sub-clusters re-merged) never satisfy "different
        # APA from a same-time same-APA neighbor" unless they genuinely are
        # on a different APA, so they naturally stay unmerged as before.
        entries.sort(key=lambda e: e['flash_time'])
        groups = []
        for entry in entries:
            if groups and (entry['flash_time'] - groups[-1][-1]['flash_time'] <= CATHODE_CROSSING_TIME_DIFF_MAX_US
                            and entry['flash_apa'] != groups[-1][-1]['flash_apa']):
                groups[-1].append(entry)
            else:
                groups.append([entry])

        for group in groups:
            is_cathode_crossing = len(group) > 1
            avg_time = sum(e['flash_time'] for e in group) / len(group)
            records.append({
                'file_name': file_name,
                'event': event_key,
                'event_num': event,
                'apa': apa,
                'clustering_cluster_id': clu_cid,
                'img_cluster_id': group[0]['img_cluster_id'],
                'img_cluster_ids': [e['img_cluster_id'] for e in group],
                'n_matched_points': sum(e['n_matched_points'] for e in group),
                'flash_index': group[0]['flash_index'],
                'flash_time': avg_time,
                'flash_apa': group[0]['flash_apa'] if not is_cathode_crossing else '/'.join(e['flash_apa'] for e in group),
                'is_cathode_crossing': is_cathode_crossing,
            })
    return records


def build_true_cluster_type_records(clusters_true, file_name, event, event_key=None):
    """
    Per-true-cluster {cluster_id, is_neutrino, nu_idx_values} records for
    DrawLabelsAggregated and DrawLabelsByNuIdx.

    is_neutrino is determined by q_true>0 on the cluster's points, NOT via
    cluster_category_results (which checks only the FIRST point and assumes
    q_true==1 exactly -- silently wrong once q_true can be a neutrino index
    of 2 or higher). All points in a given post-reassignment cosmic cluster
    share q_true=0, so checking any single point is safe there -- the bug in
    the pre-existing code was the ==1 comparison, not the single-point read.

    nu_idx_values: sorted list of the DISTINCT nu_idx (q_true) values present
    among the cluster's points; [] for cosmic clusters. reassign_cluster_ID_true
    merges ALL true clusters with any q_true>0 into a single cluster_id=9999,
    regardless of how many distinct neutrino interactions contributed points --
    so cluster 9999 can itself contain multiple nu_idx values, which is why
    this is a list rather than a single value. is_neutrino's definition and
    the one-record-per-post-reassignment-cluster_id shape are UNCHANGED so
    DrawLabelsAggregated's counts are unaffected by this field.

    There is deliberately NO in_beam_window field. True clusters carry no flash
    and no time (build_true_points_charge_light fills the time column with
    zeros), so any "true cluster in the beam window" flag could only be inferred
    by spatially matching to a reco cluster whose flash landed in the window --
    which mixes beam timing with reconstruction + flash-matching completeness
    while reading as a truth-level timing selection. Beam-window membership is
    a RECO-side quantity only (see build_img_cluster_flash_metadata and
    writeinformation.write_reco_cluster_info); do not reintroduce it here.

    Args:
        clusters_true: Dict {cluster_id: points}, points columns
            [x, y, z, cluster_id, q_true, energy, time] (post reassign_cluster_ID_true)
        file_name: Name of the input file (e.g., "file0")
        event: Event number
        event_key: Full event key like "file0_9" (if None, constructed from file_name and event)

    Returns:
        List of dicts, one per true cluster:
            {file_name, event, event_num, cluster_id, is_neutrino, nu_idx_values}
    """
    if event_key is None:
        event_key = f"{file_name}_{event}"
    if not clusters_true:
        return []

    records = []
    for cluster_id, points in clusters_true.items():
        points = np.array(points)
        is_neutrino = bool(points[0, 4] > 0) if len(points) > 0 else False
        if is_neutrino:
            nu_idx_values = sorted(set(int(v) for v in points[:, 4] if v > 0))
        else:
            nu_idx_values = []
        records.append({
            'file_name': file_name,
            'event': event_key,
            'event_num': event,
            'cluster_id': cluster_id,
            'is_neutrino': is_neutrino,
            'nu_idx_values': nu_idx_values,
        })
    return records


# Charged leptons that identify a charged-current interaction, by the flavor of
# neutrino they imply. mc.json writes particle names, not PDG codes, so these are
# matched by name; both charges are listed since an anti-neutrino produces the
# positive one.
CC_LEPTONS_BY_FLAVOR = {
    'numu': ('mu-', 'mu+'),
    'nue':  ('e-', 'e+'),
}


def classify_neutrino_interaction(daughter_particles, flavor=None):
    """
    Charged-current / neutral-current classification of one neutrino interaction
    from the INTERACTING NEUTRINO'S FLAVOR together with the particles in its
    FIRST list of daughters -- the direct children of the mc.json
    interaction-vertex node, i.e. what came straight out of the interaction, not
    what those products later decayed into.

    The rule:
      - a numu interaction with a muon (mu-/mu+) among the direct daughters
                                                          -> numu CC
      - a nue interaction with an electron (e-/e+) there   -> nue CC
      - no muon and no electron there                      -> NC

    A CC label therefore needs the charged lepton to MATCH the incident flavor:
    the charged lepton of a charged-current interaction is the partner of the
    neutrino that made it, so a muon in a nue event (or an electron in a numu
    event) is not that partner and cannot make the interaction CC.

    That leaves one more case the three bullets do not name: a charged lepton
    present, but of the wrong flavor. It is classified NC -- no CC lepton of the
    interacting flavor was produced, so the neutrino carried on -- and flagged
    with lepton_flavor_mismatch=True, naming the offending particle in
    mismatched_lepton, so the case stays visible instead of being folded silently
    into the NC pile. It is real physics rather than a parse error: the one
    instance in the current dataset is a numu event whose daughters are
    e+, gamma, neutron, neutron, numu -- an outgoing numu (the NC signature) with
    a positron from the photon. Under the previous, flavor-blind rule that same
    interaction was called nue CC.

    With flavor=None (older mc.json files carry no flavor on the root node) the
    flavor test cannot run, so the rule falls back to the lepton alone -- muon ->
    numu CC, electron -> nue CC -- and sets flavor_known=False to say the result
    was not flavor-checked.

    Parameters:
    - daughter_particles: iterable of particle-name strings, the direct daughters
    - flavor: the root node's neutrino species ('numu', 'nue', 'anumu', ...)

    Returns:
        Dict with
          interaction_type       : 'CC' or 'NC'
          interaction_channel    : 'numu_CC', 'nue_CC' or 'NC'
          primary_lepton         : the charged lepton that made it CC, else None
          lepton_flavor_mismatch : True when a charged lepton was present but of
                                   the wrong flavor (so the interaction is NC)
          mismatched_lepton      : that lepton's name, else None
          flavor_known           : False when no flavor was available to test
    """
    names = list(daughter_particles or [])

    # The charged lepton present for each flavor, if any.
    lepton_by_flavor = {
        candidate: next((n for n in names if n in leptons), None)
        for candidate, leptons in CC_LEPTONS_BY_FLAVOR.items()
    }

    flavor_text  = "" if flavor is None else str(flavor)
    flavor_known = flavor is not None

    if flavor_known:
        # Substring, not equality: the flavor string can carry an anti-neutrino
        # prefix/suffix ('anumu', 'numubar') that still names the same flavor.
        # 'nue' is not a substring of 'numu', so the two never cross-match.
        matching_flavors = [c for c in CC_LEPTONS_BY_FLAVOR if c in flavor_text]
    else:
        # No flavor to test against: fall back to the lepton alone, muon first so
        # that an interaction with both is numu CC.
        matching_flavors = list(CC_LEPTONS_BY_FLAVOR)

    for candidate in matching_flavors:
        lepton = lepton_by_flavor.get(candidate)
        if lepton is not None:
            return {
                'interaction_type': 'CC',
                'interaction_channel': f'{candidate}_CC',
                'primary_lepton': lepton,
                'lepton_flavor_mismatch': False,
                'mismatched_lepton': None,
                'flavor_known': flavor_known,
            }

    # NC: no charged lepton of the interacting flavor. A lepton of ANOTHER flavor
    # may still be present -- kept visible rather than dropped (see docstring).
    mismatched_lepton = next((lepton for lepton in lepton_by_flavor.values() if lepton is not None), None)
    return {
        'interaction_type': 'NC',
        'interaction_channel': 'NC',
        'primary_lepton': None,
        'lepton_flavor_mismatch': mismatched_lepton is not None,
        'mismatched_lepton': mismatched_lepton,
        'flavor_known': flavor_known,
    }


_UNSET = object()   # so an explicit None can still mean "do not set the flag"


def build_neutrino_vertex_records(mc_records, clusters_true, file_name, event, event_key=None,
                                  x_min=_UNSET, x_max=_UNSET, y_min=_UNSET, y_max=_UNSET,
                                  z_min=_UNSET, z_max=_UNSET,
                                  clusters_true_precut=None, min_cluster_energy=None):
    """
    One record per TRUE NEUTRINO INTERACTION in an event, built from mc.json's
    interaction-vertex root nodes (flatten_mc_tree records with
    is_interaction_vertex=True), joined to the true cluster that interaction
    produced.

    The join is by nu_idx, not by position: reassign_cluster_ID_true_charge_light
    gives interaction nu_idx the cluster_id 99990+nu_idx, and mc.json's root text
    carries the same nu_idx -- so the two sides link exactly, with no spatial
    matching and no tolerance to tune.

    VERTEX: the root node's start_xyz. For a root, start == end (it is a point,
    not a track), in the same cm frame as the true points -- verified by
    measuring in-volume vertices against their own cluster's deposits (agreement
    to ~0.03-0.13 cm).

    CC/NC: interaction_type / interaction_channel / primary_lepton /
    lepton_flavor_mismatch / mismatched_lepton come from
    classify_neutrino_interaction, applied to this interaction's flavor and its
    FIRST list of daughters (the direct children of the vertex node, also
    recorded as daughter_particles / n_daughters). A numu with a muon there is
    numu CC, a nue with an electron is nue CC, no muon and no electron is NC; a
    charged lepton of the OTHER flavor is NC with the mismatch flag set.

    ENERGY -- read this before using any energy from here:
      - cluster_energy_MeV is the TRUE cluster energy summed from the
        sed-sce_drift_smear_readout points (column 5), i.e. the same quantity
        apply_energy_cutoff and every completeness plot use. THIS is the energy for
        evaluation and selection.
      - mc_total_energy_MeV ('Etot') and mc_edep_MeV ('Edep') are copied from
        mc.json's root text for reference only -- Etot is the incident neutrino's
        total energy (not deposited anywhere), Edep is mc.json's own deposited
        figure. Neither is used for cuts and neither should be: they come from a
        different bookkeeping than the point cloud the pipeline measures.

    A neutrino interaction can have NO true cluster (has_true_cluster=False):
    an out-of-volume interaction may deposit nothing in the active volume, and a
    cluster may also have been removed by the pipeline's cuts. cluster_energy_MeV
    and n_true_points therefore describe the cluster AS PASSED IN -- if
    clusters_true is post-cut (as in the main pipeline), so are these numbers.

    Pass clusters_true_precut (the same clusters BEFORE the cuts) to get the
    diagnostics for those removed interactions: precut_energy_MeV,
    precut_n_points and removal_reason, which distinguishes "never deposited
    anything" from "deposited, but the cuts took it" -- the two need completely
    different follow-up. removal_reason names the energy cut specifically when
    min_cluster_energy is given and the pre-cut energy falls below it; anything
    that had enough energy and still vanished is attributed to the geometric
    cuts (fiducial / min-points). Records that survived carry
    removal_reason=None.

    NOTE on the dead-area cut: it is no longer one of the geometric cuts counted
    here. It is applied upstream by preprocess_deadarea_cut.py, before this
    pipeline sees the data, so clusters_true_precut is ALREADY dead-area-filtered
    and a dead-area removal can never appear as a removal_reason. That is
    deliberate rather than a gap: points inside a dead channel region could never
    have been reconstructed, so they are not "cut" in any meaningful sense -- they
    are outside the measurable volume, and "no true deposits" is the honest
    description of an interaction that only ever deposited there. If you point the
    pipeline back at a raw tree with Apply_deadarea_cut=True, the cut moves back to
    the end of the chain and dead-area removals fold into the geometric category
    again.

    Parameters:
    - mc_records: flatten_mc_tree(result['mc']) output
    - clusters_true: Dict {cluster_id: points}, points columns
        [x, y, z, cluster_id, q_true, energy, time]
    - file_name, event, event_key: identification, as elsewhere
    - x_min..z_max: volume bounds for the vertex_in_volume flag. DEFAULTS to the
        FIDUCIAL volume (selections.Fiducial_*), which is the signal definition:
        in-volume vs out-of-volume everywhere downstream means "the VERTEX is
        inside the fiducial volume of EITHER TPC". The test is
        selections.in_fiducial_volume, which is TPC-aware -- these bounds give
        the outer edges, and the cathode region in the middle (|x| < 4 cm) is
        excluded on top of them, belonging to neither drift volume. Those bounds apply to the vertex only -- the
        true points are cut by the wire-readout sensitive volume, not by this.
        Pass them explicitly only to ask a different question; passing an explicit
        None for any one of them disables the flag (it is left None) rather than
        guessing.

    Returns:
        List of dicts, one per neutrino interaction found in mc.json
    """
    if event_key is None:
        event_key = f"{file_name}_{event}"
    if not mc_records:
        return []

    # Unspecified bounds fall back to the FIDUCIAL volume -- the signal
    # definition, kept in selections.py so every caller flags vertices against
    # the same surface. Imported here rather than at module scope to keep
    # metadata.py importable without the selections module.
    from selections import (Fiducial_X_MIN, Fiducial_X_MAX,
                            Fiducial_Y_MIN, Fiducial_Y_MAX,
                            Fiducial_Z_MIN, Fiducial_Z_MAX)
    defaults = (Fiducial_X_MIN, Fiducial_X_MAX, Fiducial_Y_MIN,
                Fiducial_Y_MAX, Fiducial_Z_MIN, Fiducial_Z_MAX)
    bounds = tuple(default if given is _UNSET else given
                   for given, default in
                   zip((x_min, x_max, y_min, y_max, z_min, z_max), defaults))
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    have_bounds = all(b is not None for b in bounds)

    # First list of daughters, per interaction: the DIRECT children of each
    # vertex node (parent_trackid == that node's trackid), which is what the
    # CC/NC rule reads. Grandchildren are deliberately not folded in -- a muon
    # from a pion decay several steps down says nothing about the interaction.
    daughters_by_parent = {}
    for mc in mc_records:
        parent_trackid = mc.get('parent_trackid')
        if parent_trackid is not None:
            daughters_by_parent.setdefault(parent_trackid, []).append(mc)

    records = []
    for mc in mc_records:
        if not mc.get('is_interaction_vertex'):
            continue
        vertex = mc.get('start_xyz')
        nu_idx = mc.get('nu_idx')

        daughters          = daughters_by_parent.get(mc.get('trackid'), [])
        daughter_particles = sorted(d.get('particle') for d in daughters if d.get('particle'))
        interaction        = classify_neutrino_interaction(daughter_particles, flavor=mc.get('particle'))

        if vertex is not None and have_bounds:
            vx, vy, vz = vertex
            # TPC-AWARE, not a single box. SBND is two drift volumes either side
            # of a cathode at x = 0, so a vertex is in volume when it is fiducial
            # in EITHER of them, and the cathode region |x| < 4 cm belongs to
            # neither -- see selections.in_fiducial_volume.
            from selections import in_fiducial_volume
            in_volume = bool(in_fiducial_volume(vx, vy, vz,
                                                x_min=x_min, x_max=x_max,
                                                y_min=y_min, y_max=y_max,
                                                z_min=z_min, z_max=z_max))
        else:
            in_volume = None

        cluster_id = 99990.0 + nu_idx if nu_idx is not None else None
        points = clusters_true.get(cluster_id) if cluster_id is not None else None
        if points is not None and len(points) > 0:
            points = np.asarray(points)
            cluster_energy = float(points[:, 5].sum())
            n_true_points = int(len(points))
            has_cluster = True
        else:
            cluster_energy, n_true_points, has_cluster = None, 0, False

        precut_energy, precut_n_points, removal_reason, removal_category = None, 0, None, None
        if clusters_true_precut is not None and cluster_id is not None:
            precut_points = clusters_true_precut.get(cluster_id)
            if precut_points is not None and len(precut_points) > 0:
                precut_points = np.asarray(precut_points)
                precut_energy = float(precut_points[:, 5].sum())
                precut_n_points = int(len(precut_points))
        if not has_cluster:
            # removal_category is a FIXED string for grouping; removal_reason adds
            # the per-interaction detail. Keeping them separate matters: folding the
            # energy value into the grouping key makes every energy-cut row its own
            # category and the summary table degenerates into one row per cluster.
            if precut_n_points == 0:
                removal_category = "no true deposits"
                removal_reason   = "no true deposits (nothing to cut)"
            elif min_cluster_energy is not None and precut_energy is not None and precut_energy < min_cluster_energy:
                removal_category = "below energy cut"
                removal_reason   = f"below energy cut ({precut_energy:.1f} < {min_cluster_energy} MeV)"
            else:
                # No "dead area" here: it is applied upstream, before this pipeline
                # sees the data -- see the docstring's NOTE.
                removal_category = "removed by geometric cuts"
                removal_reason   = "removed by geometric cuts (fiducial / min points)"

        records.append({
            'file_name': file_name,
            'event': event_key,
            'event_num': event,
            'nu_idx': nu_idx,
            'cluster_id': cluster_id,
            'flavor': mc.get('particle'),
            # CC/NC from the interacting flavor plus the first list of daughters
            # -- see classify_neutrino_interaction for the rule, and for the
            # wrong-flavor-lepton case that lands in NC with a flag on it.
            'interaction_type': interaction['interaction_type'],
            'interaction_channel': interaction['interaction_channel'],
            'primary_lepton': interaction['primary_lepton'],
            'lepton_flavor_mismatch': interaction['lepton_flavor_mismatch'],
            'mismatched_lepton': interaction['mismatched_lepton'],
            'daughter_particles': daughter_particles,
            'n_daughters': len(daughters),
            'vertex_x': vertex[0] if vertex else None,
            'vertex_y': vertex[1] if vertex else None,
            'vertex_z': vertex[2] if vertex else None,
            'vertex_in_volume': in_volume,
            'mc_total_energy_MeV': mc.get('total_energy_MeV'),
            'mc_edep_MeV': mc.get('energy_MeV'),
            'cluster_energy_MeV': cluster_energy,
            'n_true_points': n_true_points,
            'has_true_cluster': has_cluster,
            'precut_energy_MeV': precut_energy,
            'precut_n_points': precut_n_points,
            'removal_reason': removal_reason,
            'removal_category': removal_category,
        })
    return records


def build_neutrino_volume_map(vertex_records):
    """
    {(event_key, cluster_id): 'in' | 'out'} for the true neutrino interactions in
    build_neutrino_vertex_records' output, so any record list keyed by
    (event, true cluster id) can be split by where the interaction vertex sits.

    The key is exact, not spatial: reassign_cluster_ID_true_charge_light gives
    interaction nu_idx the cluster_id 99990+nu_idx, and that same id is what
    completeness/purity/metadata records carry as 'true_cluster_id'.

    Interactions whose vertex_in_volume is None -- no vertex in mc.json, or no
    volume bounds passed to build_neutrino_vertex_records -- are LEFT OUT rather
    than guessed into one side, so they fall out of both subsets instead of
    silently inflating one.

    COSMIC clusters never appear here: they have no mc.json interaction and hence
    no vertex record. Filtering by this map therefore keeps true neutrinos only,
    which is exactly what the in/out-of-volume evaluation roots want.

    Parameters:
    - vertex_records: build_neutrino_vertex_records output (one dict per
      interaction), at any aggregation level -- event, file or job

    Returns:
        Dict {(event_key, cluster_id): 'in'|'out'}
    """
    volume_map = {}
    for record in vertex_records or []:
        cluster_id = record.get('cluster_id')
        in_volume  = record.get('vertex_in_volume')
        if cluster_id is None or in_volume is None:
            continue
        volume_map[(record['event'], cluster_id)] = 'in' if in_volume else 'out'
    return volume_map


def build_neutrino_channel_map(vertex_records):
    """
    {(event_key, cluster_id): 'numu_CC' | 'nue_CC' | 'NC'} for the true neutrino
    interactions in build_neutrino_vertex_records' output -- the CC/NC
    counterpart of build_neutrino_volume_map, keyed identically, so the same
    record lists can be split by interaction channel instead of by vertex volume
    with the same filter.

    See classify_neutrino_interaction for how the channel is decided. Like the
    volume map, this contains neutrinos only (cosmic clusters have no mc.json
    interaction), and interactions with no channel are left out rather than
    guessed.

    Parameters:
    - vertex_records: build_neutrino_vertex_records output, at any level

    Returns:
        Dict {(event_key, cluster_id): channel}
    """
    channel_map = {}
    for record in vertex_records or []:
        cluster_id = record.get('cluster_id')
        channel    = record.get('interaction_channel')
        if cluster_id is None or channel is None:
            continue
        channel_map[(record['event'], cluster_id)] = channel
    return channel_map


def restrict_label_map(label_map, restrict_map, restrict_label):
    """
    One label map narrowed to the entries another map labels a given way, so two
    splits can be composed: restricting the channel map to the volume map's 'in'
    entries gives the numu CC / nue CC / NC labels OF the in-volume neutrinos.

    Both maps are keyed the same way ((event, true cluster id)), so this is a
    plain key intersection -- no matching, no tolerance.

    Parameters:
    - label_map: the map whose labels are kept (e.g. build_neutrino_channel_map)
    - restrict_map: the map that selects which keys survive (e.g.
      build_neutrino_volume_map)
    - restrict_label: the value restrict_map must hold for a key to survive

    Returns:
        A new dict; the inputs are not modified
    """
    return {key: label for key, label in label_map.items()
            if restrict_map.get(key) == restrict_label}


def filter_records_by_label(records, label_map, label, id_key='true_cluster_id'):
    """
    The records belonging to one labelled population, for re-rendering an
    already-computed evaluation per population without recomputing anything.

    The label is whatever the map holds -- 'in'/'out' from
    build_neutrino_volume_map, 'numu_CC'/'nue_CC'/'NC' from
    build_neutrino_channel_map -- so one filter serves every split.

    Nothing is recalculated here: completeness, purity and every matching decision
    were made against the FULL true and reco populations, and this only selects
    which of those finished records a given output root shows. In particular the
    reco side is never cut -- a purity value kept here still reflects the cosmic
    contamination in its reco cluster, which is the number worth reading.

    Parameters:
    - records: any list of dicts carrying an 'event' key and a true-cluster id
    - label_map: build_neutrino_volume_map / build_neutrino_channel_map output
    - label: the population to keep, or 'all' to return the list unchanged
    - id_key: the record's true-cluster id field -- 'true_cluster_id' for
      completeness / purity / metadata / pair / 1-to-many records, 'cluster_id' for
      build_true_cluster_type_records and build_neutrino_vertex_records output

    Returns:
        A new list (the input is never mutated)

    Note which rows disappear for any label other than 'all', all deliberately:
      - cosmic true clusters, which have no vertex record at all
      - EvaluatePurity's unmatched-reco rows (true_cluster_id=8888), which
        describe reco clusters that touched no true cluster and so belong to no
        neutrino population
    An unmatched true NEUTRINO is kept: EvaluateCompleteness's unmatched row is
    keyed by the neutrino's own cluster id (only its reco_cluster_id is the 8888
    sentinel), so a neutrino that reconstructed to nothing stays in the
    completeness denominator at 0.
    """
    if label == 'all':
        return list(records or [])
    return [r for r in (records or [])
            if label_map.get((r.get('event'), r.get(id_key))) == label]


# The name this was introduced under, when the only split was by vertex volume.
# Kept so existing callers keep working; new code can use either.
filter_records_by_volume = filter_records_by_label
