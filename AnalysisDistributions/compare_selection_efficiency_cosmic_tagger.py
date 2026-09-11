"""
Overlay the 1D signal selection efficiency curves from a BEFORE-cosmic-tagger
SignalBackground_Distributions run and an AFTER-cosmic-tagger one, so the two
can be read off one plot instead of two separate directory trees.

WHY FROM TEXT FILES, NOT FROM THE PLOTTING CODE DIRECTLY. draw_selection_
performance.draw_selection_efficiency() never persists the arrays it plots --
only the PNG -- so there is nothing to load back for an old run. Two text files
ARE written for exactly this:
  efficiency.txt              (write_efficiency_summary) -- always written,
      default threshold only (completeness_and_purity_gt_80pc), finest bin
      width (100 MeV here).
  efficiency_by_threshold.txt (write_efficiency_by_threshold_summary) -- added
      alongside it once this comparison needed a threshold efficiency.txt does
      not cover; written per (bin width, rebinning) actually drawn, i.e. here
      at 100 and 200 MeV, tail_1bin_above_1000MeV rebinning, EVERY entry in
      EFFICIENCY_THRESHOLDS (completeness_and_purity_gt_50/60/70/80pc,
      completeness_gt_60pc, completeness_gt_60pc_purity_gt_80pc).
Both runs had to be RE-RUN once to produce efficiency_by_threshold.txt (it did
not exist before); from here on both files exist for any new run, so no
further re-run is needed to extend this comparison again.

TWO COMPARISON SETS, mirroring the two text files:
  100 MeV, default threshold only -- from efficiency.txt, unchanged since this
    script's first version.
  200 MeV, EVERY threshold, tail_1bin_above_1000MeV binning (edges 0-200,
    200-400, ..., 800-1000, then one bin for everything above 1000 -- the
    binning EFFICIENCY_BINNINGS_SELECTED already draws at 200 MeV) -- from
    efficiency_by_threshold.txt, one comparison per threshold, mirroring the
    combinations under selection_efficiency/200MeV/<channel>/
    tail_1bin_above_1000MeV/with_uncertainty_band/ in either run.

In both sets only the SINGLE-LINE curves are drawn: "high signal" (completeness
AND purity above the threshold) and "good+bad" (any in-volume pair, any
quality -- identical at every threshold, drawn once per threshold folder anyway
so the folder is self-contained) -- EFFICIENCY_CURVE_SETS' 'high' and 'goodbad'
tags. The multi-curve figures ('both', 'relaxed_signal', 'relaxed_signal_all')
and the single-curve 'relaxed' one need a numerator neither text file has
(purity-relaxed) and are still skipped.

UNCERTAINTY. Both runs' single-line PNGs already draw a Clopper-Pearson band
(the only uncertainty style either job drew -- UNCERTAINTY_STYLES_DRAWN =
('band',) in SignalBackground_Distributions.ipynb), so this script recomputes
the SAME interval (draw_selection_performance.clopper_pearson, same 68.27% CL)
from the numerator/denominator counts the text files report per bin -- not a
new choice, just reusing the one this codebase already draws everywhere else.

OUTPUT layout mirrors selection_efficiency/ as far as the scope above reaches:
  <OUTPUT_DIR>/100MeV/<channel>/completeness_and_purity_gt_80pc/
      selection_efficiency_compare_<channel>_{high,goodbad}_job_Combined.png
  <OUTPUT_DIR>/200MeV/<channel>/<threshold_dirname>/          (x6 thresholds)
      selection_efficiency_compare_<channel>_{high,goodbad}_job_Combined.png
  <OUTPUT_DIR>/comparison_efficiency.txt   -- integrated before/after numbers,
      same shape as efficiency.txt's summary table, for every case above

Run directly: python3 compare_selection_efficiency_cosmic_tagger.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# This script lives in AnalysisDistributions/, one level below the repository
# root where DrawRecoTrueFlashes.py etc. live -- draw_selection_performance.py
# imports from there, so the repo root needs to be on sys.path first, exactly
# like every notebook in this directory (see their sys.path-setup cell).
_NB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _NB_DIR.parent
for _path in (str(_REPO_ROOT), str(_NB_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from draw_selection_performance import (
    CHANNELS, EFFICIENCY_THRESHOLDS, HIGH_SIGNAL_THRESHOLD, MIN_TRUE_ENERGY_MEV,
    PLOT_X_MAX_MEV, ENERGY_AXIS_TICK_MEV, EFFICIENCY_CL, clopper_pearson,
    threshold_dirname, threshold_label,
    _AXIS_LABEL_FONTSIZE, _TITLE_FONTSIZE, _TICK_LABEL_FONTSIZE, _LEGEND_FONTSIZE,
)
from draw_signal_background import set_fitted_title

# ============================================================================
# CONFIG
# ============================================================================
# SAMPLE_NAME: the per-sample subdirectory the SignalBackground notebooks now
# write into (Signal_Background_Distributions[_BeforeCosmicTagger]/<SAMPLE_NAME>/)
# and that this comparison writes its own output into. Match it to the notebooks.
#   "NuECC_Sample"  -- the nue CC production
#   "NuMuCC_Sample" -- the numu beam production
SAMPLE_NAME = "NuMuCC_Sample"

_PLOTS = Path("/Users/prabhjotsingh/Experiments/SBND/WireCell_Reconstruction/"
              "AnalysisDistributions/multi_file_plots_charge_light_matching")

# AFTER: the SignalBackground_Distributions.ipynb run (cosmic tagger cut applied).
# BEFORE: the SignalBackground_Distributions_BeforeCosmicTagger.ipynb run.
# Both auto-discover the latest combined_apa_* run under their SAMPLE_NAME
# subdirectory, so a re-run is picked up without editing these paths. Set
# AFTER_DIR / BEFORE_DIR directly to pin a specific older run instead.
AFTER_BASE_DIR  = _PLOTS / "Signal_Background_Distributions" / SAMPLE_NAME
AFTER_DIR = None   # set to a job_summary/selection_efficiency path to override auto-discovery

BEFORE_BASE_DIR = _PLOTS / "Signal_Background_Distributions_BeforeCosmicTagger" / SAMPLE_NAME
BEFORE_DIR = None   # set to a job_summary/selection_efficiency path to override auto-discovery

OUTPUT_DIR = _PLOTS / "Compare_Selection_Efficiency_CosmicTagger" / SAMPLE_NAME

# The by-threshold comparison set (200 MeV, every threshold) reads
# efficiency_by_threshold.txt from this bin width / rebinning -- 0-200, 200 MeV
# steps to 1000, one bin above 1000, exactly what the user asked for and exactly
# what selection_efficiency/200MeV/<channel>/tail_1bin_above_1000MeV/ already
# draws at every threshold, so the two are directly comparable.
BY_THRESHOLD_BIN_WIDTH_DIR = "200MeV"
BY_THRESHOLD_BINNING_DIR = "tail_1bin_above_1000MeV"

# The two single-line curves efficiency.txt supports -- see module docstring.
# key: column names in efficiency.txt (num_col, eff_col); label: legend/box name;
# curve_tag: matches EFFICIENCY_CURVE_SETS' filename tag for the equivalent
# single-curve PNG in selection_efficiency/, so the two are easy to cross-reference.
CURVES = [
    {'key': 'high', 'label': 'Signal (completeness & purity > 80%)',
     'box_name': 'Signal', 'curve_tag': 'high'},
    {'key': 'any', 'label': 'All Selected (any in-volume pair)',
     'box_name': 'All Selected', 'curve_tag': 'goodbad'},
]

# Before/after styling -- colour alone tells the two runs apart (which curve is
# which is already the plot's title/filename), marker+linestyle repeat it so it
# still reads in black and white.
_RUN_STYLE = {
    'before': {'color': 'tab:red',  'marker': 's', 'linestyle': '--', 'alpha_line': 0.9, 'zorder': 3},
    'after':  {'color': 'tab:blue', 'marker': 'o', 'linestyle': '-',  'alpha_line': 1.0, 'zorder': 4},
}


def find_latest_run(base_dir, glob='combined_apa_*'):
    """Latest (lexicographically last -- the timestamp sorts correctly) run
    directory under base_dir, its job_summary/selection_efficiency path."""
    candidates = sorted(p for p in Path(base_dir).glob(glob) if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no {glob!r} run directory under {base_dir}")
    return candidates[-1] / "job_summary" / "selection_efficiency"


# ============================================================================
# PARSE efficiency.txt
# ============================================================================
_OVERALL_ROW_RE = re.compile(
    r'^\s*(?P<channel>\S+)\s+(?P<denom>\d+)\s+(?P<high>\d+)\s+(?P<eff_high>[\d.]+|nan)'
    r'\s+(?P<any>\d+)\s+(?P<eff_any>[\d.]+|nan)\s*$')
_BIN_HEADER_RE = re.compile(r'^(?P<channel>\S+) -- per (?P<width>\d+) MeV bin of true deposited energy$')
_BIN_ROW_RE = re.compile(
    r'^\s*(?P<low>\d+)\s*-\s*(?P<high_edge>\d+)\s+(?P<denom>\d+)\s+(?P<high>\d+)'
    r'\s+(?P<eff_high>[\d.]+|nan)\s+(?P<any>\d+)\s+(?P<eff_any>[\d.]+|nan)\s*$')


def parse_efficiency_txt(path):
    """
    Returns {'overall': {channel: {'denominator', 'high', 'eff_high', 'any',
    'eff_any'}}, 'bins': {channel: {'low': [...], 'high': [...], 'denominator':
    np.array, 'high': np.array, 'any': np.array}}} -- 'bins' arrays are the
    per-100MeV-bin table (default threshold, finest width); 'overall' is the
    job-integrated summary table (covers every energy, not just the binned
    range -- see write_efficiency_summary's own note on the two differing).
    """
    lines = Path(path).read_text().splitlines()

    overall = {}
    channels_seen = set()
    for i, line in enumerate(lines):
        m = _OVERALL_ROW_RE.match(line)
        if m and m.group('channel') in CHANNELS:
            overall[m.group('channel')] = {
                'denominator': int(m.group('denom')),
                'high': int(m.group('high')), 'eff_high': float(m.group('eff_high')),
                'any': int(m.group('any')), 'eff_any': float(m.group('eff_any')),
            }
            channels_seen.add(m.group('channel'))
        if len(channels_seen) == len(CHANNELS):
            break
    if not channels_seen:
        raise ValueError(f"{path}: overall summary table has no recognised channel rows")
    # A channel legitimately absent from a sample (e.g. no nue_CC in a numu run)
    # simply does not appear here; the comparison loops skip channels that are
    # not present -- with real data -- in both runs.

    bins = {}
    channel = None
    rows = None
    for line in lines:
        m = _BIN_HEADER_RE.match(line.strip())
        if m:
            channel = m.group('channel')
            rows = {'low': [], 'high': [], 'denominator': [], 'high_num': [], 'any_num': []}
            bins[channel] = rows
            continue
        if rows is None:
            continue
        if line.strip().startswith('TOTAL'):
            channel, rows = None, None
            continue
        m = _BIN_ROW_RE.match(line)
        if m:
            rows['low'].append(int(m.group('low')))
            rows['high'].append(int(m.group('high_edge')))
            rows['denominator'].append(int(m.group('denom')))
            rows['high_num'].append(int(m.group('high')))
            rows['any_num'].append(int(m.group('any')))

    # Same tolerance as the overall table: a channel with no interactions has no
    # per-bin block. Callers guard on channel presence.
    for channel, rows in bins.items():
        for key in ('low', 'high', 'denominator', 'high_num', 'any_num'):
            rows[key] = np.array(rows[key])

    return {'overall': overall, 'bins': bins}


_BY_THRESHOLD_HEADER_RE = re.compile(r'^(?P<channel>\S+) -- .+ \[(?P<dirname>[a-z0-9_]+)\]$')
_TOTAL_ROW_RE = re.compile(
    r'^\s*TOTAL \((?P<kind>binned|integrated)\)\s+(?P<denom>\d+)\s+(?P<high>\d+)'
    r'\s+(?P<eff_high>[\d.]+|nan)\s+(?P<any>\d+)\s+(?P<eff_any>[\d.]+|nan)\s*$')


def parse_efficiency_by_threshold_txt(path):
    """
    Returns {channel: {threshold_dirname: {'low', 'high', 'denominator',
    'high_num', 'any_num' (np.array, the per-bin table), 'n_denominator',
    'n_high', 'n_any', 'eff_high', 'eff_any' (the block's own TOTAL
    (integrated) row -- covers every energy, not just the binned range, same
    convention as efficiency.txt's overall summary)}} from
    write_efficiency_by_threshold_summary's output. One block per channel per
    entry in EFFICIENCY_THRESHOLDS; a block with zero denominator is never
    written (see that function), so a missing (channel, threshold) key means
    genuinely no data, not a parse failure.
    """
    lines = Path(path).read_text().splitlines()

    result = {}
    channel = None
    rows = None
    for line in lines:
        m = _BY_THRESHOLD_HEADER_RE.match(line.strip())
        if m:
            channel = m.group('channel')
            rows = {'low': [], 'high': [], 'denominator': [], 'high_num': [], 'any_num': []}
            result.setdefault(channel, {})[m.group('dirname')] = rows
            continue
        if rows is None:
            continue
        m = _TOTAL_ROW_RE.match(line)
        if m:
            if m.group('kind') == 'integrated':
                rows['n_denominator'] = int(m.group('denom'))
                rows['n_high'] = int(m.group('high'))
                rows['n_any'] = int(m.group('any'))
                rows['eff_high'] = float(m.group('eff_high'))
                rows['eff_any'] = float(m.group('eff_any'))
                channel, rows = None, None   # block is done; blank line follows
            continue
        m = _BIN_ROW_RE.match(line)
        if m:
            rows['low'].append(int(m.group('low')))
            rows['high'].append(int(m.group('high_edge')))
            rows['denominator'].append(int(m.group('denom')))
            rows['high_num'].append(int(m.group('high')))
            rows['any_num'].append(int(m.group('any')))

    for by_threshold in result.values():
        for rows in by_threshold.values():
            for key in ('low', 'high', 'denominator', 'high_num', 'any_num'):
                rows[key] = np.array(rows[key])

    return result


def comparable_channels(before, after):
    """CHANNELS that carry real data (denominator > 0) in the overall table AND
    a per-bin block in BOTH runs -- the ones a before/after overlay can be drawn
    for. A channel with no interactions in the sample (e.g. nue_CC in a numu
    run) is silently dropped."""
    out = []
    for channel in CHANNELS:
        b, a = before['overall'].get(channel), after['overall'].get(channel)
        if not b or not a or b['denominator'] == 0 or a['denominator'] == 0:
            continue
        if channel not in before['bins'] or channel not in after['bins']:
            continue
        out.append(channel)
    return out


# ============================================================================
# DRAW
# ============================================================================
def draw_run(ax, bins, curve_key, run_label, style):
    """One run's curve + Clopper-Pearson band for one channel/curve."""
    centers = (bins['low'] + bins['high']) / 2.0
    denominator = bins['denominator'].astype(float)
    numerator = bins[f'{curve_key}_num'].astype(float)
    filled = denominator > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(filled, numerator / denominator, np.nan)

    low, high = clopper_pearson(numerator[filled], denominator[filled])
    color = style['color']
    ax.fill_between(centers[filled], low, high, color=color, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(centers[filled], ratio[filled], marker=style['marker'], color=color,
            linestyle=style['linestyle'], linewidth=1.8, markersize=6,
            alpha=style['alpha_line'], label=run_label, zorder=style['zorder'])


def place_legend_and_box(fig, ax, box_lines):
    """
    Legend upper-right, box text directly below it, both confined to the
    y in (1.0, 1.25) strip above the axhline at 1.0 -- efficiency never draws
    there (it is bounded by 1, ignoring the rare over-unity bin noted in
    draw_selection_efficiency), so anything placed in that strip cannot cover a
    curve or its band no matter how the bins happen to fall. The legend's own
    height is measured with the renderer rather than guessed, so the box
    strictly clears it however many entries the legend ends up with.
    """
    legend = ax.legend(fontsize=_LEGEND_FONTSIZE, loc='upper right',
                       bbox_to_anchor=(0.995, 0.995), framealpha=0.92)
    fig.canvas.draw()  # legend needs a real renderer pass before its extent exists
    bbox_axes = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    ax.text(0.995, bbox_axes.y0 - 0.015, "\n".join(box_lines),
            transform=ax.transAxes, ha='right', va='top',
            fontsize=_LEGEND_FONTSIZE - 1, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='0.6'))


def curve_label(curve, threshold):
    """'high' names its actual threshold (varies across the by-threshold set);
    'any' ("good+bad") never depends on it, so its label is fixed."""
    if curve['key'] == 'high':
        return f"Signal ({threshold_label(threshold)})"
    return curve['label']


def draw_comparison(before_bins, after_bins, before_eff, after_eff,
                    channel, curve, threshold, out_dir):
    """
    One channel/curve/threshold's before-vs-after overlay. before_bins/
    after_bins are one {'low','high','denominator','high_num','any_num'} dict
    each (a single (channel, threshold) block from either parser above);
    before_eff/after_eff are that same block's integrated efficiency for this
    curve (a plain float, already picked out by the caller).
    """
    channel_label = {'numu_CC': r'$\nu_\mu$ CC', 'nue_CC': r'$\nu_e$ CC',
                     'NC': 'NC'}.get(channel, channel)
    key = curve['key']

    fig, ax = plt.subplots(figsize=(10, 7))
    draw_run(ax, before_bins, key, 'Before cosmic tagger', _RUN_STYLE['before'])
    # After drawn solid where before is dashed, so the two runs read as "same
    # quantity, two moments" rather than two different metrics.
    draw_run(ax, after_bins, key, 'After cosmic tagger', _RUN_STYLE['after'])

    ax.set_xlabel('True Deposited Cluster Energy (MeV)',
                  fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Selection Efficiency', fontsize=_AXIS_LABEL_FONTSIZE, fontweight='bold')
    set_fitted_title(ax, f'{channel_label} selection efficiency, vertex in volume -- '
                 f'{curve_label(curve, threshold)}\nbefore vs. after cosmic tagger cut',
                 _TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=_TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(MIN_TRUE_ENERGY_MEV, PLOT_X_MAX_MEV)
    ax.xaxis.set_major_locator(MultipleLocator(ENERGY_AXIS_TICK_MEV))
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color='black', linewidth=1.0, linestyle=':')

    box_lines = [
        f"{curve['box_name']} Selection Efficiency (before) = {100.0 * before_eff:.1f}%",
        f"{curve['box_name']} Selection Efficiency (after)  = {100.0 * after_eff:.1f}%",
    ]
    place_legend_and_box(fig, ax, box_lines)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"selection_efficiency_compare_{channel}_{curve['curve_tag']}_job_Combined.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return path


def write_comparison_summary(before, after, before_by_threshold, after_by_threshold, output_dir):
    lines = ["=" * 92,
             "SELECTION EFFICIENCY -- BEFORE vs. AFTER COSMIC TAGGER (Job Level, integrated)",
             "=" * 92, "",
             f"100 MeV set -- default threshold only (completeness & purity > "
             f"{HIGH_SIGNAL_THRESHOLD:.0%}), from efficiency.txt ({EFFICIENCY_CL:.2%} CL bands on the plots).",
             "", f"  {'channel':<10s}{'high (before)':>15s}{'high (after)':>15s}"
                 f"{'good+bad (before)':>20s}{'good+bad (after)':>19s}",
             "  " + "-" * 79]
    for channel in comparable_channels(before, after):
        b, a = before['overall'][channel], after['overall'][channel]
        lines.append(f"  {channel:<10s}{b['eff_high']:>15.4f}{a['eff_high']:>15.4f}"
                     f"{b['eff_any']:>20.4f}{a['eff_any']:>19.4f}")

    lines += ["", "-" * 92,
             "200 MeV set -- every threshold, tail_1bin_above_1000MeV binning, "
             "from efficiency_by_threshold.txt.", "-" * 92, "",
             f"  {'channel':<10s}{'threshold':<26s}{'high (before)':>15s}{'high (after)':>15s}"
             f"{'good+bad (before)':>20s}{'good+bad (after)':>19s}",
             "  " + "-" * 95]
    for channel in CHANNELS:
        for threshold in EFFICIENCY_THRESHOLDS:
            dirname = threshold_dirname(threshold)
            b = before_by_threshold.get(channel, {}).get(dirname)
            a = after_by_threshold.get(channel, {}).get(dirname)
            if not b or not a:
                continue
            lines.append(f"  {channel:<10s}{dirname:<26s}{b['eff_high']:>15.4f}{a['eff_high']:>15.4f}"
                         f"{b['eff_any']:>20.4f}{a['eff_any']:>19.4f}")

    lines += ["", "Source runs:", f"  before: {BEFORE_DIR}", f"  after:  {AFTER_DIR}"]
    path = Path(output_dir) / 'comparison_efficiency.txt'
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    global BEFORE_DIR, AFTER_DIR
    if BEFORE_DIR is None:
        BEFORE_DIR = find_latest_run(BEFORE_BASE_DIR)
    if AFTER_DIR is None:
        AFTER_DIR = find_latest_run(AFTER_BASE_DIR)
    print(f"Before (no cosmic tagger cut): {BEFORE_DIR}")
    print(f"After  (cosmic tagger cut):    {AFTER_DIR}")

    before = parse_efficiency_txt(BEFORE_DIR / 'efficiency.txt')
    after = parse_efficiency_txt(AFTER_DIR / 'efficiency.txt')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n100 MeV set (default threshold, completeness_and_purity_gt_"
          f"{HIGH_SIGNAL_THRESHOLD*100:.0f}pc):")
    channels_100 = comparable_channels(before, after)
    for channel in CHANNELS:
        if channel not in channels_100:
            print(f"  {channel:<8s}      : no data in one or both runs -- skipped")
            continue
    for channel in channels_100:
        for curve in CURVES:
            key = curve['key']
            before_eff = before['overall'][channel][f'eff_{key}']
            after_eff  = after['overall'][channel][f'eff_{key}']
            out_dir = OUTPUT_DIR / "100MeV" / channel / threshold_dirname(HIGH_SIGNAL_THRESHOLD)
            path = draw_comparison(before['bins'][channel], after['bins'][channel],
                                   before_eff, after_eff, channel, curve,
                                   HIGH_SIGNAL_THRESHOLD, out_dir)
            print(f"  {channel:<8s} {key:<5s}: before={100*before_eff:5.1f}%  "
                  f"after={100*after_eff:5.1f}%  -> {path}")

    by_threshold_filename = BEFORE_DIR / BY_THRESHOLD_BIN_WIDTH_DIR / BY_THRESHOLD_BINNING_DIR / 'efficiency_by_threshold.txt'
    before_bt = parse_efficiency_by_threshold_txt(by_threshold_filename)
    after_bt = parse_efficiency_by_threshold_txt(
        AFTER_DIR / BY_THRESHOLD_BIN_WIDTH_DIR / BY_THRESHOLD_BINNING_DIR / 'efficiency_by_threshold.txt')

    print(f"\n200 MeV set (every threshold, {BY_THRESHOLD_BINNING_DIR}):")
    for channel in CHANNELS:
        if channel not in before_bt or channel not in after_bt:
            print(f"  {channel:<8s}: no data in one or both runs -- skipped")
            continue
        for threshold in EFFICIENCY_THRESHOLDS:
            dirname = threshold_dirname(threshold)
            b_rows = before_bt.get(channel, {}).get(dirname)
            a_rows = after_bt.get(channel, {}).get(dirname)
            if b_rows is None or a_rows is None:
                print(f"  {channel:<8s} {dirname:<26s}: no data in one of the two runs -- skipped")
                continue
            for curve in CURVES:
                key = curve['key']
                out_dir = OUTPUT_DIR / BY_THRESHOLD_BIN_WIDTH_DIR / channel / dirname
                path = draw_comparison(b_rows, a_rows, b_rows[f'eff_{key}'], a_rows[f'eff_{key}'],
                                       channel, curve, threshold, out_dir)
                print(f"  {channel:<8s} {dirname:<26s} {key:<5s}: before={100*b_rows[f'eff_{key}']:5.1f}%  "
                      f"after={100*a_rows[f'eff_{key}']:5.1f}%  -> {path}")

    summary_path = write_comparison_summary(before, after, before_bt, after_bt, OUTPUT_DIR)
    print(f"\nSummary: {summary_path}")
    print(f"Output written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
