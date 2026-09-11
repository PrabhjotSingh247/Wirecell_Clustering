"""
TRUE-CLUSTER SIGNAL/BACKGROUND STACKS -- driven by
AnalysisDistributions/Draw_Signal_Background_True.ipynb.

What the selected TRUE population is made of, as a stacked histogram of true
cluster energy: the signal channel, the other in-volume channels, and what came
from outside the volume. Pure truth -- no reco cluster is drawn on it, and
nothing here depends on the reco selection.

WHAT IS WRITTEN, per bin width in BIN_WIDTHS_MEV:

    signal_background_true/
        100MeV/
            signal_background_true_energy_stack_100MeV_<scale>_job_Combined.png
            signal_background_info_100MeV.txt      per-bin table + component counts
        200MeV/
            ...
        signal_background_histograms.root          both widths, one file

The ROOT file sits at the top rather than inside either width's directory because
it holds the histograms of both.

ONE SHARED Y RANGE per (scale, bin width), applied to the LOG versions only. A
narrower bin holds fewer clusters, so the tallest bin -- and therefore the axis --
differs between widths, and the headroom differs between log and linear, so a
single number cannot serve every figure. Sharing costs nothing on log, where the
bands span decades; on linear a shared top set by the tallest figure flattens the
other one, so the linear versions autoscale per figure.

THE VARIANTS. PLOT_VARIANTS in the driving notebook chooses which component lists
are drawn: the plain four-band stack, and optionally the five-band one that splits
the signal by how well each cluster was reconstructed. Only the second needs the
reco-true pairing, and only that one makes this notebook expensive -- see
NEEDS_PAIRING there.

WHY A SEPARATE MODULE. These stacks used to be drawn by
SignalBackground_Distributions.ipynb, in a loop inline in its job-level cell.
They are the truth-side half of that notebook and nothing else in it depends on
them, so they now stand alone and can be re-run without the reco-space plots --
which, with the plain stack alone, means without the pairing at all.
"""

from datetime import datetime, timedelta

from draw_signal_background import (
    BIN_WIDTHS_MEV, SIGNAL_BACKGROUND_COMPONENTS, Y_SCALES,
    draw_stacked_true_energy, shared_y_top, write_signal_background_info,
    write_signal_background_root, order_components_for_stack,
)
from draw_selection_performance import plot_directory


SIGNAL_BACKGROUND_TRUE_DIR_NAME = 'signal_background_true'


def _fmt_seconds(seconds):
    """'0:01:51 (110.6 seconds)' -- the same shape SignalBackground's summary uses."""
    if seconds is None:
        return "n/a"
    return f"{timedelta(seconds=int(seconds))} ({seconds:.1f} seconds)"


def write_signal_background_true_summary(
        output_root, job_true_var_records, *,
        total_files, total_events, job_start_time, job_runtime_s,
        staging_seconds=None, event_loop_seconds=None, draw_seconds=None,
        config_lines=None, components=None):
    """
    signal_background_true/summary.txt -- the run's configuration, the stack
    component counts (= histogram entries, recomputed from the same selectors the
    figures were drawn from) and a JOB RUNTIME block broken down into staging,
    the event loop and drawing. Returns the path written.

    config_lines is an optional list of pre-formatted 'key: value' strings from
    the notebook (parent dir, cut flags, ...) -- printed verbatim under
    "Configuration:" so the file is self-contained without this module needing to
    know the notebook's knobs.
    """
    components = components if components is not None else SIGNAL_BACKGROUND_COMPONENTS
    true_dir = plot_directory(output_root, SIGNAL_BACKGROUND_TRUE_DIR_NAME)

    component_counts = {c['key']: len(c['select'](job_true_var_records)) for c in components}
    n_neutrino = sum(1 for r in job_true_var_records if r.get('is_neutrino'))
    finish_dt = datetime.fromtimestamp(job_start_time + job_runtime_s)

    lines = []
    lines.append("=" * 80)
    lines.append("JOB SUMMARY -- SIGNAL & BACKGROUND, TRUE CLUSTERS")
    lines.append("=" * 80)
    lines.append(f"Generated: {finish_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    if config_lines:
        lines.append("")
        lines.append("Configuration:")
        lines.extend(f"  {line}" for line in config_lines)
    lines.append("")
    lines.append("=" * 80)
    lines.append("JOB-LEVEL AGGREGATION")
    lines.append("=" * 80)
    lines.append(f"Total files processed:  {total_files}")
    lines.append(f"Total events processed: {total_events}")
    lines.append(f"Total selected true clusters: {len(job_true_var_records)}")
    lines.append(f"  of which true neutrino clusters: {n_neutrino}")
    lines.append("")
    lines.append("Stack components (= histogram entries, bottom of the stack first):")
    for component in order_components_for_stack(components):
        lines.append(f"  {component['key']:<28s} {component_counts[component['key']]:6d} clusters")
    lines.append(f"  {'TOTAL IN STACK':<28s} {sum(component_counts.values()):6d} clusters")
    lines.append("")
    lines.append("=" * 80)
    lines.append("JOB RUNTIME")
    lines.append("=" * 80)
    lines.append(f"Job started at:  {datetime.fromtimestamp(job_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Job finished at: {finish_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total job runtime: {_fmt_seconds(job_runtime_s)}")
    lines.append("")
    lines.append("Breakdown:")
    lines.append(f"  input staging:  {_fmt_seconds(staging_seconds)}"
                 + ("   (0 on re-runs / non-nuecc samples -- data already extracted)"
                    if staging_seconds is not None and staging_seconds < 1 else ""))
    lines.append(f"  event loop:     {_fmt_seconds(event_loop_seconds)}"
                 + (f"   ({event_loop_seconds / total_events:.2f} s/event)"
                    if event_loop_seconds is not None and total_events else ""))
    lines.append(f"  drawing + I/O:  {_fmt_seconds(draw_seconds)}")
    lines.append("=" * 80)

    summary_path = true_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path

# The default figure set: the plain stack only. Mirrors PLOT_VARIANTS in the
# notebook, which is where it is normally chosen.
DEFAULT_PLOT_VARIANTS = [(None, None)]


def draw_job_signal_background_true(job_true_var_records, output_root,
                                    plot_variants=None,
                                    bin_widths=BIN_WIDTHS_MEV,
                                    y_scales=Y_SCALES,
                                    level_name="Job Level",
                                    filename_prefix="job", apa="Combined",
                                    title="True Signal and Backgrounds"):
    """
    Every stack, table and the ROOT file, into SIGNAL_BACKGROUND_TRUE_DIR_NAME
    under output_root.

    title is the base plot title on every figure ("True Signal and Backgrounds"
    by default); a variant_label is still appended to it where one applies.

    Returns (root_path, counts_by_variant): the ROOT file written, and
    {(variant label, bin width): {component key: n clusters}} -- the histogram
    entry counts, taken from the same mapping each figure was drawn from rather
    than recomputed, so they cannot disagree with what was plotted.
    """
    plot_variants = plot_variants if plot_variants is not None else DEFAULT_PLOT_VARIANTS

    # One shared top per (scale, width) -- see the module docstring for why it is
    # keyed on both and applied to log only.
    job_y_top = {(scale, width): shared_y_top(job_true_var_records, [],
                                              bin_width=width, y_scale=scale)
                 for scale in y_scales for width in bin_widths}

    root_entries = []   # one per FIGURE-pair; the ROOT histograms do not depend on the y scale
    counts_by_variant = {}
    for variant_components, variant_label in plot_variants:
        for bin_width in bin_widths:
            true_dir = plot_directory(output_root, SIGNAL_BACKGROUND_TRUE_DIR_NAME, bin_width)
            for y_scale in y_scales:
                selected_by_key, _reco_values = draw_stacked_true_energy(
                    job_true_var_records, true_dir, level_name, filename_prefix, apa,
                    components=variant_components, bin_width=bin_width,
                    y_top=job_y_top[(y_scale, bin_width)] if y_scale == 'log' else None,
                    variant_label=variant_label, y_scale=y_scale, title=title)
                # _reco_values is empty by construction -- no reco_records were passed.

            # The table and the ROOT histograms are scale-independent but NOT
            # bin-width independent, so they are written once per (variant, width).
            write_signal_background_info(selected_by_key, true_dir, level_name,
                                         components=variant_components,
                                         bin_width=bin_width,
                                         variant_label=variant_label)
            root_entries.append({
                'dir_name': ((variant_label + '_' if variant_label else 'stack_')
                             + f'{bin_width:.0f}MeV'),
                'selected_by_key': selected_by_key,
                'reco_values': [],
                'components': variant_components,
                'bin_width': bin_width,
            })
            counts_by_variant[(variant_label, bin_width)] = {
                key: len(records) for key, records in selected_by_key.items()}

    # One file for both widths, so it sits at the top of the directory rather than
    # inside either width's.
    root_path = write_signal_background_root(
        root_entries, plot_directory(output_root, SIGNAL_BACKGROUND_TRUE_DIR_NAME))
    return root_path, counts_by_variant


def component_counts(job_true_var_records, components=None):
    """
    {component key: n clusters} for the DEFAULT four-band stack, recomputed from
    the selectors rather than taken from a drawing loop.

    The selectors are pure, so this gives exactly the counts the figures were
    drawn from -- and unlike the loop's last mapping, it belongs to the component
    list asked for rather than to whichever variant happened to run last.
    """
    components = components if components is not None else SIGNAL_BACKGROUND_COMPONENTS
    return {c['key']: len(c['select'](job_true_var_records)) for c in components}
