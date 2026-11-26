from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import npc_ephys
import npc_ephys.openephys
import npc_sessions
import npc_stim
import numpy as np
import pandas as pd
import rich

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

import npc_sessions_cache.plots.plot_utils as plot_utils

BARCODE_TTL_CHANNEL = npc_ephys.openephys.BARCODE_TTL_CHANNEL


def _plot_barcode_times(
    session: npc_sessions.DynamicRoutingSession,
) -> plt.Figure:
    timing_info = session.ephys_timing_data  # skips unused probes
    fig = plt.figure()
    for info in timing_info:
        (
            ephys_barcode_times,
            ephys_barcode_ids,
        ) = npc_ephys.extract_barcodes_from_times(
            on_times=info.device.ttl_sample_numbers[info.device.ttl_states > 0]
            / info.sampling_rate,
            off_times=info.device.ttl_sample_numbers[info.device.ttl_states < 0]
            / info.sampling_rate,
            total_time_on_line=info.device.ttl_sample_numbers[-1] / info.sampling_rate,
        )
        plt.plot(np.diff(ephys_barcode_times))
    return fig


def _barcode_interval_deviations(
    barcode_times: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Deviation (s) of each barcode interval from the nearest whole multiple of the median
    interval. A multiple > 1 is a missing barcode (e.g. dropped near a replayed sync interval),
    anything else is a displacement."""
    intervals = np.diff(barcode_times)
    median = np.median(intervals)
    multiples = np.maximum(np.round(intervals / median), 1)
    return np.abs(intervals - multiples * median)


def plot_barcode_intervals(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, dict] | None:
    """
    Plot barcode intervals for sync and for each probe after timing correction.

    - corrected probe times use each device's timing model (`EphysTimingInfoOnSync.to_sync_time`:
      affine fit plus samples lost by the acquisition system), not a plain sample-rate correction
    - devices that failed ephys timing QC are listed in the returned dict with their
      `status` and `reason`, and aren't plotted
    - sync barcodes inside (padded) replayed intervals are dropped, as for the timing fit, so an
      interval of 2 x 30 s is a missing barcode, not a deviation
    """
    if not session.is_sync or not session.is_ephys:
        return None
    device_barcode_dict: dict[str, dict] = {}
    nominal_AP_rate = 30000
    for info in session.ephys_timing_qc:  # every device in use, including failures
        if "NI-DAQmx" in info.device.name or "LFP" in info.device.name:
            continue
        if not getattr(info, "is_valid", True):
            device_barcode_dict[info.device.name] = {
                "status": info.status,
                "reason": info.reason,
            }
            continue
        d = info.device
        (
            ephys_barcode_times,
            ephys_barcode_ids,
        ) = npc_ephys.extract_barcodes_from_times(
            on_times=d.ttl_sample_numbers[d.ttl_states == BARCODE_TTL_CHANNEL]
            / nominal_AP_rate,
            off_times=d.ttl_sample_numbers[d.ttl_states == -BARCODE_TTL_CHANNEL]
            / nominal_AP_rate,
            total_time_on_line=d.ttl_sample_numbers[-1] / nominal_AP_rate,
        )
        raw = ephys_barcode_times
        corrected = info.to_sync_time(raw * nominal_AP_rate)
        deviations = _barcode_interval_deviations(corrected)

        device_barcode_dict[info.device.name] = {
            "status": info.status,
            "reason": info.reason,
            "barcode_times_raw": raw,
            "barcode_times_corrected": corrected,
            "max_deviation_from_median_interval": np.max(deviations),
            "n_missing_barcodes": int(
                np.sum(
                    np.maximum(
                        np.round(np.diff(corrected) / np.median(np.diff(corrected))), 1
                    )
                    - 1
                )
            ),
            "n_drops": len(info.drops),
            "total_samples_lost": info.total_samples_lost,
            **{f"qc_{k}": v for k, v in info.qc_metrics.items() if k != "note"},
        }
    if not device_barcode_dict:
        raise ValueError(f"No ephys timing data available for {session.id}")
    # same sync barcodes as used for the device timing fits (masks replayed intervals)
    barcode_times, barcodes = npc_ephys.openephys._sync_barcodes(session.sync_data)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches((8, 4))
    sync_intervals = np.diff(barcode_times)
    sync_max_deviation_from_median_interval = np.max(
        _barcode_interval_deviations(barcode_times)
    )
    sync_max_deviation_string = plot_utils.add_valence_to_string(
        f"Sync deviation: {sync_max_deviation_from_median_interval}",
        sync_max_deviation_from_median_interval,
        sync_max_deviation_from_median_interval < 0.001,
        sync_max_deviation_from_median_interval > 0.001,
    )
    rich.print(sync_max_deviation_string)

    ax[0].plot(sync_intervals, "k")
    legend = []
    for device_name, device_data in device_barcode_dict.items():
        if "barcode_times_raw" not in device_data:
            rich.print(
                f"[bold magenta]{device_name}: excluded, {device_data['reason']}[/bold magenta]"
            )
            continue
        ax[1].plot(np.diff(device_data["barcode_times_raw"]))
        ax[2].plot(np.diff(device_data["barcode_times_corrected"]))
        legend.append(device_name.split("Probe")[1])
        max_deviation = device_data["max_deviation_from_median_interval"]
        max_deviation_string = plot_utils.add_valence_to_string(
            f"{device_name}: {max_deviation}",
            max_deviation,
            max_deviation < 0.001,
            max_deviation > 0.001,
        )

        rich.print(max_deviation_string)

    ax[2].plot(sync_intervals, "k")
    ax[2].legend(legend + ["sync"])
    ax[0].set_title("sync")
    ax[1].set_title("probe")
    ax[2].set_title("probe corrected")
    ax[1].set_xlabel("barcode number")
    ax[0].set_ylabel("barcode interval (s)")

    plt.tight_layout()

    for k, v in device_barcode_dict.items():
        device_barcode_dict[k] = {
            k2: v2 for k2, v2 in v.items() if "barcode_times" not in k2
        }
    return fig, device_barcode_dict


def _get_display_blocks(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[npc_stim.BlockDisplayTimes, ...]:
    """One entry per block of vsyncs on (patched) sync, with the matched stim file and the
    source of display times. Replaces pairing `sync_data.vsync_times_in_blocks` with
    `session.stim_frame_times` by position: blocks without a stim file or display times
    would shift every title."""
    return npc_stim.get_display_times(session.sync_data, session.stim_paths)


def _block_name(block: npc_stim.BlockDisplayTimes) -> str:
    if block.stim_path is None:
        return f"block {block.block_index} (no stim file)"
    return block.stim_path.stem.split("_")[0]


def _set_block_titles(
    fig: plt.Figure, blocks: Sequence[npc_stim.BlockDisplayTimes]
) -> None:
    if len(blocks) != len(fig.axes):
        return
    for ax, block in zip(fig.axes, blocks):
        ax.set_title(_block_name(block), fontsize=8)


def _print_block_sources(blocks: Sequence[npc_stim.BlockDisplayTimes]) -> None:
    for block in blocks:
        if block.is_photodiode:
            msg = f"{_block_name(block)}: photodiode"
            if block.n_interpolated:
                msg += f", {block.n_interpolated} frames interpolated inside replayed sync intervals"
            rich.print(msg)
        else:
            rich.print(
                f"[bold magenta]{_block_name(block)}: {block.source}, {block.reason}[/bold magenta]"
            )


def plot_vsync_interval_dist(
    session: npc_sessions.DynamicRoutingSession,
) -> plt.Figure | None:
    if not session.is_sync or not session.is_task:
        # no task stim to match to a vsync block (e.g. opto/RF-mapping-only sessions)
        return None
    for block in _get_display_blocks(session):
        if (
            block.stim_path is not None
            and session.task_stim_name in block.stim_path.stem
        ):
            vsync_block = block.vsync_times
            break
    else:
        raise ValueError(
            f"No vsync block matched to a stim file containing {session.task_stim_name} for {session.id}"
        )

    fig, ax = plt.figure(figsize=(4, 4)), plt.gca()
    fig.set_size_inches(5, 4)
    xlim = 1000 * 2 / 60
    ax.hist(np.diff(vsync_block) * 1000, bins=np.arange(0, xlim, xlim / 200))
    n_outliers = int(np.sum(np.diff(vsync_block) * 1000 > xlim))
    # ax.set_yscale("log")
    ax.axvline(1 / 60, c="k", ls="dotted")
    ax.set_title(
        f"vsync intervals around expected 1/60s ({100 * n_outliers/len(vsync_block):.2f}% ({n_outliers}) intervals > {xlim:.2f})\n{session.task_stim_name}\nphotodiode available = {session.is_photodiode}",
        fontsize=7,
    )
    ax.set_xlabel("interval length (ms)")
    ax.set_ylabel("count")
    return fig


def plot_diode_flip_intervals(
    session: npc_sessions.DynamicRoutingSession,
) -> plt.Figure | None:
    if not session.is_sync:
        return None
    fig = npc_stim.plot_diode_measured_sync_square_flips(session.sync_data)
    _set_block_titles(fig, _get_display_blocks(session))
    fig.set_size_inches(12, 6)
    return fig


def _plot_intervals_in_blocks(
    times_per_block: Sequence[npt.NDArray[np.floating]],
    names: Sequence[str],
    expected_period: float,
    ylabel: str,
    interpolated_per_block: Sequence[npt.NDArray[np.bool_] | None] | None = None,
) -> plt.Figure:
    """Stem plot of intervals between successive times, one panel per block. Where
    `interpolated_per_block` is given, intervals touching a reconstructed time are marked."""
    num_per_block = np.array([len(_) for _ in times_per_block])
    fig, _ = plt.subplots(
        1,
        len(times_per_block),
        sharey=True,
        gridspec_kw={"width_ratios": num_per_block / min(num_per_block)},
        squeeze=False,
    )
    axes = fig.axes
    fig.suptitle(f"{ylabel}s, {expected_period = } s")
    y_deviations_from_expected_period: list[float] = []
    for idx, (ax, d) in enumerate(zip(axes, times_per_block)):
        # add horizontal line at expected period
        ax.axhline(expected_period, linewidth=0.5, c="k", linestyle="--", alpha=0.3)
        plt.sca(ax)
        intervals = np.diff(d)
        times = np.diff(d) / 2 + d[:-1]  # plot at mid-point of interval
        markerline, stemline, baseline = plt.stem(
            times, intervals, bottom=expected_period
        )
        plt.setp(stemline, linewidth=0.5, alpha=0.3)
        plt.setp(markerline, markersize=0.5, alpha=0.8)
        plt.setp(baseline, visible=False)

        interpolated = (
            None if interpolated_per_block is None else interpolated_per_block[idx]
        )
        if interpolated is not None and interpolated.any():
            touches = interpolated[:-1] | interpolated[1:]
            ax.scatter(
                times[touches],
                intervals[touches],
                s=6,
                c="r",
                marker="x",
                linewidths=0.5,
                label="interpolated",
                zorder=3,
            )
            ax.legend(fontsize=6)

        y_deviations_from_expected_period.append(max(intervals - expected_period))
        y_deviations_from_expected_period.append(max(expected_period - intervals))
        ax.set_title(names[idx], fontsize=8)
        ax.set_xlabel("time (s)")
        if idx == 0:
            ax.set_ylabel(f"{ylabel} (s)")
        ax.set_xlim(min(d) - 20, max(d) + 20)

    for ax in axes:
        # after all ylims are established
        ax.set_ylim(
            bottom=max(
                0,
                expected_period - np.max(np.abs(y_deviations_from_expected_period)),
            ),
        )
        ticks_with_period = sorted(set(ax.get_yticks()) | {expected_period})
        ax.set_yticks(ticks_with_period)
    axes[0].set_yticklabels([f"{_:.3f}" for _ in ticks_with_period])
    fig.set_layout_engine("tight")
    fig.set_size_inches(12, 6)
    return fig


def plot_vsync_intervals(
    session: npc_sessions.DynamicRoutingSession,
) -> plt.Figure | None:
    if not session.is_sync:
        return None
    sync = session.sync_data
    blocks = _get_display_blocks(session)
    expected_period = 1 / npc_stim.get_expected_diode_flip_rate(sync)
    return _plot_intervals_in_blocks(
        [b.vsync_times for b in blocks],
        [_block_name(b) for b in blocks],
        expected_period,
        "vsync interval",
    )


def plot_frametime_intervals(
    session: npc_sessions.DynamicRoutingSession,
) -> plt.Figure | None:
    """Intervals between frame display times, for each stim with display times.

    Blocks without display times (no stim file, photodiode unusable, or overlapping a
    replayed interval that couldn't be reconstructed) are listed on stdout with their
    reason. Intervals touching a frame reconstructed by the sync patch are marked."""
    if not session.is_sync:
        return None
    sync = session.sync_data
    expected_period = 1 / npc_stim.get_expected_diode_flip_rate(sync)

    blocks = _get_display_blocks(session)
    _print_block_sources(blocks)
    blocks_with_times = tuple(b for b in blocks if b.display_times is not None)
    if not blocks_with_times:
        raise ValueError(f"No stim blocks with frame display times for {session.id}")
    return _plot_intervals_in_blocks(
        [b.display_times for b in blocks_with_times],
        [_block_name(b) for b in blocks_with_times],
        expected_period,
        "frametime interval",
        [
            None if b.frame_source is None else b.frame_source == 1
            for b in blocks_with_times
        ],
    )


def plot_sync_timing_report(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, str] | None:
    """Replayed ('invalid') sync intervals and how each stim block was handled: the
    per-block status from `session.sync_timing_report`, and the fraction of frames in each
    stim with interpolated (reconstructed) display times."""
    if not session.is_sync:
        return None
    report = session.sync_timing_report
    invalid = session.sync_data.invalid_intervals
    with pd.option_context("display.width", 250, "display.max_columns", None):
        text = f"{len(invalid)} replayed sync interval(s): {invalid}\n\n{report.to_string()}"

    sources = session.stim_frame_time_sources
    fig, ax = plt.subplots(figsize=(6, 3))
    names = [f"{k.split('_')[0]}\n{k.split('_')[-1]}" for k in sources]
    ax.bar(names, [100 * np.mean(v == 1) if len(v) else 0 for v in sources.values()])
    ax.set_ylabel("% frames interpolated")
    ax.set_title(f"{session.id}: {len(invalid)} replayed sync interval(s)", fontsize=8)
    fig.tight_layout()
    return fig, text


def _plot_vsyncs_and_diode_flips_at_ends_of_each_stim(
    session: npc_sessions.DynamicRoutingSession,
) -> plt.Figure | None:
    if not session.is_sync:
        return None
    rich.print("[bold] Fraction long frames [/bold]")
    for stim_name, stim_times in session.stim_frame_times.items():
        if isinstance(stim_times, Exception):
            continue
        intervals = np.diff(stim_times)
        fraction_long = np.sum(intervals > 0.02) / len(intervals)
        longest_interval = max(intervals)
        start_tag, end_tag = (
            ("[bold green]", "[/bold green]")
            if fraction_long < 0.01 and longest_interval < 0.5
            else ("[bold magenta]", "[/bold magenta]")
        )
        rich.print(
            start_tag
            + stim_name
            + ": "
            + str(fraction_long)
            + " \t\t longest interval:"
            + str(longest_interval)
            + end_tag
        )

    # TODO switch this to get epoch start/ stop times and plot only for good stimuli
    # blocks are titled with their matched stim file
    blocks = _get_display_blocks(session)
    fig = npc_stim.plot_stim_onsets(session.sync_data, blocks)
    fig = npc_stim.plot_stim_offsets(session.sync_data, blocks)

    return fig


def _plot_histogram_of_frame_intervals(session) -> plt.Figure:
    stim_frame_times = {
        k: v
        for k, v in session.stim_frame_times.items()
        if not isinstance(v, Exception)
    }

    fig_hist, axes_hist = plt.subplots(2, len(stim_frame_times))
    fig_hist.set_size_inches(12, 6 * len(stim_frame_times))

    for ax, (stim_name, stim_times) in zip(axes_hist, stim_frame_times.items()):
        ax.hist(np.diff(stim_times) * 1000, bins=np.arange(0, 0.1, 0.001))
        ax.set_yscale("log")
        ax.axvline(1 / 60, c="k", ls="dotted")
        ax.set_title(stim_name.split("_")[0])
        ax.set_xlabel("vsync interval (ms)")
        ax.set_ylabel("frame interval count")
    plt.tight_layout()
    return fig_hist
