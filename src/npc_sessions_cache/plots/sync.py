from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.figure
import matplotlib.pyplot as plt
import npc_ephys
import npc_sessions
import numpy as np
import rich

if TYPE_CHECKING:
    pass

import npc_sessions_cache.plots.plot_utils as plot_utils


def _plot_barcode_times(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
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


def plot_barcode_intervals(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[matplotlib.figure.Figure, dict] | None:
    """
    Plot barcode intervals for sync and for each probe after sample rate
    correction
    """
    if not session.is_sync or not session.is_ephys:
        return None
    device_barcode_dict = {}
    nominal_AP_rate = 30000
    for info in session.ephys_timing_data:  # skips unused probes
        if "NI-DAQmx" in info.device.name or "LFP" in info.device.name:
            continue
        (
            ephys_barcode_times,
            ephys_barcode_ids,
        ) = npc_ephys.extract_barcodes_from_times(
            on_times=info.device.ttl_sample_numbers[info.device.ttl_states > 0]
            / nominal_AP_rate,
            off_times=info.device.ttl_sample_numbers[info.device.ttl_states < 0]
            / nominal_AP_rate,
            total_time_on_line=info.device.ttl_sample_numbers[-1] / nominal_AP_rate,
        )
        raw = ephys_barcode_times
        corrected = ephys_barcode_times * (nominal_AP_rate / info.sampling_rate) + info.start_time
        intervals = np.diff(corrected)
        max_deviation = np.max(np.abs(intervals - np.median(intervals)))

        device_barcode_dict[info.device.name] = {
            "barcode_times_raw": raw,
            "barcode_times_corrected": corrected,
            "max_deviation_from_median_interval": max_deviation,
            "max_deviation_from_30s_interval": np.max(np.abs(intervals - 30)),
        }
    if not device_barcode_dict: 
        raise ValueError(f"No ephys timing data available for {session.id}")
    barcode_rising = session.sync_data.get_rising_edges(0, "seconds")
    barcode_falling = session.sync_data.get_falling_edges(0, "seconds")
    t0 = min([min(v["barcode_times_corrected"]) for v in device_barcode_dict.values()])
    t1 = max([max(v["barcode_times_corrected"]) for v in device_barcode_dict.values()])
    barcode_times, barcodes = npc_ephys.extract_barcodes_from_times(
        barcode_rising[(barcode_rising >= t0) & (barcode_rising <= t1)],
        barcode_falling[(barcode_falling >= t0) & (barcode_falling <= t1)],
        total_time_on_line=session.sync_data.total_seconds,
    )

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches((8, 4))
    sync_intervals = np.diff(barcode_times)
    sync_max_deviation_from_median_interval = np.max(
        np.abs(sync_intervals - np.median(sync_intervals))
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


def plot_vsync_interval_dist(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure | None:
    if not session.is_sync:
        return None
    for vsync_block in session.sync_data.vsync_times_in_blocks:
        if len(vsync_block) == len(
            next(
                v
                for k, v in session.stim_frame_times.items()
                if session.task_stim_name in k
            )
        ):
            break
    else:
        raise ValueError(
            f"No block with matching stim_frame_times for {session.id} containing {session.task_stim_name}: {session.stim_frame_times.values()}"
        )

    fig, ax = plt.figure(figsize=(4, 4)), plt.gca()
    fig.set_size_inches(5, 4)
    xlim = 1000 * 2 / 60
    ax.hist(np.diff(vsync_block) * 1000, bins=np.arange(0, xlim, xlim / 200))
    n_outliers = len(np.diff(vsync_block) > xlim)
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
) -> matplotlib.figure.Figure | None:
    if not session.is_sync:
        return None
    fig = session.sync_data.plot_diode_measured_sync_square_flips()
    names = tuple(
        k for k, v in session.stim_frame_times.items() if not isinstance(v, Exception)
    )
    for idx, ax in enumerate(fig.axes):
        if len(names) == len(fig.axes):
            ax.set_title(names[idx].split("_")[0])
    fig.set_size_inches(12, 6)
    return fig

def plot_vsync_intervals(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure | None:
    if not session.is_sync:
        return None
    sync = session.sync_data
    stim_ons, stim_offs = sync.stim_onsets, sync.stim_offsets

    vsyncs_per_stim = sync.vsync_times_in_blocks

    frequency = sync.expected_diode_flip_rate
    expected_period = 1 / frequency

    num_vsyncs_per_stim = np.array([len(_) for _ in vsyncs_per_stim])
    # add ` width_ratios=num_vsyncs/min(num_vsyncs)``
    fig, _ = plt.subplots(
        1,
        len(vsyncs_per_stim),
        sharey=True,
        gridspec_kw={
            "width_ratios": num_vsyncs_per_stim / min(num_vsyncs_per_stim)
        },
    )
    fig.suptitle(
        f"vsync intervals, {expected_period = } s"
    )
    y_deviations_from_expected_period: list[float] = []
    for idx, (ax, d) in enumerate(zip(fig.axes, vsyncs_per_stim)):
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

        y_deviations_from_expected_period.append(max(intervals - expected_period))
        y_deviations_from_expected_period.append(max(expected_period - intervals))
        if len(fig.axes) > 1:
            ax.set_title(f"stim {idx}", fontsize=8)
        ax.set_xlabel("time (s)")
        if idx == 0:
            ax.set_ylabel("vsync interval (s)")
        ax.set_xlim(min(d) - 20, max(d) + 20)

    for ax in fig.axes:
        # after all ylims are established
        ax.set_ylim(
            bottom=max(
                0,
                expected_period - np.max(np.abs(y_deviations_from_expected_period)),
            ),
        )
        ticks_with_period = sorted(set(ax.get_yticks()) | {expected_period})
        ax.set_yticks(ticks_with_period)
        if idx == 0:
            ax.set_yticklabels([f"{_:.3f}" for _ in ticks_with_period])
    fig.set_layout_engine("tight")

    names = tuple(
        k for k, v in session.stim_frame_times.items() if not isinstance(v, Exception)
    )
    for idx, ax in enumerate(fig.axes):
        if len(names) == len(fig.axes):
            ax.set_title(names[idx].split("_")[0])
    fig.set_size_inches(12, 6)
    return fig


def plot_frametime_intervals(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure | None:
    if not session.is_sync:
        return None
    sync = session.sync_data
    frequency = sync.expected_diode_flip_rate
    expected_period = 1 / frequency

    # get the intervals in parts (one for each stimulus)
    frametimes_per_stim = sync.frame_display_time_blocks

    num_frametimes_per_stim = np.array([len(_) for _ in frametimes_per_stim])
    fig, _ = plt.subplots(
        1,
        len(frametimes_per_stim),
        sharey=True,
        gridspec_kw={
            "width_ratios": num_frametimes_per_stim / min(num_frametimes_per_stim)
        },
    )
    fig.suptitle(
        f"frametime intervals, {expected_period = } s"
    )
    y_deviations_from_expected_period: list[float] = []
    for idx, (ax, d) in enumerate(zip(fig.axes, frametimes_per_stim)):
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

        y_deviations_from_expected_period.append(max(intervals - expected_period))
        y_deviations_from_expected_period.append(max(expected_period - intervals))
        if len(fig.axes) > 1:
            ax.set_title(f"stim {idx}", fontsize=8)
        ax.set_xlabel("time (s)")
        if idx == 0:
            ax.set_ylabel("frametime interval (s)")
        ax.set_xlim(min(d) - 20, max(d) + 20)

    for ax in fig.axes:
        # after all ylims are established
        ax.set_ylim(
            bottom=max(
                0,
                expected_period - np.max(np.abs(y_deviations_from_expected_period)),
            ),
        )
        ticks_with_period = sorted(set(ax.get_yticks()) | {expected_period})
        ax.set_yticks(ticks_with_period)
        if idx == 0:
            ax.set_yticklabels([f"{_:.3f}" for _ in ticks_with_period])
    fig.set_layout_engine("tight")

    names = tuple(
        k for k, v in session.stim_frame_times.items() if not isinstance(v, Exception)
    )
    for idx, ax in enumerate(fig.axes):
        if len(names) == len(fig.axes):
            ax.set_title(names[idx].split("_")[0])
    fig.set_size_inches(12, 6)
    return fig


    
def _plot_vsyncs_and_diode_flips_at_ends_of_each_stim(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure | None:
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
    names = tuple(
        k for k, v in session.stim_frame_times.items() if not isinstance(v, Exception)
    )
    fig = session.sync_data.plot_stim_onsets()
    for idx, ax in enumerate(fig.axes):
        if len(names) == len(fig.axes):
            ax.set_title(names[idx].split("_")[0])
    fig.set_size_inches(10, 5 * len(fig.axes))
    fig.subplots_adjust(hspace=0.3)

    fig = session.sync_data.plot_stim_offsets()
    names = tuple(k for k, v in session.stim_frame_times.items() if v is not None)
    for idx, ax in enumerate(fig.axes):
        if len(names) == len(fig.axes):
            ax.set_title(names[idx].split("_")[0])
    fig.set_size_inches(10, 5 * len(fig.axes))
    fig.subplots_adjust(hspace=0.3)

    return fig


def _plot_histogram_of_frame_intervals(session) -> matplotlib.figure.Figure:
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

