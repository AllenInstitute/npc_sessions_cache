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
    timing_info = session.ephys_timing_data # skips unused probes
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
) -> matplotlib.figure.Figure:
    """
    Plot barcode intervals for sync and for each probe after sample rate
    correction
    """

    barcode_rising = session.sync_data.get_rising_edges(0, "seconds")
    barcode_falling = session.sync_data.get_falling_edges(0, "seconds")
    barcode_times, barcodes = npc_ephys.extract_barcodes_from_times(
        barcode_rising,
        barcode_falling,
        total_time_on_line=session.sync_data.total_seconds,
    )

    device_barcode_dict = {}
    for info in session.ephys_timing_data: # skips unused probes
        if "NI-DAQmx" in info.device.name or "LFP" in info.device.name:
            continue
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
        raw = ephys_barcode_times
        corrected = ephys_barcode_times * (30000 / info.sampling_rate)
        intervals = np.diff(corrected)
        max_deviation = np.max(np.abs(intervals - np.median(intervals)))

        device_barcode_dict[info.device.name] = {
            "barcode_times_raw": raw,
            "barcode_times_corrected": corrected,
            "max_deviation_from_median_interval": max_deviation,
            "max_deviation_from_30s_interval": np.max(np.abs(intervals - 30)),
        }

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
    ax[0].set_title("Sync Barcode Intervals")
    ax[1].set_title("Probe Barcode Intervals")
    ax[2].set_title("Probe Barcode Intervals Corrected")

    plt.tight_layout()
    return fig, {k:v for k,v in device_barcode_dict.items() if "barcode_times" not in k}
