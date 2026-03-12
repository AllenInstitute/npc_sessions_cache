from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING, Literal

import aind_session
import codeocean
import codeocean.computation
import matplotlib.axes
import matplotlib.colors
import matplotlib.gridspec as gs
import matplotlib.pyplot
import matplotlib.pyplot as plt
import npc_session
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import upath
from PIL import Image, ImageEnhance

if TYPE_CHECKING:
    import pynwb

import functools
import sqlite3
import tempfile

import npc_lims
import npc_sessions
import nrrd

import npc_sessions_cache.plots.plot_utils as plot_utils
import npc_sessions_cache.utils as utils

matplotlib.rcParams.update({"font.size": 8})

STRUCTURE_TREE = pd.read_csv(
    upath.UPath(
        "https://raw.githubusercontent.com/cortex-lab/allenCCF/master/structure_tree_safe_2017.csv"
    )
)
SLICE_IMAGE_OFFSET = 300
UNIT_DENSITY_OFFSET = 200
NUM_CHANNELS = 384
CCF_NUM_COLUMNS = 4  # ap, dv, ml, and region - for insertion db
RESOLUTION_UM = 25
MICRONS_PER_PIXEL = 10
BRIGHTNESS_FACTOR = 1.5
LFP_CORRELATION_NUM_SECS_TO_USE = 300
TASK_NAME_FOR_LFP_CORRELATION = "DynamicRouting1"

@numba.njit
def makePSTH_numba(
    spikes: npt.NDArray[np.floating],
    startTimes: npt.NDArray[np.floating],
    windowDur: float,
    binSize: float = 0.001,
    convolution_kernel: float = 0.05,
):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel / 2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0, windowDur + binSize, binSize)
    convkernel = np.ones(int(convolution_kernel / binSize))
    counts = np.zeros(bins.size - 1)
    for _i, start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start + windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd] - start, bins)[0]

    counts = counts / startTimes.size
    counts = np.convolve(counts, convkernel) / (binSize * convkernel.size)
    return (
        counts[convkernel.size - 1 : -convkernel.size],
        bins[: -convkernel.size - 1],
    )


def plot_unit_metrics(
    session: npc_sessions.DynamicRoutingSession,
    probe_letter: str | npc_session.ProbeRecord | None = None,
) -> tuple[plt.Figure, ...]:
    units: pd.DataFrame = session.units[:].query("default_qc")

    metrics = [
        "drift_ptp",
        "isi_violations_ratio",
        "amplitude",
        "amplitude_cutoff",
        "presence_ratio",
    ]
    probes = units["electrode_group_name"].unique()

    x_labels = {
        "presence_ratio": "fraction of session",
        "isi_violations_ratio": "violation rate",
        "drift_ptp": "microns",
        "amplitude": "uV",
        "amplitude_cutoff": "frequency",
    }
    figures = []
    for metric in metrics:
        fig, _ = plt.subplots(1, len(probes))
        probe_index = 0
        fig.suptitle(f"{metric}")
        for probe in probes:
            is_probe = (probe_letter is not None) and (
                npc_session.ProbeRecord(probe) == npc_session.ProbeRecord(probe_letter)
            )
            units_probe_metric = units[units["electrode_group_name"] == probe][metric]
            fig.axes[probe_index].hist(
                units_probe_metric,
                bins=20,
                density=True,
                color="orange" if is_probe else None,
            )
            fig.axes[probe_index].set_title(f"{probe}")
            fig.axes[probe_index].set_xlabel(x_labels[metric])
            probe_index += 1

        fig.set_size_inches([10, 6])
        plt.tight_layout()
        figures.append(fig)
    return tuple(figures)


def get_sorting_view_links(
    session: npc_sessions.DynamicRoutingSession,
    key: Literal["sorting_summary", "timeseries"],
    probe_letter: str | npc_session.ProbeRecord | None = None,
) -> tuple[str, ...]:
    vis = session.sorted_data.visualization_output_json()
    links = []
    for v in vis.values():
        if link := v.get(key):
            if probe_letter is not None:
                if npc_session.ProbeRecord(probe_letter) != npc_session.ProbeRecord(
                    link
                ):
                    continue
            links.append(link)
    return tuple(links)


def plot_sorting_view_summary_links(
    session: npc_sessions.DynamicRoutingSession,
    probe_letter: str | npc_session.ProbeRecord | None = None,
) -> tuple[str, ...]:
    return get_sorting_view_links(session, "sorting_summary", probe_letter=probe_letter)


def plot_sorting_view_timeseries_links(
    session: npc_sessions.DynamicRoutingSession,
    probe_letter: str | npc_session.ProbeRecord | None = None,
) -> tuple[str, ...]:
    return get_sorting_view_links(session, "timeseries", probe_letter=probe_letter)


def plot_all_spike_histograms(
    session: npc_sessions.DynamicRoutingSession,
    probe_letter: str | npc_session.ProbeRecord | None = None,
) -> tuple[plt.Figure, ...]:
    session.units[:].query("default_qc")
    figs: list[plt.Figure] = []
    for obj in session.all_spike_histograms.children:
        if probe_letter is not None and npc_session.ProbeRecord(
            obj.name
        ) != npc_session.ProbeRecord(probe_letter):
            continue
        fig, ax = plt.subplots()
        ax.plot(obj.timestamps, obj.data, linewidth=0.1, alpha=0.8, color="k")
        plot_utils.add_epoch_color_bars(
            ax, session.epochs[:], y=50, va="bottom", rotation=90
        )
        ax.set_title(obj.description, fontsize=8)
        fig.suptitle(session.session_id, fontsize=10)
        ax.set_xlabel(obj.timestamps_unit)
        ax.set_ylabel(obj.unit)
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.margins(0)
        ax.set_frame_on(False)
        fig.set_layout_engine("tight")
        fig.set_size_inches(5, 5)
        figs.append(fig)
    return tuple(figs)


def _plot_unit_waveform(
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile, index_or_id: int | str
) -> plt.Figure:
    """Waveform on peak channel"""
    fig = plt.figure()
    unit = (
        session.units[:].iloc[index_or_id]
        if isinstance(index_or_id, int)
        else session.units[:].query("unit_id == @index_or_id").iloc[0]
    )

    electrodes: list[int] = unit["electrodes"]
    peak_channel_idx = electrodes.index(unit["peak_electrode"])
    mean = unit["waveform_mean"][:, peak_channel_idx]
    sd = unit["waveform_sd"][:, peak_channel_idx]
    t = np.arange(mean.size) / session.units.waveform_rate * 1000  # convert to ms
    t -= max(t) / 2  # center around 0

    ax = fig.add_subplot(111)
    # ax.hlines(0, t[0], t[-1], color='grey', linestyle='--')
    m = ax.plot(t, mean, label=f"Unit {unit['unit_id']}")
    ax.fill_between(t, mean + sd, mean - sd, color=m[0].get_color(), alpha=0.25)
    ax.set_xlabel("milliseconds")
    ax.set_ylabel(session.units.waveform_unit)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(min(-100, ax.get_ylim()[0]), max(50, ax.get_ylim()[1]))
    ax.set_xmargin(0)
    if session.units.waveform_unit == "microvolts":
        ax.set_aspect(1 / 25)
    ax.grid(True)

    return fig


def _plot_unit_spatiotemporal_waveform(
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile,
    index_or_id: int | str,
    **pcolormesh_kwargs,
) -> plt.Figure:
    """Waveforms across channels around peak channel - currently no interpolation"""

    unit = (
        session.units[:].iloc[index_or_id]
        if isinstance(index_or_id, int)
        else session.units[:].query("unit_id == @index_or_id").iloc[0]
    )

    # assemble df of channels whose data we'll plot

    # electrodes with waveforms for this unit:
    electrode_group = session.electrodes[:].loc[unit["electrodes"]]

    # get largest signal from each row of electrodes on probe
    electrode_group["amplitudes"] = np.max(unit["waveform_mean"], axis=0) - np.min(
        unit["waveform_mean"], axis=0
    )

    peak_electrode = session.electrodes[:].loc[unit["peak_electrode"]]
    # ^ this is incorrect until annotations have been updated
    peak_electrode = electrode_group.sort_values(by="amplitudes").iloc[-1]

    rows = []
    for _rel_y in electrode_group.rel_y.unique():
        rows.append(
            electrode_group.query(f"rel_y == {_rel_y}")
            .sort_values(by="amplitudes")
            .iloc[-1]
        )
    selected_electrodes = pd.DataFrame(rows)
    assert len(selected_electrodes) == len(electrode_group.rel_y.unique())

    electrode_indices: list[int] = unit["electrodes"]
    waveforms = unit["waveform_mean"][
        :, np.searchsorted(electrode_indices, selected_electrodes.index)
    ]

    t = (
        np.arange(waveforms.shape[0]) / session.units.waveform_rate * 1000
    )  # convert to ms
    t -= max(t) / 2  # center around 0
    absolute_y = sorted(selected_electrodes.rel_y)
    relative_y = absolute_y - peak_electrode.rel_y  # center around peak electrode

    fig = plt.figure()
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=-150,
        vcenter=0,
        vmax=150,
    )  # otherwise, if all waveforms are zeros the vmin/vmax args become invalid

    pcolormesh_kwargs.setdefault("cmap", "bwr")
    _ = plt.pcolormesh(t, relative_y, waveforms.T, norm=norm, **pcolormesh_kwargs)
    ax = fig.gca()
    ax.set_xmargin(0)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("milliseconds")
    ax.set_ylabel("microns from peak channel")
    ax.set_yticks(relative_y)
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda y: y + peak_electrode.rel_y,
            lambda y: y - peak_electrode.rel_y,
        ),
    )
    secax.set_ylabel("microns from tip")
    secax.set_yticks(absolute_y)
    ax.set_aspect(1 / 50)
    ax.grid(True, axis="x", lw=0.5, color="grey", alpha=0.5)
    plt.colorbar(
        ax=ax,
        fraction=0.01,
        pad=0.2,
        label=session.units.waveform_unit,
        ticks=[norm.vmin, norm.vcenter, norm.vmax],
    )
    fig.suptitle(
        f"{unit['unit_id']}\n{unit.peak_channel=}\nunit.amplitude={electrode_group['amplitudes'].max():.0f} {session.units.waveform_unit}",
        fontsize=8,
    )
    return fig


def _plot_ephys_noise(
    timeseries: pynwb.TimeSeries,
    interval: utils.Interval | None = None,
    median_subtraction: bool = True,
    y_range: npt.NDArray | None = None,
    ax: matplotlib.axes.Axes | None = None,
    **plot_kwargs,
) -> plt.Figure:
    timestamps = timeseries.get_timestamps()
    if interval is None:
        interval = ((t := np.ceil(timestamps)[0]), t + 1)
    t0, t1 = npc_sessions.parse_intervals(interval)[0]
    s0, s1 = np.searchsorted(timestamps, (t0, t1))
    if s0 == s1:
        raise ValueError(
            f"{interval=} is out of bounds ({timestamps[0]=}, {timestamps[-1]=})"
        )
    samples = np.arange(s0, s1)
    data = timeseries.data[samples, :] * timeseries.conversion * 1000  # microvolts

    def std(data):
        std = np.nanstd(data, axis=0)
        return std

    if ax is None:
        ax = plt.subplot()

    plot_kwargs.setdefault("lw", 0.5)
    plot_kwargs.setdefault("color", "k")
    if y_range is None:
        ax.plot(std(data), np.arange(data.shape[1]), **plot_kwargs, alpha=0.3)
    else:
        ax.plot(std(data), y_range, **plot_kwargs, alpha=0.3)
    if median_subtraction:
        offset_corrected_data = data - np.nanmedian(data, axis=0)
        median_subtracted_data = (
            offset_corrected_data.T - np.nanmedian(offset_corrected_data, axis=1)
        ).T
        if y_range is None:
            ax.plot(
                std(median_subtracted_data),
                np.arange(median_subtracted_data.shape[1]),
                **plot_kwargs | {"color": "r", "alpha": 0.3},
            )
        else:
            ax.plot(
                std(median_subtracted_data),
                y_range,
                **plot_kwargs | {"color": "r", "alpha": 0.3},
            )
    ax.set_ymargin(0)
    ax.set_xlabel("SD (microvolts)")
    ax.set_ylabel("channel number")
    ax.set_title(f"noise on {timeseries.electrodes.description}")
    fig = ax.get_figure()
    assert fig is not None
    return fig


def _plot_ephys_image(
    timeseries: pynwb.TimeSeries,
    interval: utils.Interval | None = None,
    median_subtraction: bool = True,
    offset_correction: bool = True,
    ax: matplotlib.axes.Axes | None = None,
    **imshow_kwargs,
) -> plt.Figure:
    timestamps = timeseries.get_timestamps()
    if interval is None:
        interval = ((t := np.ceil(timestamps)[0]), t + 1)
    t0, t1 = npc_sessions.parse_intervals(interval)[0]
    s0, s1 = np.searchsorted(timestamps, (t0, t1))
    if s0 == s1:
        raise ValueError(
            f"{interval=} is out of bounds ({timestamps[0]=}, {timestamps[-1]=})"
        )
    samples = np.arange(s0, s1)
    data = timeseries.data[samples, :] * timeseries.conversion
    if offset_correction:
        data = data - np.nanmedian(data, axis=0)
    if median_subtraction:
        data = (data.T - np.nanmedian(data, axis=1)).T

    if ax is None:
        ax = plt.subplot()

    imshow_kwargs.setdefault("vmin", -(vrange := np.nanstd(data) * 3))
    imshow_kwargs.setdefault("vmax", vrange)
    imshow_kwargs.setdefault("cmap", "bwr")
    imshow_kwargs.setdefault("interpolation", "none")

    ax.imshow(
        data.T,
        aspect=1 / data.shape[1],  # assumes`extent` provided with seconds
        extent=(t0, t1, data.shape[1], 0),
        **imshow_kwargs,
    )
    ax.invert_yaxis()
    ax.set_ylabel("channel number")
    ax.set_xlabel("seconds")
    fig = ax.get_figure()
    assert fig is not None
    return fig


def _plot_session_ephys_noise(
    session: npc_sessions.DynamicRoutingSession,
    lfp: bool = False,
    interval: utils.Interval = None,
    median_subtraction: bool = True,
    offset_correction: bool = True,
    **plot_kwargs,
) -> plt.Figure:
    if lfp:
        container = session._raw_lfp
    else:
        container = session._raw_ap
    fig, _ = plt.subplots(1, len(container.electrical_series), sharex=True, sharey=True)
    for idx, (label, timeseries) in enumerate(container.electrical_series.items()):
        ax = fig.axes[idx]
        _plot_ephys_noise(
            timeseries,
            ax=ax,
            interval=interval,
            median_subtraction=median_subtraction,
            offset_correction=offset_correction,
            **plot_kwargs,
        )
        ax.set_title(label, fontsize=8)
        if idx > 0:
            ax.yaxis.set_visible(False)

        if idx != round(len(fig.axes) / 2):
            ax.xaxis.set_visible(False)

    fig.suptitle(
        f'noise on channels with {"LFP" if lfp else "AP"} data (red: med subtracted)\n{session.session_id}',
        fontsize=10,
    )
    return fig


def plot_raw_ephys_segments(
    session: npc_sessions.DynamicRoutingSession,
    lfp: bool = False,
    interval: utils.Interval = None,
    median_subtraction: bool = True,
    probe_letter: str | npc_session.ProbeRecord | None = None,
    **imshow_kwargs,
) -> tuple[plt.Figure, ...]:
    if lfp:
        container = session._raw_lfp
    else:
        container = session._raw_ap
    start_times = (1, 100, -10)
    figures = []
    for device, timeseries in container.electrical_series.items():
        if probe_letter is not None:
            if npc_session.ProbeRecord(probe_letter) != npc_session.ProbeRecord(device):
                continue
        fig, _ = plt.subplots(1, len(start_times), sharey=True)
        for idx, start_time in enumerate(start_times):
            ax = fig.axes[idx]
            duration = 0.3
            if interval is None:
                if timeseries.timestamps is not None:
                    if start_time > 0:
                        t0 = timeseries.timestamps[0] + start_time
                    else:
                        t0 = timeseries.timestamps[-1] + start_time
                else:
                    if start_time > 0:
                        t0 = timeseries.starting_time + start_time
                    else:
                        t0 = (
                            timeseries.starting_time
                            + (timeseries.data.shape[0] / timeseries.rate)
                            + start_time
                        )
                temp_interval = (t0, t0 + duration)
            print(temp_interval)
            _plot_ephys_image(
                timeseries,
                ax=ax,
                interval=temp_interval,
                median_subtraction=median_subtraction,
                **imshow_kwargs,
            )
            if idx > 0:
                ax.yaxis.set_visible(False)
        fig.suptitle(
            f'raw {"LFP" if lfp else "AP"} data {device}\n{median_subtraction=}\n{session.session_id}',
            fontsize=10,
        )
        figures.append(fig)
    return tuple(figures)


def plot_raw_ap_vs_surface(
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile,
    probe_letter: str | npc_session.ProbeRecord | None = None,
) -> tuple[plt.Figure, ...] | None:
    if not session.is_surface_channels:
        return None
    time_window = 0.5

    figs = []
    for probe in session._raw_ap.electrical_series.keys():
        if probe_letter is not None and npc_session.ProbeRecord(
            probe
        ) != npc_session.ProbeRecord(probe_letter):
            continue
        n_samples = int(time_window * session._raw_ap[probe].rate)
        offset_corrected = session._raw_ap[probe].data[-n_samples:, :] - np.median(
            session._raw_ap[probe].data[-n_samples:, :], axis=0
        )
        car = (offset_corrected.T - np.median(offset_corrected, axis=1)).T

        if (
            session.is_surface_channels
            and probe in session.surface_recording._raw_ap.fields["electrical_series"]
        ):
            n_samples = int(time_window * session.surface_recording._raw_ap[probe].rate)
            offset_corrected_surface = session.surface_recording._raw_ap[probe].data[
                -n_samples:, :
            ] - np.median(
                session.surface_recording._raw_ap[probe].data[-n_samples:, :], axis=0
            )
            car_surface = (
                offset_corrected_surface.T - np.median(offset_corrected_surface, axis=1)
            ).T
            surface_channel_recording = True
        else:
            surface_channel_recording = False

        range = np.nanstd(car.flatten()) * 3

        fig, ax = plt.subplots(1, 2, figsize=(15, 8))

        ax[0].imshow(
            car.T,
            aspect="auto",
            interpolation="none",
            cmap="bwr",
            vmin=-range,
            vmax=range,
        )
        ax[0].invert_yaxis()
        ax[0].set_title(
            "deep channels (last " + str(time_window) + " sec of recording)"
        )
        ax[0].set_ylabel("channel number")
        ax[0].set_xlabel("samples")

        if surface_channel_recording:
            ax[1].imshow(
                car_surface.T,
                aspect="auto",
                interpolation="none",
                cmap="bwr",
                vmin=-range,
                vmax=range,
            )

        ax[1].invert_yaxis()
        ax[1].set_title(
            "surface channels (first " + str(time_window) + " sec of recording)"
        )
        ax[1].set_xlabel("samples")

        fig.suptitle(session.session_id + " " + probe)

        figs.append(fig)

    return tuple(figs)


def get_optotagging_params(optotagging_trials: pd.DataFrame) -> dict[str, list]:
    optotagging_params = {
        c: sorted(set(optotagging_trials[c]))
        for c in optotagging_trials.columns
        if not any(c.endswith(n) for n in ("_time", "_index"))
    }
    if any(v for v in optotagging_params.get("location", [])):
        del optotagging_params["bregma_x"]
        del optotagging_params["bregma_y"]
    return optotagging_params


# adapted from nwb_validation_optotagging.py
# https://github.com/AllenInstitute/np_pipeline_qc/blob/main/src/np_pipeline_qc/legacy/nwb_validation_optotagging.py


def plot_optotagging(
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile,
    combine_locations: bool = True,
    combine_probes: bool = False,
) -> tuple[plt.Figure, ...] | None:
    try:
        opto_trials = session.intervals["optotagging_trials"][:]
    except KeyError:
        return None
    electrodes = session.electrodes[:]
    units = session.units[:]
    good_unit_filter = (
        (units["snr"] > 1)
        & (units["isi_violations_ratio"] < 1)
        & (units["firing_rate"] > 0.1)
    )
    units = units.loc[good_unit_filter]
    units.drop(columns="group_name", errors="ignore", inplace=True)
    units_electrodes = units.merge(
        electrodes[["rel_x", "rel_y", "channel", "group_name"]],
        left_on=["electrode_group_name", "peak_channel"],
        right_on=["group_name", "channel"],
    ).drop(columns=["channel", "group_name"])

    durations = sorted(opto_trials.duration.unique())
    powers = sorted(opto_trials.power.unique())
    probes = sorted(units.electrode_group_name.unique())
    locations = sorted(opto_trials.location.unique())

    locations_are_probes = all(loc in probes for loc in locations)

    figs = []
    for location in (
        (None,) if (combine_locations or locations_are_probes) else locations
    ):
        for probe in probes:

            if not combine_probes:
                filtered_units = units_electrodes.query(
                    f"electrode_group_name == {probe!r}"
                )
            else:
                filtered_units = units_electrodes

            fig, axes = plt.subplots(len(powers), len(durations))
            fig.set_size_inches([1 + 6 * len(durations), 1 + 2 * len(powers)])
            title_text = f"{session.session_id} | {session.subject.genotype}"
            save_suffix = f"{session.session_id}"
            if location and not combine_locations:
                title_text = f"{title_text}\n{location}"
                save_suffix = f"{location}_{save_suffix}"
            else:
                title_text = f"{title_text}\npooled: {locations!r}"
            if not combine_probes:
                title_text = f"{title_text}\n{probe}"
                save_suffix = f"{probe}_{save_suffix}"
            else:
                title_text = f"{title_text}\npooled: {probes!r}"
            fig.suptitle(title_text)

            for idur, duration in enumerate(durations):
                for il, power in enumerate(powers):
                    filtered_trials = opto_trials.query(
                        f"duration == {duration!r} & power == {power!r}"
                    )
                    if not combine_locations:
                        filtered_trials = filtered_trials.query(
                            f"location == {location if location else probe!r}"
                        )
                    start_times = filtered_trials["start_time"].values

                    bin_size = 0.001
                    window_dur = 5 * duration * round(np.log10(1 / duration))
                    baseline_dur = (window_dur - duration) / 2
                    convolution_kernel = max(duration / 10, 2 * bin_size)
                    all_resp = []
                    for _iu, unit in filtered_units.sort_values("rel_y").iterrows():
                        sts = np.array(unit["spike_times"])
                        resp = makePSTH_numba(
                            sts,
                            start_times - baseline_dur,
                            window_dur,
                            binSize=bin_size,
                            convolution_kernel=convolution_kernel,
                        )[0]
                        resp = resp - np.mean(resp[: int(baseline_dur / bin_size) - 1])
                        all_resp.append(resp)

                    t = (np.arange(0, window_dur, bin_size) - baseline_dur) / bin_size
                    all_resp = np.array(all_resp)
                    min_clim_val = -5
                    max_clim_val = 50
                    norm = matplotlib.colors.TwoSlopeNorm(
                        vmin=min_clim_val,
                        vcenter=(min_clim_val + max_clim_val) / 2,
                        vmax=max_clim_val,
                    )
                    if len(powers) == 1 and len(durations) == 1:
                        ax = axes
                    elif len(powers) == 1:
                        ax = axes[idur]
                    elif len(durations) == 1:
                        ax = axes[il]
                    else:
                        ax = axes[il][idur]
                    fig.sca(ax)
                    _ = plt.pcolormesh(
                        t,
                        np.arange(all_resp.shape[0]),
                        all_resp,
                        cmap="viridis",
                        norm=norm,
                    )
                    ax.set_xmargin(0)

                    ax.set_aspect(
                        0.25 * window_dur * 1000 / 300
                    )  # 300 units in Y == 1/3 time in X (remember X is in milliseconds)
                    ax.set_ylabel(
                        f"units [ch{filtered_units.peak_channel.min()}-{filtered_units.peak_channel.max()}]"
                    )
                    if il != len(powers) - 1:
                        ax.set_xticklabels([])
                    else:
                        ax.set_xlabel("milliseconds")
                    for marker_position in (0, duration / bin_size):
                        ax.annotate(
                            "",
                            xy=(marker_position, all_resp.shape[0]),
                            xycoords="data",
                            xytext=(marker_position, all_resp.shape[0] + 0.5),
                            textcoords="data",
                            arrowprops={
                                "arrowstyle": "simple",
                                "color": "black",
                                "lw": 0,
                            },
                        )
                    ax.set_title(f"{power = :.1f}", y=1.05)
            figs.append(fig)
            if combine_probes:
                break
        if combine_locations:
            break
    return tuple(figs)


def plot_probe_yield(
    session: npc_sessions.DynamicRoutingSession,
    probe_letter: str | npc_session.ProbeRecord | None = None,
) -> plt.Figure:
    del probe_letter  # unused  - just allows functools.partial application
    units = session.units[:]
    good_unit_filter = (
        (units["amplitude_cutoff"] < 0.1)
        & (units["isi_violations_ratio"] < 0.5)
        & (units["presence_ratio"] > 0.95)
    )
    counts = []
    for probe in sorted(units["electrode_group_name"].unique()):
        probe_filter = units["electrode_group_name"] == probe
        probe_units = units.loc[probe_filter]
        good_units = units.loc[good_unit_filter & probe_filter]
        counts.append(
            {
                "probe": probe.removeprefix("probe"),
                "good": len(good_units),
                "bad": len(probe_units) - len(good_units),
            }
        )

    ax = pd.DataFrame.from_records(counts).plot.bar(
        x="probe",
        ylabel="units",
        stacked=True,
        color={"bad": "grey", "good": "green"},
        width=0.5,
    )
    ax.set_title(
        f"unit yield (total={len(units)})\namplitude_cutoff < 0.1 | isi_violations_ratio < 0.5 | presence_ratio > 0.95\n{session.id}",
        fontsize=8,
    )
    ax.set_aspect(1 / 100)
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    return fig


def _get_unit_denisty_per_channel(
    unit_channel_counts: npt.NDArray[np.int64], num_channels: int = 384
) -> npt.NDArray[np.int64]:
    unit_channels = unit_channel_counts[:, 0].tolist()
    unit_denisty_values = []

    for i in range(num_channels):
        if i in unit_channels:
            index = unit_channels.index(i)
            unit_denisty_values.append([i, unit_channel_counts[index, 1]])
        else:
            unit_denisty_values.append([i, 0])

    unit_denisty_values_array = np.array(unit_denisty_values)
    return unit_denisty_values_array


def _plot_structure_areas(
    electrodes_probe: pd.DataFrame,
    y_positions: list,
    ax: matplotlib.axes.Axes,
    num_channels: int = 384,
    unit_density_offset: int = UNIT_DENSITY_OFFSET,
    use_median_for_text: bool = True,
) -> tuple[str, ...]:
    color = "000000"
    structures_seen = {}
    structure_positions = {}

    for i in range(num_channels - 1, -1, -1):
        structure = electrodes_probe[electrodes_probe["channel"] == i][
            "structure"
        ].values[0]
        if structure not in structures_seen:
            if structure != "out of brain":
                if structure == "undefined":
                    color = "000000"
                else:
                    color = STRUCTURE_TREE[STRUCTURE_TREE["acronym"] == structure][
                        "color_hex_triplet"
                    ].values[0]

            matplotlib.patches.Patch(color=f"#{color}", label=structure)
            # legend.append(patch)
            structures_seen[structure] = color
            structure_positions[structure] = [
                (
                    ax.get_xlim()[1] - 0.0007,
                    (y_positions[i] - unit_density_offset) * MICRONS_PER_PIXEL,
                )
            ]
        else:
            color = structures_seen[structure]
            structure_positions[structure].append(
                (
                    ax.get_xlim()[1] - 0.0007,
                    (y_positions[i] - unit_density_offset) * MICRONS_PER_PIXEL,
                )
            )

        rect = matplotlib.patches.Rectangle(
            (
                ax.get_xlim()[1] - 0.0007,
                (y_positions[i] - unit_density_offset) * MICRONS_PER_PIXEL,
            ),
            width=0.0005,
            height=0.0005,
            color=f"#{color}",
        )
        ax.add_patch(rect)

    # ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1, 0.5))
    for structure in structure_positions:
        y_structure_position = np.array(
            [pos[1] for pos in structure_positions[structure]]
        )
        if use_median_for_text:
            text_position = np.median(y_structure_position)
        else:
            text_position = y_structure_position[0]

        ax.text(
            ax.get_xlim()[1],
            text_position,
            s=structure,
            color=f"#{structures_seen[structure]}",
            fontweight="bold",
        )

    return tuple(structure_positions.keys())


def _plot_ephys_noise_with_unit_density_areas(
    session: npc_sessions.DynamicRoutingSession,
    probe: str,
    num_channels: int = 384,
    lfp_correlation: npt.NDArray | None = None
) -> plt.Figure:

    # Here, we are recreating the unit denisty directly from the sorted data
    electrodes = session.electrodes[:]
    units = session.units[:]
    units_probe = units[units["electrode_group_name"] == probe]
    electrodes_probe = electrodes[electrodes["group_name"] == probe]
    peak_channel = units_probe["peak_channel"]

    unit_channel_counts = (
        peak_channel.value_counts().sort_index().reset_index().to_numpy()
    )
    unit_denisty_values_from_spike_sorting = _get_unit_denisty_per_channel(unit_channel_counts)

    timeseries_probe = session._raw_ap.electrical_series[probe]
    kernel_size = 10
    conv = np.ones(kernel_size) / kernel_size

    smoothed = np.convolve(unit_denisty_values_from_spike_sorting[:, 1], conv, mode="same") / 1000

    # pull relevant paths that have been uploaded from the gui
    image_path = (
        upath.UPath(
            "s3://aind-scratch-data/arjun.sridhar/tissuecyte_cloud_processed/slice_images"
        )
        / f"{session.info.subject}"
        / f"Probe_{probe[-1]}{session.info.experiment_day}_slice.png"
    )
    anchors_path = (
        upath.UPath(
            "s3://aind-scratch-data/arjun.sridhar/tissuecyte_cloud_processed/alignment_anchors"
        )
        / f"{session.info.subject}"
        / f"Probe_{probe[-1]}{session.info.experiment_day}_anchors.pickle"
    )

    correlation_plot_path = (
        upath.UPath(
            "s3://aind-scratch-data/arjun.sridhar/tissuecyte_cloud_processed/correlation_plots"
        )
        / f"{session.info.subject}"
        / f"{probe[-1]}{session.info.experiment_day}_corr_full.pickle"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"No slice images for session {session.id} and probe {probe}"
        )

    if not anchors_path.exists():
        raise FileNotFoundError(
            f"No alignments for session {session.id} and probe {probe}"
        )

    if not correlation_plot_path.exists():
        raise FileNotFoundError(
            f"No correlation plot for session {session.id} and probe {probe}"
        )

    with io.BytesIO(image_path.read_bytes()) as f:
        image = Image.open(f)
        enhancer = ImageEnhance.Brightness(image)
        brightened_image = enhancer.enhance(BRIGHTNESS_FACTOR)
        slice_image = np.array(brightened_image)

    # read data - pickle file is poor but such is life
    correlation_plot_data = pd.read_pickle(correlation_plot_path)["img"]
    anchors = pd.read_pickle(anchors_path)
    unit_density_points_gui = np.array(anchors[0])
    y_positions_gui = [point[1] for point in unit_density_points_gui]

    grid_spec = gs.GridSpec(1, 4, width_ratios=[2, 0.5, 1, 1])
    fig = plt.figure(figsize=(12, 6))

    # Create subplots using the grid
    ax1 = fig.add_subplot(grid_spec[0])  # First plot (larger)
    ax2 = fig.add_subplot(grid_spec[1])  # Second plot
    ax3 = fig.add_subplot(grid_spec[2])  # Third plot
    ax4 = fig.add_subplot(grid_spec[3])  # Fourth plot

    anchor_positions = anchors[3]
    probe_channel_space = [] # get channel index in probe space

    for anchor in anchors[3]:
        if anchor in y_positions_gui:
            probe_channel_space.append(y_positions_gui.index(anchor))

    ax4.imshow(slice_image[SLICE_IMAGE_OFFSET:, :])
  
    _plot_ephys_noise(
        timeseries_probe,
        ax=ax3,
        y_range=(unit_density_points_gui[:, 1][:num_channels] - UNIT_DENSITY_OFFSET)
        * MICRONS_PER_PIXEL,
    )
    ax3.plot(
        smoothed,
        (unit_density_points_gui[:, 1][:num_channels] - UNIT_DENSITY_OFFSET)
        * MICRONS_PER_PIXEL,
    )

    for position in anchor_positions:
        ax3.axhline(y=(position - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL, c="r")
        ax4.axhline(y=position - UNIT_DENSITY_OFFSET, c="r")

    scales_for_plotting = []
    probe_channel_space = sorted(probe_channel_space)[::-1]
    anchor_positions = sorted(anchor_positions)
    for anchor_index, position in enumerate(anchor_positions):
        if anchor_index - 1 >= 0:
            # since we assume 10 micron spacing on the probe, and the image space is 10 microns
            # this can be computed directly, where a pixel is 10 microns. The probe channel space is on the denominator here
            # current formula is scale = (hist_position - hist_position(i - 1)) / (channel - channel(i - 1))
            # so scale is taking number of channels between 2 anchors that have either been stretched or compressed 
            # Thus, I think a value > 1 means the channels have been stretched between the 2 anchors and opposite for < 1
            # The above interpreation could be completely wrong so take it with a grain of salt
            scale = np.abs((position - anchor_positions[anchor_index - 1]) \
            / (probe_channel_space[anchor_index] - probe_channel_space[anchor_index - 1]))
            # x pos and y_mid are for displaying
            x_pos = ax3.get_xlim()[0] - 0.08 * (ax3.get_xlim()[1] - ax3.get_xlim()[0])  
            y_mid = ((position + anchor_positions[anchor_index - 1]) / 2 - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL

            scales_for_plotting.append((scale, x_pos, y_mid))

        ax3.axhline(y=(position - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL, c="r")
        ax4.axhline(y=position - UNIT_DENSITY_OFFSET, c="r")

    # get final scale factor and plotting stuff
    # This should be around the same for all of them
    last_scale_factor = np.abs((max(y_positions_gui) - anchor_positions[-1]) \
            / (0 - probe_channel_space[-1]))
    x_pos_last = ax3.get_xlim()[0] - 0.08 * (ax3.get_xlim()[1] - ax3.get_xlim()[0]) 
    y_mid_last = ((max(unit_density_points_gui[:, 1][:num_channels]) + anchor_positions[-1]) / 2 - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL
    scales_for_plotting.append((last_scale_factor, x_pos_last, y_mid_last))

    for scale_to_plot in scales_for_plotting:
        scale, x_pos, y_mid = scale_to_plot
        # Place the text
        ax3.text(
            x_pos,
            y_mid,
            f"{scale:.2f}",
            color="orange",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )
    ax3.set_ylim(max(y_positions_gui) * MICRONS_PER_PIXEL, 0)
    ax2.plot(
        unit_density_points_gui[:, 0], (unit_density_points_gui[:, 1] - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL
    )

    if lfp_correlation is None:
        ax1.imshow(
            np.flipud(correlation_plot_data),
            extent=[
                0,
                num_channels * MICRONS_PER_PIXEL,
                (min(unit_density_points_gui[:, 1] - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL),
                (max(unit_density_points_gui[:, 1] - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL),
            ],
            cmap="viridis",
        )
    else:
        order = np.arange(lfp_correlation.shape[0])[::-1]
        lfp_corr_sorted = np.fliplr(lfp_correlation[np.ix_(order, order)])
        ax1.imshow(
            np.flipud(lfp_corr_sorted),
            extent=[
                0,
                num_channels * MICRONS_PER_PIXEL,
                (unit_density_points_gui[:, 1][0] - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL,
                (unit_density_points_gui[:, 1][-1] - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL,
            ],
            cmap="viridis",
        )

    # plot anchors on correlation image
    for position in anchor_positions:
        ax1.axhline(y=(position - UNIT_DENSITY_OFFSET) * MICRONS_PER_PIXEL, c="r")

    ax1.set_axis_off()
    ax1.set_aspect("auto")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.sharey(ax3)

    ax2.set_xticks([])
    ax2.set_ylim(max(y_positions_gui) * MICRONS_PER_PIXEL, 0)
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax4.set_ylim(max(y_positions_gui), 0)
    ax4.yaxis.tick_right()
    _plot_structure_areas(electrodes_probe, y_positions_gui, ax3)

    ax3.set_ylim(max(y_positions_gui) * MICRONS_PER_PIXEL, 0)
    ax3.set_yticks([0, 500, 5000, 6000])
    ax3.set_title("")
    ax3.set_ylabel("Microns")
    ax3.set_xlabel("")

    is_deep_insertion = "deep_insertions" in session.keywords
    if is_deep_insertion:
        deep_probes = next(
            kw for kw in session.keywords if "deep_insertion_probe_letters" in kw
        ).split("=")[-1]
        is_deep_probe = npc_session.ProbeRecord(probe) in deep_probes
    else:
        is_deep_probe = False
    ax3.set_title(
        f"CCF aligned with unit density (blue) and raw ephys noise (black/red)\n{probe} | {'deep' if is_deep_insertion and is_deep_probe else 'regular'} insertion"
    )
    plt.tight_layout()

    return fig

def extract_lfp_epoch_segment(
    lfp: np.ndarray,
    fs: float,
    lfp_start_time: float,
    epochs_df: pd.DataFrame,
    task_epoch_name: str,
    duration_s: Optional[float] = None,
    channels_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a continuous LFP segment from a named epoch.

    Parameters
    ----------
    lfp : np.ndarray
        LFP array, shape (n_samples, n_channels) or (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    lfp_start_time : float
        Absolute start time of the LFP recording (seconds)
    epochs_df : pd.DataFrame
        Epoch table with columns ['start_time', 'stop_time', 'script_name']
    task_epoch_name : str
        Epoch name to extract (e.g. 'DynamicRouting1')
    duration_s : float, optional
        Duration to extract from epoch start (seconds).
        If None, extract full epoch.
    channels_first : bool
        Set True if lfp is shaped (n_channels, n_samples)

    Returns
    -------
    lfp_segment : np.ndarray
        Extracted LFP segment (same channel/sample ordering as input)
    t : np.ndarray
        Time vector in seconds (absolute)
    """

    # --- get epoch ---
    epoch = epochs_df.loc[epochs_df["script_name"] == task_epoch_name].iloc[0]
    epoch_start = epoch["start_time"]
    epoch_stop = epoch["stop_time"]

    # --- define window ---
    if duration_s is None:
        window_stop = epoch_stop
    else:
        window_stop = min(epoch_start + duration_s, epoch_stop)

    window_start = epoch_start

    # --- time → sample ---
    start_idx = int(np.round((window_start - lfp_start_time) * fs))
    stop_idx  = int(np.round((window_stop  - lfp_start_time) * fs))

    if start_idx < 0 or stop_idx > lfp.shape[-1 if channels_first else 0]:
        raise ValueError("Requested window is outside LFP data bounds")

    # --- slice ---
    if channels_first:
        lfp_segment = lfp[:, start_idx:stop_idx]
    else:
        lfp_segment = lfp[start_idx:stop_idx, :]

    # --- time vector ---
    n_samples = stop_idx - start_idx
    t = np.arange(n_samples) / fs + window_start

    return lfp_segment, t

def get_lfp_cmr_corr(
    lfp: npt.NDArray,
    axis_time: int = 0,
) -> npt.NDArray:
    """
    Compute channel-channel correlation on single-shank LFP
    using common median reference

    Parameters
    ----------
    lfp : npt.NDArray
        LFP data, shape (time, channels) by default
    axis_time : int
        Axis corresponding to time (0 or 1)

    Returns
    -------
    corr : npt.NDArray
        Correlation matrix (channels x channels)
    """

    # normalize to (time, channels)
    if axis_time != 0:
        lfp = np.swapaxes(lfp, axis_time, 0)

    # common median reference
    cmr = np.nanmedian(lfp, axis=1, keepdims=True)
    lfp_cmr = lfp - cmr

    # correlation
    corr = np.corrcoef(lfp_cmr.T)

    return corr

def get_lfp_correlation(
    raw_lfp_probe: npt.NDArray,
    epochs: pd.DataFrame,
    task_name: str,
) -> npt.NDArray:
    """
    Extract LFP segment for a probe and compute the reordered CMR correlation matrix.

    Parameters
    ----------
    raw_lfp_probe : object
        Raw LFP object for a specific probe, with attributes `data`, `rate`, and `starting_time`.
    epochs : pd.DataFrame
        DataFrame of epochs (pre-fetched from session).
    task_epoch_name : str
        Task name used to select the epoch.

    Returns
    -------
    npt.NDArray
        Reordered (flipped) channel x channel correlation matrix.
    """
    lfp_segment, _ = extract_lfp_epoch_segment(
        raw_lfp_probe.data,
        fs=raw_lfp_probe.rate,
        lfp_start_time=raw_lfp_probe.starting_time,
        epochs_df=epochs,
        task_epoch_name=task_name,
        duration_s=LFP_CORRELATION_NUM_SECS_TO_USE,
    )

    corr = get_lfp_cmr_corr(lfp_segment)
    order = np.arange(corr.shape[0])[::-1]
    return np.fliplr(corr[np.ix_(order, order)])

def plot_ccf_aligned_ephys(
    session: npc_sessions.DynamicRoutingSession,  probe: str | None = None, use_lfp_correlation: bool = True
) -> tuple[plt.Figure, ...] | None:
    """
    Plots the raw ephys noise with the unit density from sorting, along with the channel alignments and slice the probe went through

    Uses either spike correlation (from gui), or computes lfp correlation and uses that. Default is to use lfp correlation
    """
    if not session.is_annotated:
        return None

    figures = []
    # Fetch once, reuse across probes
    epochs = session.epochs[:] if use_lfp_correlation else None
    raw_lfp = session._raw_lfp if use_lfp_correlation else None
    lfp_correlation = None

    if probe is not None:
        lfp_correlation = (
            get_lfp_correlation(raw_lfp_probe=raw_lfp[probe], epochs=epochs, task_name=TASK_NAME_FOR_LFP_CORRELATION)
            if use_lfp_correlation
            else None
        )
        figures.append(_plot_ephys_noise_with_unit_density_areas(session, probe, lfp_correlation=lfp_correlation))
    else:
        probes = sorted(session.electrodes[:]["group_name"].unique())
        for probe in probes:
            lfp_correlation = (
                get_lfp_correlation(raw_lfp_probe=raw_lfp[probe], epochs=epochs, task_name=TASK_NAME_FOR_LFP_CORRELATION)
                if use_lfp_correlation
                else None
            )
            figures.append(_plot_ephys_noise_with_unit_density_areas(session, probe, lfp_correlation=lfp_correlation))

    return tuple(figures)


def _plot_electrodes_implant_hole(
    session: npc_sessions.DynamicRoutingSession,
    probe: str,
    electrodes: pd.DataFrame,
    ccf_volume: npt.NDArray,
    probe_insertion_db_connection: sqlite3.Connection,
) -> plt.Figure:
    electrodes_probe = electrodes[electrodes["group_name"] == probe]
    electrode_groups = session.electrode_groups

    implant = electrode_groups[probe].location.split(" ")[0]
    hole = electrode_groups[probe].location.split(" ")[1]
    cursor = probe_insertion_db_connection.execute(
        f"SELECT * FROM channel_ccf_coords cf WHERE (cf.Probe = '{probe[-1]}') AND (cf.Implant = '{implant}') AND (cf.Hole = '{hole}')"
    )

    sessions_probe_hole_implant = pd.DataFrame(cursor.fetchall()).to_numpy()
    electrode_session_coordinates = (
        electrodes_probe[["x", "y", "z"]].to_numpy() / RESOLUTION_UM
    )
    electrodes_coordinates = None

    for row in sessions_probe_hole_implant:
        if electrodes_coordinates is None:
            electrodes_coordinates = row[9:].reshape((NUM_CHANNELS, CCF_NUM_COLUMNS))
        else:
            electrodes_coordinates = np.concatenate(
                (
                    electrodes_coordinates,
                    row[9:].reshape((NUM_CHANNELS, CCF_NUM_COLUMNS)),
                )
            )

    if electrodes_coordinates is not None:
        electrodes_coordinates = electrodes_coordinates[
            electrodes_coordinates[:, 3] != "out of brain"
        ]

    fig, ax = plt.subplots()
    ax.imshow(ccf_volume.sum(axis=1))
    if (
        electrodes_coordinates is not None
    ):  # no other insertions for probe implant hole configuration
        ax.scatter(
            electrodes_coordinates[:, 2], electrodes_coordinates[:, 0], c="r", s=2
        )

    ax.scatter(
        electrode_session_coordinates[:, 2],
        electrode_session_coordinates[:, 0],
        c="y",
        s=2,
    )
    ax.set_title(
        f"Session {session.id} electrodes for {probe} with implant hole {electrode_groups[probe].location} spread"
    )

    return fig


@functools.cache
def _get_ccf_volume(ccf_template_path: upath.UPath) -> npt.NDArray:
    tempdir = tempfile.mkdtemp()
    temp_path = upath.UPath(tempdir) / ccf_template_path.name
    temp_path.write_bytes(ccf_template_path.read_bytes())
    path = temp_path

    return nrrd.read(path)[0]


def plot_insertion_history(
    session: npc_sessions.DynamicRoutingSession, probe: str | None = None
) -> tuple[plt.Figure, ...] | None:
    """
    Plots horizontal view of ccf volume with probe for session in yellow, and all other probes that went through same insertion configuration (same probe, hole, and implant) in red
    """
    if not session.is_annotated:
        return None
    figures = []

    ccf_template_path = upath.UPath(
        "s3://aind-scratch-data/arjun.sridhar/average_template_25.nrrd"
    )  # different volume than ccf.py
    ccf_volume = _get_ccf_volume(ccf_template_path)
    electrodes = session.electrodes[:]
    probe_insertion_db_connection = npc_lims.get_probe_target_db()

    if probe is not None:
        figures.append(
            _plot_electrodes_implant_hole(
                session, probe, electrodes, ccf_volume, probe_insertion_db_connection
            )
        )
    else:
        probes = sorted(electrodes["group_name"].unique())
        for probe in probes:
            figures.append(
                _plot_electrodes_implant_hole(
                    session,
                    probe,
                    electrodes,
                    ccf_volume,
                    probe_insertion_db_connection,
                )
            )

    return tuple(figures)


def plot_sensory_responses(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, ...]:
    from npc_sessions_cache.figures.paper2 import fig3c

    trials = session.trials[:]
    units = session.units[:].query(
        "amplitude_cutoff < 0.1 & isi_violations_ratio < 0.5 & presence_ratio == 1"
    )

    block_catch_vis_aud_startstop = [
        [
            trials.query(f"block_index == {block_index} & {stim}")[
                ["start_time", "stop_time"]
            ].values
            for stim in ["is_catch", "is_vis_stim", "is_aud_stim"]
        ]
        for block_index in trials["block_index"].unique()
    ]
    records = []
    for _, unit in units.iterrows():
        unit_id = unit["unit_id"]
        unit_spike_times = unit["spike_times"]
        block_resp = {"catch": [], "vis": [], "aud": []}
        for catch_vis_aud_startstop in block_catch_vis_aud_startstop:
            catch_vis_aud_counts: list[int] = []
            for start_stop in catch_vis_aud_startstop:
                times = [
                    (
                        unit_spike_times[slice(start, stop)]
                        if 0 <= start < stop <= len(unit_spike_times)
                        else []
                    )
                    for start, stop in np.searchsorted(unit_spike_times, start_stop)
                ]
                if not times or not any(times):
                    catch_vis_aud_counts.append(0)
                else:
                    catch_vis_aud_counts.append(
                        len(np.concatenate(times)) / start_stop.shape[0]
                    )  # divide by number of trials
            for count, block_name in zip(catch_vis_aud_counts, block_resp):
                block_resp[block_name].append(count)
            for count, block_name in zip(catch_vis_aud_counts, block_resp):
                block_resp[block_name].append(count)
        records.append(
            {
                "unit_id": unit_id,
                "vis_resp": np.median(
                    v := np.subtract(block_resp["vis"], block_resp["aud"])
                ),
                "aud_resp": np.median(
                    v := np.subtract(block_resp["aud"], block_resp["vis"])
                ),
                "stim_resp": np.median(
                    v := np.subtract(
                        np.add(block_resp["aud"], block_resp["vis"]) * 0.5,
                        block_resp["catch"],
                    )
                ),
            }
        )
    stim_resp_df = pd.DataFrame(records)
    max_probe_unit_ids = []
    for _, probe_df in units.merge(stim_resp_df, on="unit_id").groupby(
        "electrode_group_name"
    ):
        for col in ["vis_resp", "aud_resp", "stim_resp"]:
            if probe_df[col].std() == 0:
                continue  # all values are the same (not applicable, lack of trials)
            unit_id = probe_df.sort_values(col, ascending=False).iloc[0]["unit_id"]
            if unit_id not in max_probe_unit_ids:
                max_probe_unit_ids.append(unit_id)

    figs = []
    for unit_id in max_probe_unit_ids:
        figs.append(
            fig3c.plot(
                unit_id=unit_id,
                max_psth_spike_rate=200,
                session=session,
                xlim_0=-0.25,
                xlim_1=0.75,
            )
        )
    return tuple(figs)


def plot_drift_maps(
    session: npc_sessions.DynamicRoutingSession,
    probe_letter: str | npc_session.ProbeRecord | None = None,
) -> tuple[plt.Figure, ...] | None:
    if not session.is_sorted:
        return None
    if probe_letter:
        probes = (probe_letter,)
    else:
        probes = session.probe_letters_to_use
    paths = [
        upath.UPath(
            f"s3://aind-scratch-data/dynamic-routing/drift_maps/{session.session_id}_{probe}.png"
        )
        for probe in probes
    ]
    if not any(path.exists() for path in paths):
        print(
            f"Drift maps not found for {session.id}: running drift map generation in codeocean (takes ~1 minute)"
        )
        try:
            computation = run_drift_map_capsule(session.id)
        except ValueError:
            return None
        t0 = time.time()

        while aind_session.get_codeocean_model(
            computation.id, is_computation=True
        ).state not in [
            codeocean.computation.ComputationState.Completed,
            codeocean.computation.ComputationState.Failed,
        ]:
            if time.time() - t0 > 20 * 60:
                raise TimeoutError(
                    f"Drift map computation took >20 mins to complete: {computation.id}"
                )
            time.sleep(10)

    figs = []
    for path in sorted(paths):
        if not path.exists():
            continue
        fig = plt.figure()
        plt.imshow(plt.imread(io.BytesIO(path.read_bytes())))
        plt.gca().axis("off")
        plt.suptitle(path.stem, fontsize=8)
        fig.set_layout_engine("tight")
        figs.append(fig)
    return tuple(figs)


def run_drift_map_capsule(session_id: str) -> codeocean.computation.Computation:
    record = npc_session.SessionRecord(session_id)
    session: aind_session.Session = aind_session.get_sessions(
        subject_id=record.subject, date=record.date
    )[0]
    raw_asset = session.raw_data_asset
    sorted_assets = session.ecephys.sorter.kilosort2_5.sorted_data_assets
    if not sorted_assets:
        raise ValueError(f"{session} has not sorted assets")
    sorted_asset = sorted_assets[-1]
    return aind_session.get_codeocean_client().computations.run_capsule(
        codeocean.computation.RunParams(
            capsule_id="556afd63-a439-4fd5-8e37-705ff059ea93",
            data_assets=[
                codeocean.computation.DataAssetsRunParam(id=asset.id, mount=asset.name)
                for asset in [raw_asset, sorted_asset]
            ],
        )
    )
