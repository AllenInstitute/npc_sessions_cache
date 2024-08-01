from typing import TYPE_CHECKING

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

import npc_sessions

import npc_sessions_cache.plots.plot_utils as plot_utils


def plot_performance_by_block(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    task_performance_by_block_df: pd.DataFrame = session.performance[:]

    dprime_threshold = 1.5 if session.is_training else 1.0
    n_passing_blocks = np.sum(task_performance_by_block_df["cross_modal_dprime"] >= dprime_threshold)
    failed_block_ind = task_performance_by_block_df["cross_modal_dprime"] < dprime_threshold

    # blockwise behavioral performance
    xvect = task_performance_by_block_df.index.values
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(
        xvect,
        task_performance_by_block_df["signed_cross_modal_dprime"],
        "ko-",
        label="cross-modal",
    )
    ax[0].plot(
        xvect[failed_block_ind],
        task_performance_by_block_df["signed_cross_modal_dprime"][failed_block_ind],
        "ro",
        label="failed",
    )
    ax[0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax[0].set_title(
        "cross-modal dprime: "
        + str(n_passing_blocks)
        + "/"
        + str(len(task_performance_by_block_df))
        + " blocks passed"
    )
    ax[0].set_ylabel("aud <- dprime -> vis")

    ax[1].plot(
        xvect, task_performance_by_block_df["vis_intra_dprime"], "go-", label="vis"
    )
    ax[1].plot(
        xvect, task_performance_by_block_df["aud_intra_dprime"], "bo-", label="aud"
    )
    ax[1].set_title("intra-modal dprime")
    ax[1].legend(["vis", "aud"])
    ax[1].set_xlabel("block index")
    ax[1].set_ylabel("dprime")

    fig.suptitle(session.id)
    fig.tight_layout()

    return fig


def plot_first_lick_latency_hist(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    # first lick latency histogram

    trials: pd.DataFrame = session.trials[:]

    xbins = np.arange(0, 1, 0.05)
    fig, ax = plt.subplots(1, 1)
    ax.hist(
        trials.query("is_vis_stim==True")["response_time"]
        - trials.query("is_vis_stim==True")["stim_start_time"],
        bins=xbins,
        alpha=0.5,
    )

    ax.hist(
        trials.query("is_aud_stim==True")["response_time"]
        - trials.query("is_aud_stim==True")["stim_start_time"],
        bins=xbins,
        alpha=0.5,
    )

    ax.legend(["vis stim", "aud stim"])
    ax.set_xlabel("lick latency (s)")
    ax.set_ylabel("trial count")
    ax.set_title("lick latency: " + session.id)

    return fig


def plot_lick_raster(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    timeseries = session.processing["behavior"]["licks"]
    trials: pd.DataFrame = session.trials[:]

    fig, ax = plt.subplots(1, 1)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    for tt, trial in trials.iterrows():
        trial_licks = (
            timeseries.timestamps[
                (timeseries.timestamps > trial["stim_start_time"] - 1)
                & (timeseries.timestamps < trial["stim_start_time"] + 2)
            ]
            - trial["stim_start_time"]
        )

        ax.vlines(trial_licks, tt, tt + 1)

    ax.set_xlim([-1, 2])
    ax.set_xlabel("time rel to stim onset (s)")
    ax.set_ylabel("trial number")
    ax.set_title(timeseries.description, fontsize=8)
    fig.suptitle(session.id, fontsize=10)

    return fig


def plot_running(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    timeseries = session.processing["behavior"]["running_speed"]
    epochs: pd.DataFrame = session.epochs[:]
    licks = session.processing["behavior"]["licks"]
    plt.style.use("seaborn-v0_8-notebook")

    fig, ax = plt.subplots()

    for _, epoch in epochs.iterrows():
        epoch_indices = (timeseries.timestamps >= epoch["start_time"]) & (
            timeseries.timestamps <= epoch["stop_time"]
        )
        if len(epoch_indices) > 0:
            ax.plot(
                timeseries.timestamps[epoch_indices],
                timeseries.data[epoch_indices],
                linewidth=0.1,
                alpha=1,
                color="k",
                label="speed",
                zorder=30,
            )
    k = 100 if "cm" in timeseries.unit else 1
    ymax = 0.8 * k
    ax.set_ylim([-0.05 * k, ymax])
    ax.vlines(
        licks.timestamps,
        *ax.get_ylim(),
        color="lime",
        linestyle="-",
        linewidth=0.05,
        zorder=10,
    )
    ax.hlines(
        0,
        0,
        max(timeseries.timestamps),
        color="k",
        linestyle="--",
        linewidth=0.5,
        zorder=20,
    )
    plot_utils.add_epoch_color_bars(ax, epochs, rotation=90, y=ymax, va="top")
    ax.margins(0)
    ax.set_frame_on(False)
    ax.set_ylabel(timeseries.unit)
    ax.set_xlabel(timeseries.timestamps_unit)
    title = timeseries.description
    if max(timeseries.data) > ax.get_ylim()[1]:
        title += f"\ndata clipped: {round(max(timeseries.data)) = } {timeseries.unit} at {timeseries.timestamps[np.argmax(timeseries.data)]:.0f} {timeseries.timestamps_unit}"
    ax.set_title(title, fontsize=8)
    fig.suptitle(session.id, fontsize=10)
    fig.set_size_inches(10, 4)
    fig.set_layout_engine("tight")
    return fig


def plot_response_rate_by_stimulus_type(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:

    trials = session.trials[:]
    start_time = trials.iloc[0]['start_time']
    end_time = trials.iloc[-1]['stop_time']
    
    switch_times = trials[trials['is_context_switch']]['start_time']
    switch_starts = np.insert(switch_times, 0, start_time)
    switch_ends = np.append(switch_times, end_time)
    switch_durations = switch_ends - switch_starts
    
    window_size = 120 #seconds
    
    time = np.arange(start_time,end_time,window_size)
    
    stim_types = ['vis_target', 'aud_target', 'vis_nontarget', 'aud_nontarget']
    rate_dict = {stim_type:[] for stim_type in stim_types}
    for stim_type in stim_types:
        
        response_times = trials[trials[f'is_{stim_type}'] & trials['is_response']]['stim_start_time'].values
        trial_type_times = trials[trials[f'is_{stim_type}']]['stim_start_time'].values
    
        rt_hist, _ = np.histogram(response_times, bins = time)
        tt_hist, _ = np.histogram(trial_type_times, bins = time)
        
        rate_dict[stim_type] = rt_hist/tt_hist

    aud_block_inds = np.arange(0, len(switch_starts), 2) + trials.iloc[0]['is_vis_context']

    fig, ax = plt.subplots()
    for stim_type in stim_types:
        ax.plot(time[:-1], rate_dict[stim_type])
        
    ax.set_ylabel('Response Rate')
    ax.set_xlabel('Session Time (s)')
    
    for aud_block in aud_block_inds:
        rectangle = Rectangle((switch_starts[aud_block], 0), switch_durations[aud_block], 1, color='k', alpha=0.2)
        ax.add_artist(rectangle)
        
    ax.legend(stim_types + ['aud_block'])
    return fig
