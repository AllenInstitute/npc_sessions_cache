from __future__ import annotations

import datetime
import itertools
from typing import TYPE_CHECKING, Iterable, Literal
import wave

import matplotlib
import matplotlib.axes
import matplotlib.figure
import npc_samstim
import npc_sync
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

import npc_sessions

MIC_BIT_VOLTS = 0.0003052 # from structure.oebin

def get_audio_waveforms(
    session: npc_sessions.DynamicRoutingSession,
    start_times_on_sync: Iterable[float],
    duration_sec: float,
    resampling_factor: int | float | None = None,
) -> tuple[npc_samstim.SimpleWaveform | None, ...]:
    """Extract sections of audio recording.
    
    - resulting length of samples will be original * resampling_factor, if not
      None
    """
    return npc_samstim.get_waveforms_from_nidaq_recording(
        start_times_on_sync=start_times_on_sync,
        duration_sec=duration_sec,
        sync=session.sync_data,
        recording_dirs=session.ephys_recording_dirs,
        waveform_type="audio",
        resampling_factor=resampling_factor,
    )


def plot_microphone_response(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    """
    - plot difference between audio volume and baseline
    - take 90th percentile of voltage signals at:
            - quiescent period start time 
            - audio stim start time 
        - with duration of audio stim
        - preferred to the max, which could be a single outlier, and mean/median
          which might not capture AM noise well
    - only for task trials with aud stim 
    - plot on time course of entire task to make it easier to debug
    """
    aud_trials = session.trials[:].query("is_aud_stim")
    baseline_start_times = aud_trials["quiescent_start_time"].to_numpy()
    stim_start_times = aud_trials["stim_start_time"].to_numpy()
    waveforms = get_audio_waveforms(
        session=session,
        start_times_on_sync=np.concatenate([baseline_start_times, stim_start_times]),
        duration_sec=np.nanmedian(aud_trials["stim_stop_time"].to_numpy() - stim_start_times),
        resampling_factor=None,
    )
    baseline_volumes = [np.percentile(waveform.samples, 95) if waveform is not None else None for waveform in waveforms[: len(baseline_start_times)]]
    stim_volumes = [np.percentile(waveform.samples, 95) if waveform is not None else None for waveform in waveforms[len(baseline_start_times) :]]
    assert len(baseline_volumes) == len(baseline_start_times) and len(stim_volumes) == len(stim_start_times)
    volume_deltas = np.array([(stim - baseline if stim is not None and baseline is not None else np.nan) for stim, baseline in zip(stim_volumes, baseline_volumes)])
    fig, axes = plt.subplots(1,2, figsize=(6,3), sharey=True)
    ax = axes[0]
    ax.scatter(stim_start_times, volume_deltas * MIC_BIT_VOLTS * 1000, s=.8, c=['r' if abs(v) < 10 else 'k' for v in volume_deltas])
    if np.all(volume_deltas[~np.isnan(volume_deltas)] >= 0):
        ax.set_ylim(0)
    ax.set(xlabel="experiment time (s)", ylabel="stim - quiescent (mV)")
    ax = axes[1]
    ax.hist(volume_deltas * MIC_BIT_VOLTS * 1000, bins=10, orientation='horizontal', color='k')
    # ax = plt.gca()
    ax.set(xlabel="trials")
    fig.suptitle(f"mic response for aud stim trials in task\n{session.id}")
    return fig

def plot_audio_waveforms(
    session: npc_sessions.DynamicRoutingSession,
    target_stim: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a selection of audio waveforms aligned to stim start times. Default is
    one per block"""
    aud_trials = session.trials[:].query(f"is_aud_{'' if target_stim else 'non'}target")
    start_times = np.array([trials.iloc[0]['stim_start_time'] for _, trials in aud_trials.groupby("block_index")])
    front_padding = .02 # sec
    waveforms = get_audio_waveforms(
        session=session,
        start_times_on_sync=start_times - front_padding,
        duration_sec=front_padding + .12,
        resampling_factor=None,
    )
    fig, axes = plt.subplots(len(start_times), 1,  sharex=True, sharey=True)
    if isinstance(axes, matplotlib.axes.Axes):
        axes = [axes]
    for idx, (ax, waveform) in enumerate(zip(axes, waveforms)):
        ax: plt.Axes
        ax.axvline(0, c='grey', ls='--')
        ax.set_title(f"block {aud_trials['block_index'].unique()[idx]}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if waveform is not None:
            ax.plot(waveform.timestamps - front_padding, (waveform.samples - np.nanmedian(waveform.samples)) * MIC_BIT_VOLTS * 1000, lw=.1, c='k')
            if idx == len(axes) // 2:
                ax.set_ylabel('median-subtracted voltage (mV)')
        else:
            ax.text(0, 0, "no mic data in requested range", fontsize=8)
    ax.set_xlabel("time from 'stim start time' in trials table (s)")
    fig.suptitle(f"mic rec of {'' if target_stim else 'non'}target waveforms in task\n{session.id}")
    fig.set_figheight(1.2 * len(start_times))
    return fig
    
    
def get_audio_latencies(
    session: npc_sessions.DynamicRoutingSession, stim_type: Literal["task", "mapping"]
):
    if "task" in stim_type.lower() or "behavior" in stim_type.lower():
        sel_stim = next(s for s in session.stim_paths if "DynamicRouting" in s.stem)
    elif "rf" in stim_type.lower() or "mapping" in stim_type.lower():
        sel_stim = next(s for s in session.stim_paths if "RFMapping" in s.stem)

    # get nidaq latencies
    lat_nidaq_env = npc_samstim.get_stim_latencies_from_nidaq_recording(
        sel_stim,
        session.sync_path,
        session.ephys_recording_dirs,
        "audio",
        use_envelope=True,
    )

    lat_nidaq_sig = npc_samstim.get_stim_latencies_from_nidaq_recording(
        sel_stim,
        session.sync_path,
        session.ephys_recording_dirs,
        "audio",
        use_envelope=False,
    )

    lat_nidaq_sig_list = [y.latency for y in lat_nidaq_sig if y is not None]
    lat_nidaq_env_list = [y.latency for y in lat_nidaq_env if y is not None]

    sound_type_list = [y.name for y in lat_nidaq_sig if y is not None]

    audtridx = []
    for ii, lat in enumerate(lat_nidaq_sig):
        if lat is not None:
            audtridx.append(ii)

    # get sync latencies
    if session.sync_data.start_time.date() >= datetime.date(year=2023, month=8, day=31):
        sync_sound_on = True
    else:
        sync_sound_on = False

    if sync_sound_on:
        sync_line = npc_sync.get_sync_line_for_stim_onset(
            "audio", session.sync_data.start_time.date()
        )

        lat_sync = npc_samstim.get_stim_latencies_from_sync(
            sel_stim, session.sync_path, "audio", sync_line
        )

        lat_sync_list = [y.latency for y in lat_sync if y is not None]
        audtridx_sync = []
        for ii, lat in enumerate(lat_sync):
            if lat is not None:
                audtridx_sync.append(ii)

    latency_info = {
        "nidaq_signal": lat_nidaq_sig_list,
        "nidaq_envelope": lat_nidaq_env_list,
        "aud_trial_idx": audtridx,
        "sync_sound_on": sync_sound_on,
        "sound_type": sound_type_list,
    }

    if sync_sound_on:
        latency_info["sync"] = lat_sync_list
        latency_info["aud_trial_idx_sync"] = audtridx_sync

    return latency_info


def _plot_audio_latencies(session: npc_sessions.DynamicRoutingSession):
    latency_info = get_audio_latencies(session, "task")
    latency_flags = []
    if np.sum(np.array(latency_info["nidaq_signal"]) < 0) > 0:
        latency_flags.append("signal xcorr")
    if np.sum(np.array(latency_info["nidaq_envelope"]) < 0) > 0:
        latency_flags.append("envelope xcorr")
    if latency_info["sync_sound_on"] is True:
        if np.sum(np.array(latency_info["sync"]) < 0) > 0:
            latency_flags.append("sync")

    xbins = np.arange(-0.15, 0.15, 0.001)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(
        latency_info["aud_trial_idx"], latency_info["nidaq_signal"], ".", alpha=0.5
    )
    ax[0].plot(
        latency_info["aud_trial_idx"], latency_info["nidaq_envelope"], ".", alpha=0.5
    )

    ax[0].set_xlabel("trial number")
    ax[0].set_ylabel("audio latency (s)")

    ax[1].hist(latency_info["nidaq_signal"], bins=xbins, alpha=0.5)
    ax[1].hist(latency_info["nidaq_envelope"], bins=xbins, alpha=0.5)

    ax[1].set_xlabel("audio latency (s)")
    ax[1].set_ylabel("trial count")
    ax[1].legend(["signal", "envelope"])

    if latency_info["sync_sound_on"] is True:
        ax[0].plot(
            latency_info["aud_trial_idx_sync"], latency_info["sync"], ".", alpha=0.5
        )
        ax[1].hist(latency_info["sync"], bins=xbins, alpha=0.5)
        ax[1].legend(["signal", "envelope", "sync"])

    if session.task_version and "templeton" in session.task_version:
        figtitle = "Audio latency by alignment method: " + session.id + " (templeton)"
    else:
        figtitle = "Audio latency by alignment method: " + session.id + " (DR)"

    latency_warning = "Warning! negative latencies in: "
    if len(latency_flags) > 0:
        for flag in latency_flags:
            latency_warning = latency_warning + flag + ", "
        ax[0].set_title(latency_warning[:-2])

    fig.suptitle(figtitle)
    fig.tight_layout()

    return fig


def _plot_tone_vs_AMnoise(session: npc_sessions.DynamicRoutingSession, latency_info):
    # compare tones vs. AM noise
    tone_idx = np.asarray(latency_info["sound_type"]) == "tone"
    AMnoise_idx = np.asarray(latency_info["sound_type"]) == "AM_noise"

    aud_trial_idx = np.asarray(latency_info["aud_trial_idx"])
    nidaq_signal = np.asarray(latency_info["nidaq_signal"])
    nidaq_envelope = np.asarray(latency_info["nidaq_envelope"])
    if latency_info["sync_sound_on"] is True:
        sync = np.asarray(latency_info["sync"])
        aud_trial_idx_sync = np.asarray(latency_info["aud_trial_idx_sync"])

    xbins = np.arange(-0.15, 0.15, 0.001)
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))

    idxs = [tone_idx, AMnoise_idx]
    idx_labels = ["tone", "AM noise"]

    for xx, idx in enumerate(idxs):
        latency_flags = []
        if np.sum(np.array(nidaq_signal[idx]) < 0) > 0:
            latency_flags.append("signal xcorr")
        if np.sum(np.array(nidaq_envelope[idx]) < 0) > 0:
            latency_flags.append("envelope xcorr")
        if latency_info["sync_sound_on"] is True:
            if np.sum(np.array(sync[idx]) < 0) > 0:
                latency_flags.append("sync")

        ax[0, xx].plot(aud_trial_idx[idx], nidaq_signal[idx], ".", alpha=0.5)
        ax[0, xx].plot(aud_trial_idx[idx], nidaq_envelope[idx], ".", alpha=0.5)

        ax[0, xx].set_xlabel("trial number")
        ax[0, xx].set_ylabel("audio latency (s)")

        ax[1, xx].hist(nidaq_signal[idx], bins=xbins, alpha=0.5)
        ax[1, xx].hist(nidaq_envelope[idx], bins=xbins, alpha=0.5)

        ax[1, xx].set_xlabel("audio latency (s)")
        ax[1, xx].set_ylabel("trial count")
        ax[1, xx].legend(["signal", "envelope"])

        if latency_info["sync_sound_on"] is True:
            ax[0, xx].plot(aud_trial_idx_sync[idx], sync[idx], ".", alpha=0.5)
            ax[1, xx].hist(sync[idx], bins=xbins, alpha=0.5)
            ax[1, xx].legend(["signal", "envelope", "sync"])

        latency_warning = idx_labels[xx] + ": neg lats in: "
        if len(latency_flags) > 0:
            for flag in latency_flags:
                latency_warning = latency_warning + flag + ", "
            ax[0, xx].set_title(latency_warning[:-2])
        else:
            ax[0, xx].set_title(idx_labels[xx])

    if session.task_version and "templeton" in session.task_version:
        figtitle = (
            "Comp tone vs. AM noise latency: " + session.id + " RFMapping (templeton)"
        )
    else:
        figtitle = "Comp tone vs. AM noise latency: " + session.id + " RFMapping (DR)"

    fig.suptitle(figtitle)
    fig.tight_layout()

    return fig
