"""Rasters of units with large, short-latency visual or auditory responses.

Used as a timing check: stimulus onsets in the trials table come from frame display times (vis)
or the sound line / NI-DAQ (aud). A frame offset (16.7 ms) in some trials blurs or shifts the
onset of a fast response in those trials. Trials overlapping a replayed (invalid) sync interval,
where event times were reconstructed by the sync patch, are highlighted.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

import matplotlib.pyplot as plt
import npc_sessions
import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

Modality = Literal["vis", "aud"]

RESPONSE_WINDOW: dict[Modality, tuple[float, float]] = {
    "vis": (0.02, 0.10),
    "aud": (0.005, 0.05),
}
"""Window after stim onset (s) for the response, per modality: short, to favour fast responses"""
BASELINE_WINDOW = (-0.1, 0.0)
MIN_RATE_INCREASE_HZ = 10.0
MIN_SCORE = 5.0
"""Mean / SEM of the per-trial (response - baseline) rate difference"""
N_UNITS_PER_MODALITY = 4
REPLAY_PAD = 0.05
"""Onsets within this many seconds of a replayed sync interval count as replayed"""
RASTER_XLIM = (-0.1, 0.25)
PSTH_BIN = 0.002
UNIT_QUERY = (
    "amplitude_cutoff < 0.1 & isi_violations_ratio < 0.5 & presence_ratio > 0.9"
)


def _counts_in_window(
    spike_times: npt.NDArray[np.floating],
    onsets: npt.NDArray[np.floating],
    window: tuple[float, float],
) -> npt.NDArray[np.int64]:
    return np.searchsorted(spike_times, onsets + window[1]) - np.searchsorted(
        spike_times, onsets + window[0]
    )


def _aligned_spikes(
    spike_times: npt.NDArray[np.floating],
    onsets: npt.NDArray[np.floating],
    window: tuple[float, float],
) -> list[npt.NDArray[np.floating]]:
    lo = np.searchsorted(spike_times, onsets + window[0])
    hi = np.searchsorted(spike_times, onsets + window[1])
    return [spike_times[a:b] - t for a, b, t in zip(lo, hi, onsets)]


def _onset_latency(
    aligned: list[npt.NDArray[np.floating]], n_trials: int
) -> float:
    """First 2-ms bin after onset (smoothed over 3 bins) where the rate exceeds the baseline
    mean + 5 SD; nan if none within 0.15 s."""
    bins = np.arange(BASELINE_WINDOW[0], 0.15 + PSTH_BIN, PSTH_BIN)
    if not aligned:
        return np.nan
    counts = np.histogram(np.concatenate(aligned), bins)[0] / n_trials / PSTH_BIN
    smoothed = np.convolve(counts, np.ones(3) / 3, mode="same")
    centers = bins[:-1] + PSTH_BIN / 2
    base = smoothed[centers < 0]
    threshold = base.mean() + 5 * max(base.std(), 1.0)
    above = np.flatnonzero((centers >= 0) & (smoothed > threshold))
    return float(bins[above[0]]) if above.size else np.nan


def _get_interval_df(
    session: npc_sessions.DynamicRoutingSession, name: str
) -> pd.DataFrame | None:
    try:
        return session.intervals[name][:]
    except (KeyError, AttributeError):
        return None


def get_trial_onsets(
    session: npc_sessions.DynamicRoutingSession,
) -> dict[Modality, pd.DataFrame]:
    """Stimulus onsets per modality, from the task (vis / aud trials without opto) and RF mapping
    (full-field flashes / all sounds), sorted by time. Columns: start_time, stop_time,
    stim_start_time (NaN rows dropped, counted in `.attrs['n_nan']`), source ('task' or 'rf'),
    is_replayed (stim onset inside a replayed sync interval, +-`REPLAY_PAD`). NaN onsets in
    trials overlapping a replayed interval are counted in `.attrs['n_nan']['replayed']`."""
    invalid = tuple(session.sync_data.invalid_intervals) if session.is_sync else ()
    parts: dict[Modality, list[pd.DataFrame]] = {"vis": [], "aud": []}
    cols = ["start_time", "stop_time", "stim_start_time"]
    if session.is_task:
        trials = session.trials[:]
        if "is_opto" in trials:
            # opto light evokes responses / artifacts at stim onset
            trials = trials[~trials["is_opto"].astype(bool)]
        parts["vis"].append(trials[trials["is_vis_stim"]][cols].assign(source="task"))
        parts["aud"].append(trials[trials["is_aud_stim"]][cols].assign(source="task"))
    vis_rf = _get_interval_df(session, "vis_rf_mapping_trials")
    if vis_rf is not None and "is_full_field_flash" in vis_rf:
        parts["vis"].append(
            vis_rf[vis_rf["is_full_field_flash"].astype(bool)][cols].assign(source="rf")
        )
    aud_rf = _get_interval_df(session, "aud_rf_mapping_trials")
    if aud_rf is not None:
        parts["aud"].append(aud_rf[cols].assign(source="rf"))
    out: dict[Modality, pd.DataFrame] = {}
    for modality, dfs in parts.items():
        df = (
            pd.concat(dfs, ignore_index=True)
            if dfs
            else pd.DataFrame(columns=[*cols, "source"])
        )
        onset = df["stim_start_time"].to_numpy(dtype=float)
        overlaps = np.zeros(len(df), dtype=bool)
        is_replayed = np.zeros(len(df), dtype=bool)
        for a, b in invalid:
            overlaps |= (df["stop_time"].to_numpy(dtype=float) >= a) & (
                df["start_time"].to_numpy(dtype=float) <= b
            )
            # onset inside the interval (+-50 ms): its time was reconstructed by the patch
            is_replayed |= (onset >= a - REPLAY_PAD) & (onset <= b + REPLAY_PAD)
        df = df.assign(is_replayed=is_replayed)
        nan = df["stim_start_time"].isna()
        n_nan = {
            "all": int(nan.sum()),
            "replayed": int((nan & overlaps).sum()),
        }
        df = df[~nan].sort_values("stim_start_time").reset_index(drop=True)
        df.attrs["n_nan"] = n_nan
        out[modality] = df
    return out


def get_fast_responsive_units(
    session: npc_sessions.DynamicRoutingSession,
    trials_by_modality: dict[Modality, pd.DataFrame] | None = None,
    n_per_modality: int = N_UNITS_PER_MODALITY,
) -> pd.DataFrame:
    """One row per (unit, modality) for units with a large, fast response, ranked by rate
    increase. Columns: unit_id, modality, structure, electrode_group_name, rate_increase_hz,
    score, latency_s, n_trials."""
    trials_by_modality = trials_by_modality or get_trial_onsets(session)
    units = session.units[:]
    try:
        units = units.query(UNIT_QUERY)
    except Exception as exc:  # metrics columns missing
        logger.warning(f"{session.id}: unit QC query not applied: {exc!r}")
    records = []
    for modality, trials in trials_by_modality.items():
        if (trials["source"] == "task").sum() >= 10:
            trials = trials[trials["source"] == "task"]  # select on task trials
        onsets = trials["stim_start_time"].to_numpy()
        if onsets.size < 10:
            continue
        win = RESPONSE_WINDOW[modality]
        for _, unit in units.iterrows():
            st = np.asarray(unit["spike_times"])
            resp = _counts_in_window(st, onsets, win) / (win[1] - win[0])
            base = _counts_in_window(st, onsets, BASELINE_WINDOW) / (
                BASELINE_WINDOW[1] - BASELINE_WINDOW[0]
            )
            d = resp - base
            sem = d.std(ddof=1) / np.sqrt(d.size)
            records.append(
                dict(
                    unit_id=unit["unit_id"],
                    modality=modality,
                    structure=unit.get("structure", None),
                    electrode_group_name=unit.get("electrode_group_name", None),
                    rate_increase_hz=float(d.mean()),
                    score=float(d.mean() / sem) if sem > 0 else 0.0,
                    n_trials=int(onsets.size),
                )
            )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df[(df.rate_increase_hz >= MIN_RATE_INCREASE_HZ) & (df.score >= MIN_SCORE)]
    df = (
        df.sort_values("rate_increase_hz", ascending=False)
        .groupby("modality")
        .head(n_per_modality * 3)  # candidates: rank by latency next
    )
    spikes = units.set_index("unit_id")["spike_times"]
    latencies = []
    for _, row in df.iterrows():
        onsets = trials_by_modality[row.modality]["stim_start_time"].to_numpy()
        aligned = _aligned_spikes(
            np.asarray(spikes[row.unit_id]), onsets, (BASELINE_WINDOW[0], 0.15)
        )
        latencies.append(_onset_latency(aligned, onsets.size))
    df = df.assign(latency_s=latencies)
    # big and fast: among the largest responders, prefer the shortest latency
    df = (
        df.dropna(subset=["latency_s"])
        .sort_values(["modality", "latency_s", "rate_increase_hz"], ascending=[True, True, False])
        .groupby("modality")
        .head(n_per_modality)
        .reset_index(drop=True)
    )
    return df


SHIFTS = np.arange(-0.05, 0.0505, 0.001)
"""Candidate onset shifts (s) for the per-trial shift estimate"""
SHIFT_WINDOW = (-0.05, 0.2)
"""Spikes in this window after stim onset are used for the per-trial shift estimate"""


def estimate_trial_shifts(
    spike_times_per_unit: list[npt.NDArray[np.floating]],
    onsets: npt.NDArray[np.floating],
    template_mask: npt.NDArray[np.bool_],
    return_ll: bool = False,
):
    """Per-trial onset shift (s): the shift s maximizing the Poisson log-likelihood of the
    trial's spikes (all units) under each unit's mean rate, λ_u(t - s), estimated from the
    trials in `template_mask` (1-ms bins, 5-ms boxcar). A trial whose stim onset is
    reported one frame early has s ≈ +16.7 ms."""
    dt = 0.001
    pad = SHIFTS.max() + 0.01
    edges = np.arange(SHIFT_WINDOW[0] - pad, SHIFT_WINDOW[1] + pad + dt, dt)
    centers = edges[:-1] + dt / 2
    win_bins = (centers >= SHIFT_WINDOW[0]) & (centers < SHIFT_WINDOW[1])
    ll = np.zeros((len(onsets), len(SHIFTS)))
    for st in spike_times_per_unit:
        aligned = _aligned_spikes(st, onsets, (edges[0], edges[-1]))
        counts = np.array([np.histogram(a, edges)[0] for a in aligned], dtype=float)
        rate = counts[template_mask].mean(axis=0) / dt
        rate = np.convolve(rate, np.ones(5) / 5, mode="same") + 0.5  # floor: 0.5 Hz
        log_rate = np.log(rate)
        for k, s in enumerate(SHIFTS):
            n = int(round(s / dt))
            # λ(t - s) on the window bins: shift the template right by n bins
            idx = np.flatnonzero(win_bins) - n
            ll[:, k] += counts[:, win_bins] @ log_rate[idx] - rate[idx].sum() * dt
    if return_ll:
        return SHIFTS[np.argmax(ll, axis=1)], ll
    return SHIFTS[np.argmax(ll, axis=1)]


def frame_log_likelihood_ratio(ll: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Per trial: log-likelihood at shift 0 minus the larger of shifts +-1 frame (17 ms).
    Positive: the trial's spikes favour the reported onset over a 1-frame offset."""
    i0 = int(np.argmin(np.abs(SHIFTS)))
    ip = int(np.argmin(np.abs(SHIFTS - 1 / 60)))
    im = int(np.argmin(np.abs(SHIFTS + 1 / 60)))
    return ll[:, i0] - np.maximum(ll[:, ip], ll[:, im])


def _plot_raster(
    ax_raster: plt.Axes,
    ax_psth: plt.Axes | None,
    spike_times: npt.NDArray[np.floating],
    trials: pd.DataFrame,
    title: str,
) -> None:
    onsets = trials["stim_start_time"].to_numpy()
    replayed = trials["is_replayed"].to_numpy()
    window = (RASTER_XLIM[0] - PSTH_BIN, RASTER_XLIM[1] + PSTH_BIN)
    aligned = _aligned_spikes(spike_times, onsets, window)
    for i, (sp, rep) in enumerate(zip(aligned, replayed)):
        ax_raster.vlines(
            sp,
            i + 0.5,
            i + 1.5,
            color="r" if rep else "k",
            lw=1.0 if rep else 0.4,
        )
        if rep:
            ax_raster.axhspan(i + 0.5, i + 1.5, color="r", alpha=0.15, lw=0)
    source = trials["source"].to_numpy()
    for i in np.flatnonzero(source[1:] != source[:-1]):
        ax_raster.axhline(i + 1.5, c="g", lw=0.5)
    ax_raster.axvline(0, c="b", lw=0.5, ls="--")
    ax_raster.set_ylim(0.5, len(onsets) + 0.5)
    ax_raster.set_xlim(*RASTER_XLIM)
    ax_raster.set_ylabel("trial (time order)")
    ax_raster.set_title(title, fontsize=6)
    if ax_psth is None:
        return
    bins = np.arange(window[0], window[1] + PSTH_BIN, PSTH_BIN)
    all_sp = np.concatenate(aligned) if aligned else np.array([])
    ax_psth.plot(
        bins[:-1] + PSTH_BIN / 2,
        np.histogram(all_sp, bins)[0] / len(onsets) / PSTH_BIN,
        c="k",
        lw=0.6,
        label="all",
    )
    if replayed.any():
        rep_sp = [s for s, r in zip(aligned, replayed) if r]
        ax_psth.plot(
            bins[:-1] + PSTH_BIN / 2,
            np.histogram(np.concatenate(rep_sp), bins)[0] / len(rep_sp) / PSTH_BIN,
            c="r",
            lw=0.6,
            label=f"replayed ({len(rep_sp)})",
        )
        ax_psth.legend(fontsize=5)
    ax_psth.axvline(0, c="b", lw=0.5, ls="--")
    ax_psth.set_xlim(*RASTER_XLIM)
    ax_psth.set_xlabel("time from stim_start_time (s)")
    ax_psth.set_ylabel("rate (Hz)")


def plot_fast_sensory_responses(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure | dict, ...] | None:
    """Spike rasters for units with large, short-latency vis or aud responses, aligned to
    `stim_start_time`, plus a summary of first-spike latency vs trial time.

    - one figure per unit: rasters for vis and aud trials (rows in time order), 2-ms PSTH below
    - trials with the stim onset inside a replayed sync interval are red; their PSTH is overlaid
    - last figure: per-trial median first-spike latency across the selected units, vs time in
      the session, with replayed intervals marked. A frame offset in a subset of vis trials
      appears as a ~17 ms step
    - last element: table of the selected units (dict)
    """
    if not session.is_sorted:
        return None
    trials_by_modality = get_trial_onsets(session)
    if all(len(trials) < 10 for trials in trials_by_modality.values()):
        # no vis/aud stims (e.g. opto-only sessions): not applicable
        return None
    selected = get_fast_responsive_units(session, trials_by_modality)
    if selected.empty:
        # weak responses (e.g. muscimol days, few probes in sensory areas) - not an error
        return (
            {
                "n_selected": 0,
                "criteria": f"rate increase >= {MIN_RATE_INCREASE_HZ} Hz and score >= {MIN_SCORE} in {RESPONSE_WINDOW}",
                "n_onsets": {m: len(t) for m, t in trials_by_modality.items()},
            },
        )
    spikes = session.units[:].set_index("unit_id")["spike_times"]
    invalid = tuple(session.sync_data.invalid_intervals)
    figs: list[plt.Figure | dict] = []
    for _, row in selected.iterrows():
        st = np.asarray(spikes[row.unit_id])
        fig, axes = plt.subplots(
            3,
            2,
            figsize=(6, 8),
            gridspec_kw={"height_ratios": [4, 1, 2]},
            sharex=True,
        )
        for col, modality in enumerate(("vis", "aud")):
            trials = trials_by_modality[modality]
            if trials.empty:
                axes[0, col].set_title(f"no {modality} trials", fontsize=6)
                continue
            counts = trials["source"].value_counts().to_dict()
            n_rep = int(trials["is_replayed"].sum())
            n_rep_nan = trials.attrs["n_nan"]["replayed"]
            _plot_raster(
                axes[0, col],
                axes[1, col],
                st,
                trials,
                f"{modality}: " + ", ".join(f"{k} {v}" for k, v in counts.items())
                + f"\nreplayed {n_rep} (+{n_rep_nan} with NaN onset)",
            )
            # zoom: trials within 30 s of a replayed interval
            near = np.zeros(len(trials), dtype=bool)
            t = trials["stim_start_time"].to_numpy()
            for a, b in invalid:
                near |= (t >= a - 30) & (t <= b + 30)
            if near.any():
                _plot_raster(
                    axes[2, col],
                    None,
                    st,
                    trials[near],
                    f"trials within 30 s of a replayed interval ({int(near.sum())})",
                )
            else:
                axes[2, col].set_title("no trials within 30 s of a replayed interval", fontsize=6)
            axes[2, col].set_xlabel("time from stim_start_time (s)")
        fig.suptitle(
            f"{row.unit_id} {row.structure or ''} ({row.electrode_group_name})\n"
            f"selected for {row.modality}: +{row.rate_increase_hz:.0f} Hz, "
            f"latency {1000 * row.latency_s:.0f} ms, score {row.score:.0f}\n"
            f"{len(invalid)} replayed sync interval(s): {[tuple(round(v, 1) for v in i) for i in invalid]}",
            fontsize=7,
        )
        fig.set_layout_engine("tight")
        figs.append(fig)

    # first-spike latency per trial vs session time
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, modality in zip(axes, ("vis", "aud")):
        trials = trials_by_modality[modality]
        sel = selected.query("modality == @modality")
        unit_ids = sel.unit_id
        if trials.empty or unit_ids.empty:
            ax.set_title(f"{modality}: no selected units", fontsize=7)
            continue
        onsets = trials["stim_start_time"].to_numpy()
        lat = np.full((len(unit_ids), len(onsets)), np.nan)
        for i, (uid, unit_lat) in enumerate(zip(unit_ids, sel.latency_s)):
            # first spike from 10 ms before to 40 ms after the unit's response onset
            for j, sp in enumerate(
                _aligned_spikes(
                    np.asarray(spikes[uid]),
                    onsets,
                    (max(unit_lat - 0.01, 0), unit_lat + 0.04),
                )
            ):
                if sp.size:
                    lat[i, j] = sp[0]
        median_lat = np.nanmedian(lat, axis=0)
        rep = trials["is_replayed"].to_numpy()
        is_rf = (trials["source"] == "rf").to_numpy()
        ax.scatter(onsets[~rep & ~is_rf], 1000 * median_lat[~rep & ~is_rf], s=3, c="k", label="task")
        ax.scatter(onsets[~rep & is_rf], 1000 * median_lat[~rep & is_rf], s=3, c="g", label="rf")
        ax.scatter(onsets[rep], 1000 * median_lat[rep], s=12, c="r", label="replayed")
        for a, b in invalid:
            ax.axvspan(a, b, color="r", alpha=0.3, lw=0)
        ax.set_ylabel(f"{modality}: median first\nspike latency (ms)")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_title(
            f"{modality}: {len(unit_ids)} units; median latency all trials "
            f"{1000 * np.nanmedian(median_lat):.1f} ms, replayed trials "
            f"{1000 * np.nanmedian(median_lat[rep]) if rep.any() else np.nan:.1f} ms",
            fontsize=7,
        )
    axes[-1].set_xlabel("stim_start_time (s on sync)")
    fig.suptitle(f"{session.id} first-spike latency", fontsize=8)
    fig.set_layout_engine("tight")
    figs.append(fig)

    # per-trial onset shift (likelihood fit to the mean response), replayed vs other trials
    shift_summary: dict[str, dict] = {}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, modality in zip(axes, ("vis", "aud")):
        trials = trials_by_modality[modality]
        unit_ids = selected.query("modality == @modality").unit_id
        if trials.empty or unit_ids.empty:
            ax.set_title(f"{modality}: no selected units", fontsize=7)
            continue
        rep = trials["is_replayed"].to_numpy()
        # template per source: task and RF stimuli evoke different responses
        shifts = np.full(len(trials), np.nan)
        llr = np.full(len(trials), np.nan)
        for src in trials["source"].unique():
            is_src = (trials["source"] == src).to_numpy()
            if (is_src & ~rep).sum() < 10:
                continue
            shifts[is_src], ll = estimate_trial_shifts(
                [np.asarray(spikes[u]) for u in unit_ids],
                trials["stim_start_time"].to_numpy()[is_src],
                template_mask=~rep[is_src],
                return_ll=True,
            )
            llr[is_src] = frame_log_likelihood_ratio(ll)
        valid = ~np.isnan(shifts)
        rep, shifts, llr = rep[valid], shifts[valid], llr[valid]
        trials = trials[valid]
        other = shifts[~rep]
        ax.hist(1000 * other, bins=1000 * np.append(SHIFTS, SHIFTS[-1] + 0.001) - 0.5, color="k")
        for v in shifts[rep]:
            ax.axvline(1000 * v, c="r", lw=1)
        for frame in (-1000 / 60, 1000 / 60):
            ax.axvline(frame, c="b", ls=":", lw=0.5)
        q = np.percentile(1000 * other, [2.5, 50, 97.5])
        ax.set_title(
            f"{modality}: {len(unit_ids)} units, {len(other)} other trials "
            f"(2.5/50/97.5%: {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} ms)\n"
            f"replayed (red): {[round(1000 * v) for v in shifts[rep]]} ms, "
            f"LLR(0 vs ±1 frame) {[round(float(v), 1) for v in llr[rep]]}; dotted: ±1 frame",
            fontsize=6,
        )
        ax.set_xlabel("estimated onset shift (ms)")
        shift_summary[modality] = {
            "n_units": int(len(unit_ids)),
            "other_trials_ms_p2.5_p50_p97.5": [float(v) for v in q],
            "other_trials_frac_abs_ge_1_frame": float(np.mean(np.abs(other) >= 0.0125)),
            "other_trials_frame_llr_p2.5_p50_p97.5": [
                float(v) for v in np.percentile(llr[~rep], [2.5, 50, 97.5])
            ],
            "other_trials_frac_llr_gt_0": float(np.mean(llr[~rep] > 0)),
            "replayed_trials": [
                {
                    "stim_start_time": float(t),
                    "source": str(src),
                    "shift_ms": float(1000 * v),
                    "frame_llr": float(r),
                    "percentile_among_other_trials": float(
                        100 * np.mean(other < v) + 50 * np.mean(other == v)
                    ),
                }
                for t, src, v, r in zip(
                    trials["stim_start_time"].to_numpy()[rep],
                    trials["source"].to_numpy()[rep],
                    shifts[rep],
                    llr[rep],
                )
            ],
        }
    fig.suptitle(f"{session.id}: per-trial onset shift vs mean response", fontsize=8)
    fig.set_layout_engine("tight")
    figs.append(fig)
    figs.append(
        {
            "units": json.loads(selected.to_json(orient="records")),
            "invalid_intervals": [list(i) for i in invalid],
            "n_replayed_trials": {
                m: int(t["is_replayed"].sum()) for m, t in trials_by_modality.items()
            },
            "n_nan_onsets": {m: t.attrs["n_nan"] for m, t in trials_by_modality.items()},
            "onset_shifts": shift_summary,
        }
    )
    return tuple(figs)
