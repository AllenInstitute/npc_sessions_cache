"""Do neural responses follow the photodiode on task trials whose visual stim frame was late?

`stim_start_time - quiescent_stop_time` is about constant for visual task trials (the stim
frame's display time minus the vsync of the frame before: one frame plus the monitor latency).
In ~5-10% of trials the stim frame is displayed a refresh later than usual (the frame before
it was held: hub experiments/timing-quartile, out/proxy_survey.csv). For those trials the
photodiode display time and the vsync-based time disagree by a refresh, so the neural
response says which is right:

- display times right: the response of 'late' trials is aligned with the 'normal' trials'
  response when both are aligned to `stim_start_time` (shift ~0), and is a refresh later when
  aligned to the vsync (`quiescent_stop_time` + the median offset)
- display times wrong (the late trials' diode flip assigned to the wrong frame): the late
  trials respond a refresh *earlier* than normal trials relative to `stim_start_time`

This is the only check of display times against an independent measurement (the brain) at
the level of single trials. In the 126 sorted v0.0.303 sessions with >= 12 late trials (hub
experiments/timing-quartile/pop_late_frame.py), the shift relative to `stim_start_time` was
-4 to +5 ms (median 0) and relative to the vsync +11 to +21 ms (median 16): none was near
-17 ms.

Shift: lag (1 ms steps, +-30 ms) maximising the correlation between the late trials'
population PSTH and the normal trials' (mean over units of the per-unit z-scored PSTH, 1-ms
bins, 5-ms boxcar), 0-150 ms after onset; 95% CI: 200 bootstrap resamples of the late trials
(fixed seed).
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import npc_sessions
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.ndimage

logger = logging.getLogger(__name__)

UNIT_QUERY = "amplitude_cutoff < 0.1 & isi_violations_ratio < 0.5 & presence_ratio > 0.9"
FRAME_MS = 1e3 / 60
LATE_MIN_MS = 0.5 * FRAME_MS
"""A trial is 'late' if its stim_start_time - quiescent_stop_time is this much more than the
median"""
MIN_LATE_TRIALS = 5
BIN = 0.001
EDGES = np.arange(-0.15, 0.30 + BIN, BIN)
CENTERS = EDGES[:-1] + BIN / 2
SHIFT_WINDOW = (0.0, 0.15)
MAX_LAG_MS = 30
N_BOOTSTRAP = 200
SEED = 0
MAX_DISPLAY_SHIFT_MS = 0.5 * FRAME_MS
"""Pass if the late trials' shift relative to stim_start_time is within this of 0 (and closer
to 0 than to minus the late trials' extra delay)"""


def _counts(spikes: npt.NDArray[np.float64], onsets: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    out = np.zeros((len(onsets), len(EDGES) - 1), np.float32)
    lo = np.searchsorted(spikes, onsets + EDGES[0])
    hi = np.searchsorted(spikes, onsets + EDGES[-1])
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = np.histogram(spikes[a:b] - onsets[i], EDGES)[0]
    return out


def _zpsth(c: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """units x trials x bins -> population PSTH (mean of per-unit z-scored, smoothed PSTHs)"""
    p = scipy.ndimage.uniform_filter1d(c.mean(axis=1).astype(float), 5, axis=-1)
    sd = p.std(axis=-1, keepdims=True)
    sd[sd == 0] = 1
    return ((p - p.mean(axis=-1, keepdims=True)) / sd).mean(axis=0)


def _shift_ms(p: npt.NDArray[np.float64], ref: npt.NDArray[np.float64]) -> int:
    """Lag (ms) of p relative to ref; positive: p later"""
    w = np.flatnonzero((CENTERS >= SHIFT_WINDOW[0]) & (CENTERS < SHIFT_WINDOW[1]))
    lags = np.arange(-MAX_LAG_MS, MAX_LAG_MS + 1)
    r = [np.corrcoef(p[w], ref[w - lag])[0, 1] for lag in lags]
    return int(lags[int(np.nanargmax(r))])


def late_stim_frame_alignment(
    trials: pd.DataFrame, units: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(stats, arrays for plotting). `trials`: the task trials table; `units`: the units table
    with `spike_times`. Pure function of its arguments (bootstrap with a fixed seed)."""
    vis = trials[trials["is_vis_stim"].astype(bool)]
    if "is_opto" in vis:
        vis = vis[~vis["is_opto"].astype(bool)]
    vis = vis[vis["stim_start_time"].notna() & vis["quiescent_stop_time"].notna()]
    offset = 1e3 * (vis["stim_start_time"] - vis["quiescent_stop_time"]).to_numpy(float)
    stats: dict[str, Any] = {"n_vis_trials": len(vis)}
    if not len(vis):
        return {**stats, "verdict": "untestable", "reason": "no visual trials"}, {}
    median = float(np.median(offset))
    late = offset >= median + LATE_MIN_MS
    stats.update(
        median_offset_ms=median,
        n_late=int(late.sum()),
        late_extra_delay_ms=float(np.median(offset[late]) - median) if late.any() else None,
    )
    if late.sum() < MIN_LATE_TRIALS:
        return {
            **stats,
            "verdict": "untestable",
            "reason": f"{late.sum()} trials with a late stim frame (< {MIN_LATE_TRIALS})",
        }, {}
    try:
        units = units.query(UNIT_QUERY)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"unit QC query not applied: {exc!r}")
    display = vis["stim_start_time"].to_numpy(float)
    vsync = vis["quiescent_stop_time"].to_numpy(float) + median / 1e3
    spikes = [np.asarray(st, dtype=float) for st in units["spike_times"]]
    c_disp = np.array([_counts(s, display) for s in spikes])
    if not len(c_disp):
        return {**stats, "verdict": "untestable", "reason": "no units"}, {}
    base = c_disp[:, :, (CENTERS >= -0.1) & (CENTERS < 0)].sum(-1) / 0.1
    resp = c_disp[:, :, (CENTERS >= 0.02) & (CENTERS < 0.10)].sum(-1) / 0.08
    d = resp - base
    sem = d.std(axis=1, ddof=1) / np.sqrt(d.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        responsive = (d.mean(axis=1) >= 5) & (d.mean(axis=1) / sem >= 5)
    stats["n_units"] = int(responsive.sum())
    if responsive.sum() < 5:
        return {**stats, "verdict": "untestable", "reason": f"{responsive.sum()} visually responsive units (< 5)"}, {}
    c_disp = c_disp[responsive]
    c_vs = np.array([_counts(s, vsync) for s, r in zip(spikes, responsive) if r])
    rng = np.random.default_rng(SEED)
    late_idx = np.flatnonzero(late)
    arrays: dict[str, Any] = {"t_ms": CENTERS * 1e3}
    for name, c in (("display", c_disp), ("vsync", c_vs)):
        ref = _zpsth(c[:, ~late])
        p = _zpsth(c[:, late])
        boot = [
            _shift_ms(_zpsth(c[:, rng.choice(late_idx, len(late_idx))]), ref)
            for _ in range(N_BOOTSTRAP)
        ]
        stats[f"shift_vs_{name}_ms"] = _shift_ms(p, ref)
        stats[f"shift_vs_{name}_ci_ms"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        arrays[f"{name}_late"] = p
        arrays[f"{name}_normal"] = ref
    s = stats["shift_vs_display_ms"]
    extra = stats["late_extra_delay_ms"]
    ok = abs(s) <= MAX_DISPLAY_SHIFT_MS and abs(s) < abs(s + extra)
    lo, hi = stats["shift_vs_display_ci_ms"]
    if not ok and lo <= 0 <= hi and lo <= -extra <= hi:
        # CI covers both 0 (follows photodiode) and -extra (late display times): too few
        # trials to decide (e.g. 733780_2024-09-05, n=5)
        stats["verdict"] = "untestable"
        stats["reason"] = (
            f"inconclusive: late-frame trials (n={late.sum()}) respond {s:+d} ms vs normal "
            f"trials relative to stim_start_time, but the 95% CI ({lo:+.0f} to {hi:+.0f}) "
            f"contains both 0 and {-extra:+.1f} ms"
        )
        return stats, arrays
    stats["verdict"] = "passed" if ok else "failed"
    stats["reason"] = (
        f"late-frame trials (n={late.sum()}, stim frame {extra:.1f} ms later than usual "
        f"relative to vsync) respond {s:+d} ms vs normal trials relative to stim_start_time "
        f"(95% CI {stats['shift_vs_display_ci_ms'][0]:+.0f} to {stats['shift_vs_display_ci_ms'][1]:+.0f}), "
        f"{stats['shift_vs_vsync_ms']:+d} ms relative to the vsync: "
        + ("the response follows the photodiode" if ok else "display times of late frames look wrong")
    )
    return stats, arrays


def plot_late_stim_frame_alignment(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, dict[str, Any]] | None:
    """Task trials whose visual stim frame was displayed a refresh later than usual (from the
    photodiode): does the neural response follow the photodiode display time
    (`stim_start_time`) or the vsync?

    - left: population PSTH (z-scored units) of late vs normal trials aligned to
      `stim_start_time`: should overlap (shift ~0)
    - right: aligned to the vsync (`quiescent_stop_time` + median offset): late trials should
      respond ~1 refresh later

    Stats: `verdict` ('passed' if the shift relative to `stim_start_time` is within half a
    refresh of 0; 'failed'; 'untestable': < 5 late trials, < 5 responsive units, or a CI containing both outcomes), shifts with
    bootstrap CIs, numbers of trials and units.
    """
    if not (session.is_sync and session.is_sorted and session.is_task):
        return None
    trials = session.trials[:]
    if "quiescent_stop_time" not in trials:
        return None
    stats, arrays = late_stim_frame_alignment(trials, session.units[:])
    stats["session_id"] = str(session.id)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    if arrays:
        for ax, name in zip(axes, ("display", "vsync")):
            ax.plot(arrays["t_ms"], arrays[f"{name}_normal"], color="k", lw=1, label=f"normal (n={stats['n_vis_trials'] - stats['n_late']})")
            ax.plot(arrays["t_ms"], arrays[f"{name}_late"], color="tab:red", lw=1, label=f"late stim frame (n={stats['n_late']})")
            ax.axvline(0, color="0.5", lw=0.5)
            ax.set_xlim(-30, 150)
            ci = stats[f"shift_vs_{name}_ci_ms"]
            ax.set_title(
                f"aligned to {'stim_start_time (photodiode)' if name == 'display' else 'vsync (quiescent_stop_time + median offset)'}\n"
                f"late vs normal: {stats[f'shift_vs_{name}_ms']:+d} ms (95% CI {ci[0]:+.0f} to {ci[1]:+.0f})",
                fontsize=7,
            )
            ax.set_xlabel("ms from onset")
        axes[0].set_ylabel(f"population PSTH (z), {stats['n_units']} units")
        axes[0].legend(fontsize=6)
    else:
        for ax in axes:
            ax.axis("off")
    fig.suptitle(
        f"{session.id} task: late stim frames | {stats['verdict'].upper()}: {stats['reason']}",
        fontsize=7,
        color="tab:red" if stats["verdict"] == "failed" else "k",
        wrap=True,
    )
    fig.set_layout_engine("tight")
    return fig, stats
