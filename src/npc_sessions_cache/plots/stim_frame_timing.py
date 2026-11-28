"""Are the display times of the task's visual stim onset and offset frames right?

The question that matters for neural analysis is about single frames: is each trial's
`stim_start_time` / `stim_stop_time` (the display time of the stim's first frame / the first
frame after it) right? Block-level checks (frame counts, long-frame alignment, the running-median
step check in npc_stim) can't see a single frame a refresh off.

Each frame has two independent sync edges: the frame signal goes high before the frame is drawn
(rising edge) and low when `flip()` returns at the vertical blank where it was swapped in
(falling edge, the "vsync"). The photodiode display time of a normal frame is a constant
latency after both (block medians: ~28 ms after the falling edge, ~43 ms after the rising edge,
incl. `MONITOR_CENTER_REFRESH_TIME`); the rising edge only gives a lower bound (the frame signal
goes high a variable time before the vertical blank). For each onset/offset frame:

- `ok`: display - falling edge within half a refresh of the block median
- `late falling edge`: half a refresh or more *early* relative to the falling edge, with a long
  frame-signal pulse (> `npc_stim.display_times.HELD_FRAME_MAX_PULSE` refreshes: the script
  stalled after `flip()` returned, so the falling edge is late), and not earlier than the rising
  edge + the median latency (a frame can't be swapped in before its frame signal went high).
  The frame was on screen on time and the display time is right (validated against the analog
  photodiode: npc_sessions outputs/display-times-issues.md section 1a, npc_stim
  `HELD_FRAME_MAX_PULSE`)
- `unsupported`: any other frame half a refresh or more off: nothing on sync explains the
  display time. FAIL for a measured (photodiode) frame. A frame that npc_stim has already given
  vsync + latency (`frame_source` 2) is `inferred`, not a fail
- `in run`: the frame is inside a run of >= 2 consecutive frames that are off relative to the
  falling edge (a stretch of frames assigned to the wrong flips would look like this): review

Population (v0.0.303 sessions, npc_stim 0.1.35, hub experiments/timing-quartile/
pop_stim_frame_timing.py): all 361 photodiode task blocks pass (89,717 onsets + 89,698 offsets:
0 unsupported, 3 late falling edges, 0 in off runs). Before npc_stim 0.1.34, 30 RFMapping stim
onset frames in box-3 blocks (none in the task) were a refresh early (the frame after a
3-refresh hold) and would have been unsupported (audit_runs.py, compare_runs.py).
Constant-lag blocks are 'not_applicable' (display - vsync is constant by construction).
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import npc_sessions
import npc_stim
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FRAME_MS = 1e3 / 60
HALF_REFRESH_MS = FRAME_MS / 2
CLASS_COLORS = {
    "ok": "tab:green",
    "late falling edge": "tab:orange",
    "inferred": "tab:purple",
    "unsupported": "tab:red",
}


def _runs(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """(start, length) of runs of True"""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.flatnonzero(d == 1)
    return starts, np.flatnonzero(d == -1) - starts


def stim_frame_timing(
    display_times: npt.ArrayLike,
    vsync_falling: npt.ArrayLike,
    vsync_rising: npt.ArrayLike,
    onset_frames: npt.ArrayLike,
    offset_frames: npt.ArrayLike,
    frame_source: npt.ArrayLike | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(stats, arrays) for the classification in the module docstring. Pure function.

    >>> v = 100 + np.arange(1000) / 60
    >>> d = v + 0.028
    >>> d[500] += 1 / 60  # a measured onset frame a refresh late
    >>> stats, _ = stim_frame_timing(d, v, v - 0.016, [100, 500], [130, 530])
    >>> stats["verdict"], stats["n_unsupported"]
    ('failed', 1)
    """
    display = np.asarray(display_times, dtype=float)
    fall = np.asarray(vsync_falling, dtype=float)
    rise = np.asarray(vsync_rising, dtype=float)
    src = (
        np.zeros(len(display), dtype=np.int8)
        if frame_source is None
        else np.asarray(frame_source, dtype=np.int8)
    )
    lat_fall = 1e3 * (display - fall)
    lat_rise = 1e3 * (display - rise)
    pulse = 1e3 * (fall - rise)
    med_fall = float(np.median(lat_fall))
    d_fall = lat_fall - med_fall
    long_pulse = pulse > npc_stim.display_times.HELD_FRAME_MAX_PULSE * FRAME_MS
    after_rise = lat_rise >= med_fall - HALF_REFRESH_MS
    off = np.abs(d_fall) >= HALF_REFRESH_MS
    starts, lengths = _runs(off)
    in_run = np.zeros(len(display), dtype=bool)
    for s, n in zip(starts, lengths):
        if n >= 2:
            in_run[s : s + n] = True
    on_f = np.asarray(onset_frames, dtype=int)
    off_f = np.asarray(offset_frames, dtype=int)
    frames = np.concatenate([on_f, off_f])
    is_onset = np.concatenate([np.ones(len(on_f), bool), np.zeros(len(off_f), bool)])
    keep = (frames >= 0) & (frames < len(display))
    frames, is_onset = frames[keep], is_onset[keep]
    cls = np.full(len(frames), "ok", dtype=object)
    f_off = off[frames]
    late_fall = (d_fall[frames] < 0) & long_pulse[frames] & after_rise[frames]
    cls[f_off & late_fall] = "late falling edge"
    cls[f_off & ~late_fall] = "unsupported"
    cls[(cls == "unsupported") & (src[frames] != 0)] = "inferred"
    events = [
        {
            "frame": int(f),
            "kind": "onset" if o else "offset",
            "class": str(c),
            "in_run": bool(in_run[f]),
            "display_minus_fall_dev_ms": round(float(d_fall[f]), 2),
            "display_minus_rise_ms": round(float(lat_rise[f]), 2),
            "frame_signal_pulse_ms": round(float(pulse[f]), 2),
            "frame_source": int(src[f]),
        }
        for f, o, c in zip(frames, is_onset, cls)
        if c != "ok" or in_run[f]
    ]
    n_unsupported = int((cls == "unsupported").sum())
    n_in_run = int(in_run[frames].sum())
    n_inferred = int((cls == "inferred").sum())
    if n_unsupported:
        verdict = "failed"
    elif n_in_run or n_inferred:
        verdict = "review"
    else:
        verdict = "passed"
    stats = {
        "verdict": verdict,
        "n_frames": len(display),
        "n_onset_frames": int(is_onset.sum()),
        "n_offset_frames": int((~is_onset).sum()),
        "median_display_minus_fall_ms": float(np.median(lat_fall)),
        "median_frame_signal_pulse_ms": float(np.median(pulse[np.isfinite(pulse)])),
        "n_unsupported": n_unsupported,
        "n_late_falling_edge": int((cls == "late falling edge").sum()),
        "n_inferred": n_inferred,
        "n_events_in_off_run": n_in_run,
        "n_off_frames_in_block": int(off.sum()),
        "longest_off_run": int(lengths.max()) if len(lengths) else 0,
        "events": events,
    }
    arrays = {
        "frames": frames,
        "is_onset": is_onset,
        "cls": cls,
        "d_fall": d_fall,
        "pulse": pulse,
        "off": off,
        "display": display,
    }
    return stats, arrays


def get_task_vis_stim_frames(stim_path: Any) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:  # noqa: ANN401
    """Onset frames of the task's visual stims and the first frame after each (offset), from
    the stim file (as `stim_start_time` / `stim_stop_time` in the trials table)."""
    h = npc_stim.stim.get_h5_stim_data(stim_path)
    try:
        start = h["trialStimStartFrame"][()].astype(int)
        stim = h["trialStim"].asstr()[()][: len(start)]
        n = h["trialVisStimFrames"][()][: len(start)].astype(int)
    finally:
        h.close()
    vis = np.array(["vis" in s for s in stim], dtype=bool)
    return start[vis], (start + n)[vis]


def plot_task_stim_frame_timing(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, dict[str, Any]] | None:
    """Task block: is the display time of every visual stim onset and offset frame supported by
    that frame's own sync edges? See the module docstring for the classes.

    - top: every frame of the block whose display - vsync latency is half a refresh or more from
      the median (grey), and the onset/offset frames that aren't `ok` (coloured), vs time
    - bottom left: onset/offset frames, display - falling edge (minus the block median, x) vs
      frame-signal pulse length (y). `ok` frames are in the grey band; late falling edges are
      left of it, above the dashed line, near the dotted diagonal; anything else outside the band
      is `unsupported`
    - bottom right: histogram of the onset/offset frames' latency relative to the falling edge

    Stats: `verdict` ('failed': >= 1 unsupported onset/offset frame; 'review': one in a run of
    >= 2 off frames, or given vsync + latency by npc_stim; 'passed'), counts, and every
    onset/offset frame that isn't `ok`.
    """
    from npc_sessions_cache.plots.display_qc import get_display_block

    if not (session.is_sync and session.is_task):
        return None
    stem = session.task_path.stem
    block = get_display_block(session, stem)
    if isinstance(block, Exception) or block.display_times is None:
        raise ValueError(f"{session.id}: no display times for {stem}: {block!r}")
    stim_path = next(p for p in session.stim_paths if p.stem == stem)
    onsets, offsets = get_task_vis_stim_frames(stim_path)
    rising = npc_stim.display_times.get_vsync_rising_edges(session.sync_data, block.vsync_times)
    stats, a = stim_frame_timing(
        block.display_times, block.vsync_times, rising, onsets, offsets, block.frame_source
    )
    stats = {"session_id": str(session.id), "stim": stem, "display_time_source": block.source, **stats}
    if block.source != "photodiode":
        # vsync + constant lag: display - vsync is constant by construction, nothing to check
        # here (validity: npc_sessions.qc, see quartile_visual_latency)
        stats["verdict"] = "not_applicable"

    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_t = fig.add_subplot(gs[0, :])
    ax_s = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1])
    t_min = (a["display"] - a["display"][0]) / 60
    off = a["off"]
    ax_t.plot(t_min[off], a["d_fall"][off], ".", ms=3, color="0.6", label=f"off frames in block ({off.sum()})")
    for c, color in CLASS_COLORS.items():
        sel = a["cls"] == c
        if c == "ok" or not sel.any():
            continue
        f = a["frames"][sel]
        ax_t.plot(t_min[f], a["d_fall"][f], "o", ms=5, mfc="none", color=color, label=f"{c} onset/offset ({sel.sum()})")
    ax_t.axhspan(-HALF_REFRESH_MS, HALF_REFRESH_MS, color="0.92", zorder=-1)
    for k in (-2, -1, 1, 2):
        ax_t.axhline(k * FRAME_MS, color="0.7", lw=0.5, ls=":")
    ax_t.set_xlabel("minutes into block")
    ax_t.set_ylabel("display - vsync falling edge\nminus block median (ms)")
    ax_t.legend(fontsize=6, loc="upper right")
    ax_t.set_title("frames off by >= 1/2 refresh (band: within 1/2 refresh; dotted: +-1, 2 refreshes)", fontsize=7)

    f = a["frames"]
    for c, color in CLASS_COLORS.items():
        sel = a["cls"] == c
        if sel.any():
            ax_s.plot(a["d_fall"][f[sel]], a["pulse"][f[sel]], "o", ms=3 if c == "ok" else 5, color=color, alpha=0.6 if c == "ok" else 1, label=f"{c} ({sel.sum()})")
    ax_s.axvspan(-HALF_REFRESH_MS, HALF_REFRESH_MS, color="0.92", zorder=-1)
    ax_s.axhline(npc_stim.display_times.HELD_FRAME_MAX_PULSE * FRAME_MS, color="k", lw=0.5, ls="--")
    pulses = np.linspace(FRAME_MS, max(3 * FRAME_MS, np.nanmax(a["pulse"][f])), 50)
    ax_s.plot(-(pulses - np.nanmedian(a["pulse"][f])), pulses, color="tab:orange", lw=0.5, ls=":")
    ax_s.set_xlabel("display - falling edge - block median (ms)")
    ax_s.set_ylabel("frame-signal pulse: falling - rising edge (ms)")
    ax_s.legend(fontsize=6)
    ax_s.set_title(
        "visual stim onset/offset frames (grey band: ok; dashed: long pulse = late falling edge;\n"
        "dotted: where a late falling edge puts a frame displayed on time)",
        fontsize=7,
    )

    ax_h.hist(a["d_fall"][f], bins=np.arange(-40.5, 41, 1), color="tab:blue", log=True)
    for x in (-HALF_REFRESH_MS, HALF_REFRESH_MS):
        ax_h.axvline(x, color="k", lw=0.5, ls="--")
    ax_h.set_xlabel("onset/offset frames: display - falling edge - median (ms)")
    ax_h.set_ylabel("frames")

    fig.suptitle(
        f"{session.id} {stem} | display times: {block.source} | {stats['verdict'].upper()}\n"
        f"{stats['n_onset_frames']} onsets + {stats['n_offset_frames']} offsets: "
        f"{stats['n_unsupported']} unsupported, {stats['n_late_falling_edge']} late falling edge (ok), "
        f"{stats['n_inferred']} vsync + latency, {stats['n_events_in_off_run']} in a run of >= 2 off frames; "
        f"longest off run in block: {stats['longest_off_run']} frames",
        fontsize=7.5,
        color="tab:red" if stats["verdict"] == "failed" else "k",
    )
    fig.set_layout_engine("tight")
    return fig, stats
